"""Post-training evaluation for a fine-tuned YOLO model.

Runs two passes:
  1. Fine-tuned model only: full ultralytics .val() metrics (P/R/mAP per class).
  2. Fine-tuned vs baseline: confusion-matrix comparison at production conf=0.4,
     focused on the stroller<->motorbike confusion.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from lakeside_sentinel.review.fine_tuning import FINE_TUNING_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("output/fine-tuning")
DEFAULT_BASELINE = "yolo_models/yolo26s.pt"
DEFAULT_DATA_YAML = DEFAULT_DATA_DIR / "data_split.yaml"

# COCO class ids used by the base model. Must match veh_detector.VEH_CLASSES.
COCO_VEH_CLASSES: dict[int, str] = {
    1: "bicycle",
    3: "motorbike",
}

# Sentinel prediction value when no detection is made at the production threshold.
NO_DETECTION = "none"


def _detect_device() -> str:
    """Auto-detect best available torch device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_imgsz(frame_shape: tuple[int, ...], target_width: int = 1280) -> tuple[int, int]:
    """Compute YOLO imgsz preserving aspect ratio, rounded to multiples of 32.

    Mirrors VEHDetector._compute_imgsz so inference matches production.
    """
    h, w = frame_shape[:2]
    aspect = h / w
    target_height = int(target_width * aspect)
    target_height = max(32, round(target_height / 32) * 32)
    target_width = max(32, round(target_width / 32) * 32)
    return (target_height, target_width)


def _read_val_stems(data_dir: Path) -> list[str]:
    """Read the val_split.txt file list and return image stems."""
    val_list = data_dir / "val_split.txt"
    if not val_list.exists():
        raise SystemExit(
            f"Val split not found at {val_list}. Run scripts/finetune.py first to create it."
        )
    stems: list[str] = []
    for line in val_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stems.append(Path(line).stem)
    return stems


def _read_ground_truth(data_dir: Path, stems: list[str]) -> dict[str, str]:
    """Map image stem -> ground-truth class name (from the label file's first line)."""
    id_to_name = {cls_id: name for name, cls_id in FINE_TUNING_CLASSES.items()}
    labels_dir = data_dir / "labels" / "train"

    gt: dict[str, str] = {}
    for stem in stems:
        label_path = labels_dir / f"{stem}.txt"
        try:
            line = label_path.read_text().strip().splitlines()[0]
            cls_id = int(line.split()[0])
            gt[stem] = id_to_name.get(cls_id, f"class_{cls_id}")
        except (OSError, IndexError, ValueError):
            logger.warning("Could not read ground truth for %s", stem)
            gt[stem] = "unknown"
    return gt


def _run_val_metrics(model_path: str, data_yaml: Path, imgsz: int, device: str) -> None:
    """Run ultralytics .val() on the fine-tuned model and print a per-class table."""
    print("\n" + "=" * 70)
    print(f"PASS 1 — Fine-tuned model metrics: {model_path}")
    print("=" * 70)

    model = YOLO(model_path)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        split="val",
        verbose=False,
    )

    # metrics.box has per-class arrays; metrics.names is the class id -> name map.
    names: dict[int, str] = dict(metrics.names) if hasattr(metrics, "names") else {}
    box = metrics.box
    per_class_p = list(box.p) if hasattr(box, "p") else []
    per_class_r = list(box.r) if hasattr(box, "r") else []
    per_class_map50 = list(box.ap50) if hasattr(box, "ap50") else []
    per_class_map = list(box.ap) if hasattr(box, "ap") else []
    class_indices = list(box.ap_class_index) if hasattr(box, "ap_class_index") else []

    header = f"\n{'Class':<12} | {'P':>6} | {'R':>6} | {'mAP50':>7} | {'mAP50-95':>9}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for i, cls_id in enumerate(class_indices):
        name = names.get(int(cls_id), f"class_{int(cls_id)}")
        p = per_class_p[i] if i < len(per_class_p) else float("nan")
        r = per_class_r[i] if i < len(per_class_r) else float("nan")
        m50 = per_class_map50[i] if i < len(per_class_map50) else float("nan")
        m = per_class_map[i] if i < len(per_class_map) else float("nan")
        print(f"{name:<12} | {p:>6.3f} | {r:>6.3f} | {m50:>7.3f} | {m:>9.3f}")

    print(sep)
    overall_map50 = float(box.map50) if hasattr(box, "map50") else float("nan")
    overall_map = float(box.map) if hasattr(box, "map") else float("nan")
    print(f"{'ALL':<12} | {'':>6} | {'':>6} | {overall_map50:>7.3f} | {overall_map:>9.3f}\n")


def _predict_best(
    model: YOLO,
    image_path: Path,
    conf: float,
    imgsz: tuple[int, int],
    device: str,
    class_filter: set[int] | None,
    class_names: dict[int, str],
) -> str:
    """Run inference on a single image and return the predicted class name.

    Args:
        model: Loaded YOLO model.
        image_path: Path to the image file.
        conf: Confidence threshold.
        imgsz: (height, width) inference size.
        device: Torch device.
        class_filter: Only consider detections whose class id is in this set.
            None means no filter (accept any class the model outputs).
        class_names: Map of class id -> name for translating the winning class.

    Returns:
        The class name of the highest-confidence detection above `conf`, or
        NO_DETECTION if no qualifying box was produced.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.warning("Could not read %s", image_path)
        return NO_DETECTION

    results = model(
        [frame],
        verbose=False,
        imgsz=imgsz,
        device=device,
        conf=conf,
    )
    if not results:
        return NO_DETECTION

    best_conf = -1.0
    best_name = NO_DETECTION
    for box in results[0].boxes:
        cls = int(box.cls[0])
        c = float(box.conf[0])
        if class_filter is not None and cls not in class_filter:
            continue
        if c > best_conf:
            best_conf = c
            best_name = class_names.get(cls, f"class_{cls}")

    return best_name


def _build_confusion(
    gt: dict[str, str],
    predictions: dict[str, str],
    row_classes: list[str],
    col_classes: list[str],
) -> list[list[int]]:
    """Build a confusion matrix indexed as [row=gt][col=pred]."""
    row_index = {name: i for i, name in enumerate(row_classes)}
    col_index = {name: i for i, name in enumerate(col_classes)}
    matrix = [[0 for _ in col_classes] for _ in row_classes]

    for stem, gt_name in gt.items():
        if gt_name not in row_index:
            continue
        pred_name = predictions.get(stem, NO_DETECTION)
        col_idx = col_index.get(pred_name)
        if col_idx is None:
            col_idx = col_index[NO_DETECTION]
        matrix[row_index[gt_name]][col_idx] += 1

    return matrix


def _print_confusion(
    title: str,
    matrix: list[list[int]],
    row_classes: list[str],
    col_classes: list[str],
) -> None:
    """Print a confusion matrix to stdout."""
    col_width = max(6, max(len(c) for c in col_classes) + 1)
    row_width = max(10, max(len(r) for r in row_classes))

    print(f"\n{title}")
    header = f"{'gt \\ pred':<{row_width}}"
    for col in col_classes:
        header += f" | {col:>{col_width}}"
    print(header)
    print("-" * len(header))

    for i, row in enumerate(row_classes):
        line = f"{row:<{row_width}}"
        for val in matrix[i]:
            line += f" | {val:>{col_width}}"
        print(line)


def _print_headline_metrics(
    title: str,
    gt: dict[str, str],
    predictions: dict[str, str],
) -> None:
    """Print the two metrics the user actually cares about."""
    n_motorbike_gt = sum(1 for v in gt.values() if v == "motorbike")
    n_motorbike_hit = sum(
        1 for stem, g in gt.items() if g == "motorbike" and predictions.get(stem) == "motorbike"
    )

    n_stroller_gt = sum(1 for v in gt.values() if v == "stroller")
    n_stroller_as_motorbike = sum(
        1 for stem, g in gt.items() if g == "stroller" and predictions.get(stem) == "motorbike"
    )

    def _pct(num: int, denom: int) -> str:
        if denom == 0:
            return "n/a"
        return f"{num}/{denom} = {100 * num / denom:.1f}%"

    print(f"\n{title}")
    print(f"  Motorbike recall            : {_pct(n_motorbike_hit, n_motorbike_gt)}")
    print(f"  Stroller -> motorbike FPR   : {_pct(n_stroller_as_motorbike, n_stroller_gt)}")


def _run_confusion_comparison(
    ft_model_path: str,
    baseline_model_path: str,
    data_dir: Path,
    conf: float,
    imgsz_target: int,
    device: str,
) -> None:
    """Pass 2 — confusion-matrix comparison at prod conf for both models."""
    print("\n" + "=" * 70)
    print(f"PASS 2 — Confusion matrix comparison at conf={conf}")
    print("=" * 70)

    val_stems = _read_val_stems(data_dir)
    gt = _read_ground_truth(data_dir, val_stems)
    images_dir = data_dir / "images" / "train"
    logger.info("Running inference on %d val images", len(val_stems))

    # Probe image size from the first readable image so imgsz matches production sizing.
    first_frame = cv2.imread(str(images_dir / f"{val_stems[0]}.jpg"))
    if first_frame is None:
        raise SystemExit(f"Could not read first val image {val_stems[0]}")
    imgsz = _compute_imgsz(first_frame.shape, target_width=imgsz_target)

    ft_model = YOLO(ft_model_path)
    baseline_model = YOLO(baseline_model_path)

    # Use the fine-tuned model's own class names so predictions print consistently.
    ft_names: dict[int, str] = dict(ft_model.names) if hasattr(ft_model, "names") else {}

    ft_preds: dict[str, str] = {}
    base_preds: dict[str, str] = {}

    for i, stem in enumerate(val_stems, start=1):
        image_path = images_dir / f"{stem}.jpg"
        ft_preds[stem] = _predict_best(
            model=ft_model,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            class_filter=None,
            class_names=ft_names,
        )
        base_preds[stem] = _predict_best(
            model=baseline_model,
            image_path=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            class_filter=set(COCO_VEH_CLASSES),
            class_names=COCO_VEH_CLASSES,
        )
        if i % 25 == 0:
            logger.info("Processed %d/%d images", i, len(val_stems))

    # Build confusion matrices using the custom class set as rows.
    id_to_name = {cls_id: name for name, cls_id in FINE_TUNING_CLASSES.items()}
    row_classes = [id_to_name[cls_id] for cls_id in sorted(id_to_name)]

    # Columns always include NO_DETECTION so "missed" predictions are visible.
    # For the fine-tuned model the prediction space is the full 7-class set.
    ft_col_classes = row_classes + [NO_DETECTION]
    ft_matrix = _build_confusion(gt, ft_preds, row_classes, ft_col_classes)
    _print_confusion(
        f"Fine-tuned ({Path(ft_model_path).name})",
        ft_matrix,
        row_classes,
        ft_col_classes,
    )

    # The baseline model can only emit bicycle or motorbike (we filtered everything else).
    base_col_classes = ["bicycle", "motorbike", NO_DETECTION]
    base_matrix = _build_confusion(gt, base_preds, row_classes, base_col_classes)
    _print_confusion(
        f"Baseline ({Path(baseline_model_path).name}, COCO-filtered to bicycle/motorbike)",
        base_matrix,
        row_classes,
        base_col_classes,
    )

    _print_headline_metrics("Headline metrics — fine-tuned:", gt, ft_preds)
    _print_headline_metrics("Headline metrics — baseline:", gt, base_preds)
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned YOLO model against the baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/evaluate_model.py \\\n"
            "      --model yolo_models/finetuned/ft-20260407-1030/weights/best.pt\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned .pt weights",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE,
        help="Baseline model for comparison (default: yolo_models/yolo26s.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root of the fine-tuning dataset (default: output/fine-tuning)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Split-aware data yaml (default: output/fine-tuning/data_split.yaml)",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image width")
    parser.add_argument("--conf", type=float, default=0.4, help="Production confidence threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (mps/cpu/cuda). Default: auto-detect.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip pass 1 (ultralytics .val metrics) and only run the comparison",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error("Fine-tuned model not found: %s", args.model)
        raise SystemExit(1)
    if not Path(args.baseline).exists():
        logger.error("Baseline model not found: %s", args.baseline)
        raise SystemExit(1)
    if not args.data_yaml.exists():
        logger.error(
            "Data yaml not found: %s (run scripts/finetune.py first)",
            args.data_yaml,
        )
        raise SystemExit(1)

    device = args.device or _detect_device()
    logger.info("Using device: %s", device)

    if not args.skip_val:
        _run_val_metrics(
            model_path=args.model,
            data_yaml=args.data_yaml,
            imgsz=args.imgsz,
            device=device,
        )

    _run_confusion_comparison(
        ft_model_path=args.model,
        baseline_model_path=args.baseline,
        data_dir=args.data_dir,
        conf=args.conf,
        imgsz_target=args.imgsz,
        device=device,
    )


if __name__ == "__main__":
    main()
