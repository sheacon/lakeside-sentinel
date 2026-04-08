"""Fine-tune YOLO on the collected review dataset with a reproducible stratified val split."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from lakeside_sentinel.review.fine_tuning import FINE_TUNING_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("output/fine-tuning")
DEFAULT_BASE_MODEL = "yolo_models/yolo26s.pt"
DEFAULT_PROJECT = Path("yolo_models/finetuned")
MIN_TRAIN_PER_CLASS_WARNING = 5
MIN_VAL_PER_CLASS_WARNING = 2


def _detect_device() -> str:
    """Auto-detect best available torch device, matching veh_detector.py."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _read_single_label_class(label_path: Path) -> int | None:
    """Read a YOLO label file and return its (single) class id.

    All label files in the fine-tuning dataset contain exactly one annotation.
    Returns None if the file is empty or unreadable.
    """
    try:
        line = label_path.read_text().strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    try:
        return int(line.split()[0])
    except (ValueError, IndexError):
        return None


def _collect_dataset(data_dir: Path) -> dict[int, list[str]]:
    """Walk the dataset and group image stems by class id.

    Returns:
        Mapping of class id -> sorted list of image stems (without extension).
    """
    images_dir = data_dir / "images" / "train"
    labels_dir = data_dir / "labels" / "train"

    by_class: dict[int, list[str]] = defaultdict(list)
    for image_path in sorted(images_dir.glob("*.jpg")):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            logger.warning("Image %s has no label file, skipping", stem)
            continue
        cls = _read_single_label_class(label_path)
        if cls is None:
            logger.warning("Could not parse class for %s, skipping", stem)
            continue
        by_class[cls].append(stem)

    return dict(by_class)


def _stratified_split(
    by_class: dict[int, list[str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split image stems into train and val, stratified per class.

    Each class contributes `floor(val_fraction * count)` stems to val, with a
    minimum of 1 when the class has 2 or more images. This protects tiny
    classes like motorbike (20 images) from landing entirely in train or val.
    """
    rng = random.Random(seed)
    train_stems: list[str] = []
    val_stems: list[str] = []

    for cls in sorted(by_class):
        stems = sorted(by_class[cls])
        rng.shuffle(stems)
        n_val = int(val_fraction * len(stems))
        if n_val == 0 and len(stems) >= 2:
            n_val = 1
        val_stems.extend(stems[:n_val])
        train_stems.extend(stems[n_val:])

    val_stems.sort()
    train_stems.sort()
    return train_stems, val_stems


def _ensure_val_split(
    data_dir: Path,
    val_fraction: float,
    seed: int,
    force_resplit: bool,
) -> tuple[list[str], list[str], dict[int, list[str]]]:
    """Load or create a persisted stratified val split.

    Returns:
        (train_stems, val_stems, by_class) — the full per-class mapping is
        returned so the caller can print a split summary.
    """
    split_path = data_dir / "val_split.json"
    by_class = _collect_dataset(data_dir)
    if not by_class:
        raise SystemExit(f"No labeled images found under {data_dir / 'images' / 'train'}")

    if split_path.exists() and not force_resplit:
        payload = json.loads(split_path.read_text())
        val_stems = sorted(payload["val_stems"])
        all_stems = {stem for stems in by_class.values() for stem in stems}
        val_set = set(val_stems)
        missing = val_set - all_stems
        if missing:
            logger.warning(
                "Persisted val split references %d missing images; regenerating", len(missing)
            )
        else:
            train_stems = sorted(all_stems - val_set)
            logger.info(
                "Loaded persisted val split from %s (seed=%d, val_fraction=%.2f)",
                split_path,
                payload.get("seed", seed),
                payload.get("val_fraction", val_fraction),
            )
            return train_stems, val_stems, by_class

    train_stems, val_stems = _stratified_split(by_class, val_fraction, seed)
    split_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "val_fraction": val_fraction,
                "val_stems": val_stems,
            },
            indent=2,
        )
    )
    logger.info("Wrote new val split to %s", split_path)
    return train_stems, val_stems, by_class


def _write_split_files(
    data_dir: Path,
    train_stems: list[str],
    val_stems: list[str],
) -> tuple[Path, Path]:
    """Write ultralytics-compatible train/val file lists.

    Paths are absolute because ultralytics' BaseDataset.get_img_files only
    rewrites lines that start with "./"; bare relative paths get kept verbatim
    and fail to open once ultralytics changes CWD into its runs directory.
    """
    images_dir = (data_dir / "images" / "train").resolve()
    train_list = data_dir / "train_split.txt"
    val_list = data_dir / "val_split.txt"
    train_list.write_text("\n".join(str(images_dir / f"{stem}.jpg") for stem in train_stems) + "\n")
    val_list.write_text("\n".join(str(images_dir / f"{stem}.jpg") for stem in val_stems) + "\n")
    return train_list, val_list


def _write_data_yaml(data_dir: Path) -> Path:
    """Write the split-aware data.yaml and return its path."""
    # Invert FINE_TUNING_CLASSES {name: id} -> {id: name}.
    id_to_name = {cls_id: name for name, cls_id in FINE_TUNING_CLASSES.items()}
    names = {cls_id: id_to_name[cls_id] for cls_id in sorted(id_to_name)}

    payload = {
        "path": str(data_dir.resolve()),
        "train": "train_split.txt",
        "val": "val_split.txt",
        "names": names,
    }
    yaml_path = data_dir / "data_split.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return yaml_path


def _print_split_summary(
    by_class: dict[int, list[str]],
    train_stems: list[str],
    val_stems: list[str],
) -> None:
    """Print a per-class train/val count table and warn on thin classes."""
    id_to_name = {cls_id: name for name, cls_id in FINE_TUNING_CLASSES.items()}
    train_set = set(train_stems)
    val_set = set(val_stems)

    header = f"\n{'Class':<10} | {'Total':>5} | {'Train':>5} | {'Val':>5}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    warnings: list[str] = []
    for cls_id in sorted(by_class):
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        stems = by_class[cls_id]
        n_train = sum(1 for s in stems if s in train_set)
        n_val = sum(1 for s in stems if s in val_set)
        print(f"{name:<10} | {len(stems):>5} | {n_train:>5} | {n_val:>5}")
        if n_train < MIN_TRAIN_PER_CLASS_WARNING:
            warnings.append(f"{name}: only {n_train} train images")
        if n_val < MIN_VAL_PER_CLASS_WARNING:
            warnings.append(f"{name}: only {n_val} val images")

    total_train = len(train_stems)
    total_val = len(val_stems)
    print(sep)
    print(f"{'TOTAL':<10} | {total_train + total_val:>5} | {total_train:>5} | {total_val:>5}\n")

    for warning in warnings:
        logger.warning("Thin class: %s", warning)


def _print_train_summary(results: object, weights_dir: Path, elapsed_secs: float) -> None:
    """Print a final summary of the training run."""
    # ultralytics .train() returns a DetMetrics-like object with .results_dict
    metrics: dict[str, float] = {}
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics = {k: float(v) for k, v in results_dict.items() if isinstance(v, (int, float))}

    header = f"\n{'Metric':<25} | {'Value':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    def _emit(label: str, key: str) -> None:
        if key in metrics:
            print(f"{label:<25} | {metrics[key]:>12.4f}")

    _emit("mAP50", "metrics/mAP50(B)")
    _emit("mAP50-95", "metrics/mAP50-95(B)")
    _emit("precision", "metrics/precision(B)")
    _emit("recall", "metrics/recall(B)")
    print(sep)
    print(f"{'Training time (s)':<25} | {elapsed_secs:>12.1f}")
    print(f"\nWeights written to: {weights_dir}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO on the review-collected dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default run (50 epochs, low LR, frozen backbone)\n"
            "  uv run python scripts/finetune.py\n\n"
            "  # Smoke test the split logic\n"
            "  uv run python scripts/finetune.py --epochs 1\n\n"
            "  # Force a fresh val split\n"
            "  uv run python scripts/finetune.py --force-resplit\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root of the fine-tuning dataset (default: output/fine-tuning)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Starting YOLO weights",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run subdirectory name (default: ft-YYYYMMDD-HHMM)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (matches production)")
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (low to avoid catastrophic forgetting on motorbike)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze first N backbone layers (helps preserve motorbike recall)",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early-stop patience")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Split + training seed")
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Regenerate val_split.json even if it already exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (mps/cpu/cuda). Default: auto-detect.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        raise SystemExit(1)

    device = args.device or _detect_device()
    logger.info("Using device: %s", device)

    # 1. Build or load the stratified split.
    train_stems, val_stems, by_class = _ensure_val_split(
        data_dir=data_dir,
        val_fraction=args.val_split,
        seed=args.seed,
        force_resplit=args.force_resplit,
    )

    # 2. Always regenerate the file lists and data yaml so they match the loaded split.
    _write_split_files(data_dir, train_stems, val_stems)
    data_yaml = _write_data_yaml(data_dir)

    # 3. Show the user exactly what they're about to train on.
    _print_split_summary(by_class, train_stems, val_stems)

    # 4. Train.
    run_name = args.name or f"ft-{datetime.now().strftime('%Y%m%d-%H%M')}"
    logger.info(
        "Training %s on %d images (val=%d) for %d epochs",
        args.base_model,
        len(train_stems),
        len(val_stems),
        args.epochs,
    )

    model = YOLO(args.base_model)
    start = datetime.now()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        freeze=args.freeze,
        patience=args.patience,
        device=device,
        seed=args.seed,
        # Absolute path: ultralytics treats a relative project as a child of its
        # own runs/detect/ dir, which buries outputs under runs/detect/yolo_models/...
        project=str(DEFAULT_PROJECT.resolve()),
        name=run_name,
        exist_ok=False,
        # MPS on macOS forces dataloader workers=0, so per-batch disk I/O
        # dominates wall time. Caching the whole (~53 MB) dataset in RAM
        # eliminates the stall.
        cache="ram",
        # ROI crops are ~1287x302 (~4.25:1); square letterboxing at imgsz=1280
        # wastes most of every forward pass on padding. Rectangular training
        # processes batches at the natural aspect ratio. Auto-disables mosaic.
        rect=True,
        # optimizer='auto' silently overrides lr0 and momentum. Pin AdamW so our
        # explicit low LR (to protect the 16-image motorbike class) is honored.
        optimizer="AdamW",
    )
    elapsed = (datetime.now() - start).total_seconds()

    weights_dir = DEFAULT_PROJECT.resolve() / run_name / "weights"
    _print_train_summary(results, weights_dir, elapsed)


if __name__ == "__main__":
    main()
