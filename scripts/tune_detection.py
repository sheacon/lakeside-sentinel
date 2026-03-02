"""Detection tuning harness — sweeps model/FPS combos to find missed detections."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.detection.vehicle_detector import VehicleDetector
from lakeside_sentinel.utils.image import crop_to_roi
from lakeside_sentinel.utils.video import extract_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Fixed ROI — top-right region where the motorbike appears
ROI_Y_START = 0.0
ROI_Y_END = 0.28
ROI_X_START = 0.33
ROI_X_END = 1.0


@dataclass(frozen=True)
class RunConfig:
    """Single tuning run configuration."""

    run_id: int
    model_name: str
    fps_sample: int


@dataclass
class RunResult:
    """Result of a single tuning run."""

    config: RunConfig
    best: Detection | None
    class_best: dict[str, Detection]
    frame_count: int
    elapsed_secs: float


# fmt: off
CONFIGS: list[RunConfig] = [
    RunConfig(1,  "yolo26s.pt", 2),   # Baseline (current production)
    RunConfig(2,  "yolo26s.pt", 4),   # More frames, same model
    RunConfig(3,  "yolo26s.pt", 8),   # Max frame density, same model
    RunConfig(4,  "yolo11s.pt", 2),   # Different architecture
    RunConfig(5,  "yolo11s.pt", 4),   # Different architecture + more frames
    RunConfig(6,  "yolo26m.pt", 2),   # Larger, more accurate model
    RunConfig(7,  "yolo26m.pt", 4),   # Larger model + more frames
    RunConfig(8,  "yolo26m.pt", 8),   # Larger model + max frame density
    RunConfig(9,  "yolo11n.pt", 8),   # Nano model at max fps (speed baseline)
]
# fmt: on


def annotate_frame(frame: np.ndarray, class_best: dict[str, Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame for the best detection per class."""
    annotated = frame.copy()

    colors: dict[str, tuple[int, int, int]] = {
        "Bicycle": (0, 255, 0),  # green
        "Motorcycle": (0, 0, 255),  # red
    }

    for class_name, det in class_best.items():
        color = colors.get(class_name, (255, 255, 0))
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {det.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background rectangle for text
        cv2.rectangle(annotated, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return annotated


def run_single(
    config: RunConfig,
    frames_cache: dict[int, list[np.ndarray]],
    model_cache: dict[str, VehicleDetector],
) -> RunResult:
    """Execute a single tuning run."""
    frames = frames_cache[config.fps_sample]
    detector = model_cache[config.model_name]

    t0 = time.perf_counter()
    best, class_best = detector.detect_detailed(frames)
    elapsed = time.perf_counter() - t0

    return RunResult(
        config=config,
        best=best,
        class_best=class_best,
        frame_count=len(frames),
        elapsed_secs=elapsed,
    )


def print_results_table(results: list[RunResult]) -> None:
    """Print a formatted results table to stdout."""
    header = (
        f"{'Run':>3} | {'Model':<10} | {'FPS':>3} | {'Frames':>6} "
        f"| {'Best':<12} | {'Conf':>5} | {'Sub-threshold':<25} | {'Time':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for r in results:
        best_label = r.best.class_name.upper() if r.best else "-"
        best_conf = f"{r.best.confidence:.0%}" if r.best else "-"

        sub_parts: list[str] = []
        for cls_name, det in sorted(r.class_best.items()):
            sub_parts.append(f"{cls_name}: {det.confidence:.0%}")
        sub_threshold = ", ".join(sub_parts) if sub_parts else "-"

        model_short = r.config.model_name.replace(".pt", "")
        print(
            f"{r.config.run_id:>3} | {model_short:<10} | {r.config.fps_sample:>3} "
            f"| {r.frame_count:>6} | {best_label:<12} | {best_conf:>5} "
            f"| {sub_threshold:<25} | {r.elapsed_secs:>5.1f}s"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep detection parameters on a video clip.")
    parser.add_argument(
        "--clip",
        type=Path,
        default=Path("output/backfill/2026-02-28_12-31-14.mp4"),
        help="Path to MP4 clip to analyze",
    )
    args = parser.parse_args()

    clip_path: Path = args.clip
    if not clip_path.exists():
        logger.error("Clip not found: %s", clip_path)
        raise SystemExit(1)

    mp4_bytes = clip_path.read_bytes()
    clip_stem = clip_path.stem
    output_dir = Path("output/tune") / clip_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache frames by fps_sample (extract once, crop ROI once per fps value)
    unique_fps = sorted({c.fps_sample for c in CONFIGS})
    logger.info("Extracting frames for FPS values: %s", unique_fps)

    frames_cache: dict[int, list[np.ndarray]] = {}
    for fps in unique_fps:
        raw_frames = extract_frames(mp4_bytes, fps_sample=fps)
        cropped = crop_to_roi(
            raw_frames,
            y_start=ROI_Y_START,
            y_end=ROI_Y_END,
            x_start=ROI_X_START,
            x_end=ROI_X_END,
        )
        frames_cache[fps] = cropped
        shape = cropped[0].shape[:2] if cropped else "N/A"
        logger.info(
            "FPS %d: %d frames extracted, ROI-cropped to %s",
            fps,
            len(cropped),
            shape,
        )

    # Cache models (load once per unique model name)
    unique_models = sorted({c.model_name for c in CONFIGS})
    logger.info("Loading models: %s", unique_models)

    model_cache: dict[str, VehicleDetector] = {}
    for model_name in unique_models:
        model_cache[model_name] = VehicleDetector(model_name=model_name)

    # Run all configs
    results: list[RunResult] = []
    for config in CONFIGS:
        logger.info("Run %d: model=%s fps=%d", config.run_id, config.model_name, config.fps_sample)
        result = run_single(config, frames_cache, model_cache)
        results.append(result)

        # Save annotated image if any detections found
        if result.class_best:
            # Use the frame from the highest-confidence detection
            best_det = max(result.class_best.values(), key=lambda d: d.confidence)
            annotated = annotate_frame(best_det.frame, result.class_best)
            model_short = config.model_name.replace(".pt", "")
            filename = f"run_{config.run_id:02d}_{model_short}_fps{config.fps_sample}.jpg"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), annotated)
            logger.info("Saved annotated frame: %s", out_path)

    print_results_table(results)
    logger.info("Annotated images saved to: %s", output_dir)


if __name__ == "__main__":
    main()
