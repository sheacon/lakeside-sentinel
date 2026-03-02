"""Detection tuning harness — sweeps model/FPS/threshold combos via CLI flags."""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lakeside_sentinel.detection.hsp_detector import HSPDetector, PersonTrack
from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.detection.veh_detector import VEHDetector
from lakeside_sentinel.utils.image import crop_to_roi
from lakeside_sentinel.utils.video import extract_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_CONFIGS_WARNING = 50


@dataclass(frozen=True)
class RunConfig:
    """Single tuning run configuration."""

    run_id: int
    mode: str  # "veh" or "hsp"
    model_name: str
    fps_sample: int
    # VEH-specific
    confidence_threshold: float = 0.4
    # HSP-specific
    hsp_displacement: float = 240.0
    hsp_person_confidence_threshold: float = 0.4
    hsp_max_match_distance: float = 800.0
    # ROI (shared, not swept)
    roi_y_start: float = 0.0
    roi_y_end: float = 1.0
    roi_x_start: float = 0.0
    roi_x_end: float = 1.0


@dataclass
class VEHRunResult:
    """Result of a single VEH tuning run."""

    config: RunConfig
    best: Detection | None
    class_best: dict[str, Detection]
    sub_threshold: dict[str, Detection]
    frame_count: int
    elapsed_secs: float


@dataclass
class HSPRunResult:
    """Result of a single HSP tuning run."""

    config: RunConfig
    tracks: list[PersonTrack]
    fast_tracks: list[PersonTrack]
    max_displacement: float
    frame_count: int
    elapsed_secs: float


def build_veh_configs(
    models: list[str],
    fps_values: list[int],
    confidences: list[float],
    roi_y_start: float,
    roi_y_end: float,
    roi_x_start: float,
    roi_x_end: float,
) -> list[RunConfig]:
    """Build RunConfig list from Cartesian product of VEH sweep params."""
    configs: list[RunConfig] = []
    for run_id, (model, fps, conf) in enumerate(
        itertools.product(models, fps_values, confidences), start=1
    ):
        configs.append(
            RunConfig(
                run_id=run_id,
                mode="veh",
                model_name=model,
                fps_sample=fps,
                confidence_threshold=conf,
                roi_y_start=roi_y_start,
                roi_y_end=roi_y_end,
                roi_x_start=roi_x_start,
                roi_x_end=roi_x_end,
            )
        )
    return configs


def build_hsp_configs(
    models: list[str],
    fps_values: list[int],
    person_confidence_thresholds: list[float],
    displacements: list[float],
    max_match_distances: list[float],
    roi_y_start: float,
    roi_y_end: float,
    roi_x_start: float,
    roi_x_end: float,
) -> list[RunConfig]:
    """Build RunConfig list from Cartesian product of HSP sweep params."""
    configs: list[RunConfig] = []
    for run_id, (model, fps, pconf, disp, maxd) in enumerate(
        itertools.product(
            models, fps_values, person_confidence_thresholds, displacements, max_match_distances
        ),
        start=1,
    ):
        configs.append(
            RunConfig(
                run_id=run_id,
                mode="hsp",
                model_name=model,
                fps_sample=fps,
                hsp_person_confidence_threshold=pconf,
                hsp_displacement=disp,
                hsp_max_match_distance=maxd,
                roi_y_start=roi_y_start,
                roi_y_end=roi_y_end,
                roi_x_start=roi_x_start,
                roi_x_end=roi_x_end,
            )
        )
    return configs


def apply_threshold(
    class_best: dict[str, Detection],
    threshold: float,
) -> tuple[Detection | None, dict[str, Detection]]:
    """Post-filter detect_detailed results by confidence threshold.

    Args:
        class_best: Per-class best detections (from detect_detailed with conf=0.01).
        threshold: Confidence threshold for the "best" detection.

    Returns:
        (best Detection above threshold or None, sub-threshold detections dict).
    """
    best: Detection | None = None
    sub_threshold: dict[str, Detection] = {}

    for class_name, det in class_best.items():
        if det.confidence >= threshold:
            if best is None or det.confidence > best.confidence:
                best = det
        else:
            sub_threshold[class_name] = det

    return best, sub_threshold


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


def annotate_hsp_frame(
    frame: np.ndarray,
    tracks: list[PersonTrack],
    displacement_threshold: float,
    fps: int,
) -> np.ndarray:
    """Draw HSP track visualizations on a frame.

    Fast tracks (above threshold) are drawn in red, slow tracks in green.
    """
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for track in tracks:
        disp = track.displacement_per_second(fps)
        is_fast = len(track.points) >= 2 and disp >= displacement_threshold
        color = (0, 0, 255) if is_fast else (0, 200, 0)

        # Draw trajectory line
        if len(track.points) >= 2:
            pts = [(int(p.centroid_x), int(p.centroid_y)) for p in track.points]
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], color, 2)

        # Draw bounding box and label at best point
        best = track.best_point
        x1, y1, x2, y2 = (int(v) for v in best.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"d={disp:.0f} c={best.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
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


def run_veh(
    config: RunConfig,
    frames_cache: dict[int, list[np.ndarray]],
    model_cache: dict[str, tuple[VEHDetector, dict[str, Detection]]],
) -> VEHRunResult:
    """Execute a single VEH tuning run.

    The model_cache maps model_name -> (detector, class_best_from_detect_detailed).
    detect_detailed is called once per (model, fps) pair with conf=0.01,
    then apply_threshold post-filters per confidence value.
    """
    frames = frames_cache[config.fps_sample]
    cache_key = (config.model_name, config.fps_sample)

    t0 = time.perf_counter()

    # Check if we already ran detect_detailed for this model+fps
    if cache_key not in model_cache:
        detector = _get_or_create_veh_detector(config.model_name, {})
        _, class_best = detector.detect_detailed(frames)
        model_cache[cache_key] = (detector, class_best)

    _, cached_class_best = model_cache[cache_key]

    # Post-filter by this run's confidence threshold
    best, sub_threshold = apply_threshold(cached_class_best, config.confidence_threshold)
    elapsed = time.perf_counter() - t0

    # Build the above-threshold class_best for annotation
    above_threshold: dict[str, Detection] = {}
    for class_name, det in cached_class_best.items():
        if det.confidence >= config.confidence_threshold:
            above_threshold[class_name] = det

    return VEHRunResult(
        config=config,
        best=best,
        class_best=above_threshold,
        sub_threshold=sub_threshold,
        frame_count=len(frames),
        elapsed_secs=elapsed,
    )


def _get_or_create_veh_detector(
    model_name: str,
    detector_cache: dict[str, VEHDetector],
) -> VEHDetector:
    """Get or create a VEHDetector, caching by model name."""
    if model_name not in detector_cache:
        detector_cache[model_name] = VEHDetector(model_name=model_name)
    return detector_cache[model_name]


def run_hsp(
    config: RunConfig,
    frames_cache: dict[int, list[np.ndarray]],
    hsp_cache: dict[tuple[str, float, float, int], list[PersonTrack]],
) -> HSPRunResult:
    """Execute a single HSP tuning run.

    The hsp_cache maps (model_name, person_confidence, max_match_distance, fps)
    -> all tracks. Displacement threshold is post-filtered.
    """
    frames = frames_cache[config.fps_sample]
    cache_key = (
        config.model_name,
        config.hsp_person_confidence_threshold,
        config.hsp_max_match_distance,
        config.fps_sample,
    )

    t0 = time.perf_counter()

    if cache_key not in hsp_cache:
        detector = HSPDetector(
            model_name=config.model_name,
            person_confidence=config.hsp_person_confidence_threshold,
            displacement_threshold=0.0,  # we post-filter
            max_match_distance=config.hsp_max_match_distance,
            fps_sample=config.fps_sample,
        )
        all_tracks = detector.detect_all_tracks(frames)
        hsp_cache[cache_key] = all_tracks

    all_tracks = hsp_cache[cache_key]

    # Post-filter by displacement threshold (px/sec)
    fast_tracks = [
        t
        for t in all_tracks
        if len(t.points) >= 2
        and t.displacement_per_second(config.fps_sample) >= config.hsp_displacement
    ]

    max_disp = max(
        (t.displacement_per_second(config.fps_sample) for t in all_tracks if len(t.points) >= 2),
        default=0.0,
    )

    elapsed = time.perf_counter() - t0

    return HSPRunResult(
        config=config,
        tracks=all_tracks,
        fast_tracks=fast_tracks,
        max_displacement=max_disp,
        frame_count=len(frames),
        elapsed_secs=elapsed,
    )


def print_veh_table(results: list[VEHRunResult]) -> None:
    """Print a formatted VEH results table to stdout."""
    header = (
        f"{'Run':>3} | {'Model':<10} | {'FPS':>3} | {'Conf Thr':>8} | {'Frames':>6} "
        f"| {'Best':<12} | {'Best Conf':>9} | {'Sub-threshold':<25} | {'Time':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for r in results:
        best_label = r.best.class_name.upper() if r.best else "-"
        best_conf = f"{r.best.confidence:.0%}" if r.best else "-"

        sub_parts: list[str] = []
        for cls_name, det in sorted(r.sub_threshold.items()):
            sub_parts.append(f"{cls_name}: {det.confidence:.0%}")
        sub_threshold = ", ".join(sub_parts) if sub_parts else "-"

        model_short = r.config.model_name.replace(".pt", "")
        conf_str = f"{r.config.confidence_threshold:.2f}"
        print(
            f"{r.config.run_id:>3} | {model_short:<10} | {r.config.fps_sample:>3} "
            f"| {conf_str:>8} | {r.frame_count:>6} | {best_label:<12} | {best_conf:>9} "
            f"| {sub_threshold:<25} | {r.elapsed_secs:>5.1f}s"
        )

    print()


def print_hsp_table(results: list[HSPRunResult]) -> None:
    """Print a formatted HSP results table to stdout."""
    header = (
        f"{'Run':>3} | {'Model':<10} | {'FPS':>3} | {'PConf':>5} | {'Disp Thr':>8} "
        f"| {'MaxDist':>7} | {'Frames':>6} | {'Tracks':>6} | {'Fast':>4} "
        f"| {'Max Disp':>8} | {'Time':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for r in results:
        model_short = r.config.model_name.replace(".pt", "")
        print(
            f"{r.config.run_id:>3} | {model_short:<10} | {r.config.fps_sample:>3} "
            f"| {r.config.hsp_person_confidence_threshold:>5.2f} "
            f"| {r.config.hsp_displacement:>8.1f} "
            f"| {r.config.hsp_max_match_distance:>7.0f} | {r.frame_count:>6} "
            f"| {len(r.tracks):>6} | {len(r.fast_tracks):>4} "
            f"| {r.max_displacement:>8.1f} | {r.elapsed_secs:>5.1f}s"
        )

    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Sweep detection parameters on a video clip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # VEH mode — sweep models and FPS\n"
            "  python scripts/tune_detection.py --clip clip.mp4 "
            "--model yolo_models/yolo26s.pt yolo_models/yolo26m.pt --fps 2 4 8\n\n"
            "  # HSP mode — sweep displacement thresholds\n"
            "  python scripts/tune_detection.py --clip clip.mp4 --hsp "
            "--fps 4 8 --hsp-displacement 40 60 80"
        ),
    )
    parser.add_argument("--clip", type=Path, required=True, help="Path to MP4 clip to analyze")
    parser.add_argument(
        "--hsp",
        action="store_true",
        default=False,
        help="Use HSP detection mode instead of VEH",
    )

    # Shared sweep params
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        default=["yolo_models/yolo26s.pt"],
        help="YOLO model weight files to sweep",
    )
    parser.add_argument(
        "--fps", nargs="+", type=int, default=None, help="FPS sample rates to sweep"
    )

    # ROI (single values, not swept)
    parser.add_argument(
        "--roi-y-start", type=float, default=0.0, help="ROI vertical start (0.0–1.0)"
    )
    parser.add_argument("--roi-y-end", type=float, default=1.0, help="ROI vertical end (0.0–1.0)")
    parser.add_argument(
        "--roi-x-start", type=float, default=0.0, help="ROI horizontal start (0.0–1.0)"
    )
    parser.add_argument("--roi-x-end", type=float, default=1.0, help="ROI horizontal end (0.0–1.0)")

    # VEH-specific sweep params
    parser.add_argument(
        "--confidence",
        nargs="+",
        type=float,
        default=[0.4],
        help="Confidence thresholds to sweep (VEH mode)",
    )

    # HSP-specific sweep params
    parser.add_argument(
        "--hsp-displacement",
        nargs="+",
        type=float,
        default=[240.0],
        help="Displacement thresholds in px/sec to sweep (HSP mode)",
    )
    parser.add_argument(
        "--hsp-person-confidence",
        nargs="+",
        type=float,
        default=[0.4],
        help="Person confidence thresholds to sweep (HSP mode)",
    )
    parser.add_argument(
        "--hsp-max-match-distance",
        nargs="+",
        type=float,
        default=[800.0],
        help="Max match distances in px/sec to sweep (HSP mode)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    clip_path: Path = args.clip
    if not clip_path.exists():
        logger.error("Clip not found: %s", clip_path)
        raise SystemExit(1)

    # Default FPS depends on mode
    fps_values: list[int] = args.fps if args.fps is not None else ([4] if args.hsp else [2])

    # Build configs
    if args.hsp:
        configs = build_hsp_configs(
            models=args.model,
            fps_values=fps_values,
            person_confidence_thresholds=args.hsp_person_confidence,
            displacements=args.hsp_displacement,
            max_match_distances=args.hsp_max_match_distance,
            roi_y_start=args.roi_y_start,
            roi_y_end=args.roi_y_end,
            roi_x_start=args.roi_x_start,
            roi_x_end=args.roi_x_end,
        )
    else:
        configs = build_veh_configs(
            models=args.model,
            fps_values=fps_values,
            confidences=args.confidence,
            roi_y_start=args.roi_y_start,
            roi_y_end=args.roi_y_end,
            roi_x_start=args.roi_x_start,
            roi_x_end=args.roi_x_end,
        )

    if len(configs) > MAX_CONFIGS_WARNING:
        logger.warning(
            "Large sweep: %d configurations. Consider reducing parameter combinations.",
            len(configs),
        )

    logger.info("Mode: %s | %d configurations", "HSP" if args.hsp else "VEH", len(configs))

    mp4_bytes = clip_path.read_bytes()
    clip_stem = clip_path.stem
    output_dir = Path("output/tune") / clip_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache frames by fps_sample
    unique_fps = sorted({c.fps_sample for c in configs})
    logger.info("Extracting frames for FPS values: %s", unique_fps)

    roi = configs[0]  # ROI is the same for all configs
    frames_cache: dict[int, list[np.ndarray]] = {}
    for fps in unique_fps:
        raw_frames = extract_frames(mp4_bytes, fps_sample=fps)
        cropped = crop_to_roi(
            raw_frames,
            y_start=roi.roi_y_start,
            y_end=roi.roi_y_end,
            x_start=roi.roi_x_start,
            x_end=roi.roi_x_end,
        )
        frames_cache[fps] = cropped
        shape = cropped[0].shape[:2] if cropped else "N/A"
        logger.info("FPS %d: %d frames extracted, ROI-cropped to %s", fps, len(cropped), shape)

    # Run all configs
    if args.hsp:
        _run_hsp_sweep(configs, frames_cache, output_dir)
    else:
        _run_veh_sweep(configs, frames_cache, output_dir)

    logger.info("Annotated images saved to: %s", output_dir)


def _run_veh_sweep(
    configs: list[RunConfig],
    frames_cache: dict[int, list[np.ndarray]],
    output_dir: Path,
) -> None:
    """Run all VEH configs, save annotations, and print results."""
    # Cache: (model_name, fps) -> (detector, class_best from detect_detailed)
    model_cache: dict[str, tuple[VEHDetector, dict[str, Detection]]] = {}
    results: list[VEHRunResult] = []

    for config in configs:
        logger.info(
            "Run %d: model=%s fps=%d conf=%.2f",
            config.run_id,
            config.model_name,
            config.fps_sample,
            config.confidence_threshold,
        )
        result = run_veh(config, frames_cache, model_cache)
        results.append(result)

        # Save annotated image if any detections found
        all_detections = {**result.class_best, **result.sub_threshold}
        if all_detections:
            best_det = max(all_detections.values(), key=lambda d: d.confidence)
            annotated = annotate_frame(best_det.frame, all_detections)
            model_short = config.model_name.replace(".pt", "")
            conf_str = f"{config.confidence_threshold:.2f}".replace(".", "")
            filename = (
                f"run_{config.run_id:02d}_{model_short}_fps{config.fps_sample}_conf{conf_str}.jpg"
            )
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), annotated)
            logger.info("Saved annotated frame: %s", out_path)

    print_veh_table(results)


def _run_hsp_sweep(
    configs: list[RunConfig],
    frames_cache: dict[int, list[np.ndarray]],
    output_dir: Path,
) -> None:
    """Run all HSP configs, save annotations, and print results."""
    hsp_cache: dict[tuple[str, float, float, int], list[PersonTrack]] = {}
    results: list[HSPRunResult] = []

    for config in configs:
        logger.info(
            "Run %d: model=%s fps=%d pconf=%.2f disp=%.0f maxd=%.0f",
            config.run_id,
            config.model_name,
            config.fps_sample,
            config.hsp_person_confidence_threshold,
            config.hsp_displacement,
            config.hsp_max_match_distance,
        )
        result = run_hsp(config, frames_cache, hsp_cache)
        results.append(result)

        # Save annotated image showing all tracks
        if result.tracks:
            best_track = max(
                result.tracks, key=lambda t: t.displacement_per_second(config.fps_sample)
            )
            best_point = best_track.best_point
            annotated = annotate_hsp_frame(
                best_point.frame, result.tracks, config.hsp_displacement, config.fps_sample
            )
            model_short = config.model_name.replace(".pt", "")
            disp_str = f"{config.hsp_displacement:.0f}"
            filename = (
                f"run_{config.run_id:02d}_{model_short}_fps{config.fps_sample}_disp{disp_str}.jpg"
            )
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), annotated)
            logger.info("Saved annotated frame: %s", out_path)

    print_hsp_table(results)


if __name__ == "__main__":
    main()
