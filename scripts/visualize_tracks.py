"""HSP track visualization — annotated video and summary image from person tracks."""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lakeside_sentinel.detection.hsp_detector import HSPDetector, PersonTrack
from lakeside_sentinel.utils.image import crop_to_roi
from lakeside_sentinel.utils.video import extract_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Defaults matching .env.example (avoids requiring Settings/full .env)
DEFAULT_MODEL = "yolo_models/yolo26s.pt"
DEFAULT_FPS = 4
DEFAULT_DISPLACEMENT = 240.0
DEFAULT_PERSON_CONFIDENCE = 0.4
DEFAULT_MAX_MATCH_DISTANCE = 800.0

# Colors (BGR)
COLOR_FAST = (0, 0, 255)  # red
COLOR_SLOW = (0, 200, 0)  # green


@dataclass(frozen=True)
class TrackSummary:
    """Summary of a single person track."""

    track_id: int
    num_points: int
    displacement_per_sec: float
    above_threshold: bool
    best_confidence: float


def summarize_tracks(
    tracks: list[PersonTrack],
    fps: int,
    threshold: float,
) -> list[TrackSummary]:
    """Produce a summary for each track.

    Args:
        tracks: Person tracks from HSPDetector.
        fps: Frames per second used during extraction.
        threshold: Displacement threshold (px/sec) for fast/slow classification.

    Returns:
        List of TrackSummary with sequential track IDs starting at 1.
    """
    summaries: list[TrackSummary] = []
    for i, track in enumerate(tracks, start=1):
        disp = track.displacement_per_second(fps)
        is_fast = len(track.points) >= 2 and disp >= threshold
        summaries.append(
            TrackSummary(
                track_id=i,
                num_points=len(track.points),
                displacement_per_sec=disp,
                above_threshold=is_fast,
                best_confidence=track.best_point.confidence if track.points else 0.0,
            )
        )
    return summaries


def annotate_frame_progressive(
    frame: np.ndarray,
    tracks: list[PersonTrack],
    current_frame_index: int,
    threshold: float,
    fps: int,
) -> np.ndarray:
    """Draw track data up to current_frame_index on a frame copy.

    - Trajectory line segments for points seen so far
    - Bbox + label for points active in the current frame
    - Red for fast tracks (above threshold), green for slow
    - Color based on full track displacement (not partial)

    Args:
        frame: BGR image to annotate.
        tracks: All person tracks.
        current_frame_index: Only draw data up to this frame index.
        threshold: Displacement threshold (px/sec) for coloring.
        fps: Frames per second used during extraction.

    Returns:
        Annotated copy of the frame.
    """
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for track in tracks:
        # Filter points up to current frame
        visible = [p for p in track.points if p.frame_index <= current_frame_index]
        if not visible:
            continue

        # Color based on full track displacement
        disp = track.displacement_per_second(fps)
        is_fast = len(track.points) >= 2 and disp >= threshold
        color = COLOR_FAST if is_fast else COLOR_SLOW

        # Draw trajectory line segments for visible points
        if len(visible) >= 2:
            pts = [(int(p.centroid_x), int(p.centroid_y)) for p in visible]
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], color, 2)

        # Draw bbox + label for points active in the current frame
        active = [p for p in visible if p.frame_index == current_frame_index]
        for point in active:
            x1, y1, x2, y2 = (int(v) for v in point.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"d={disp:.0f} c={point.confidence:.0%}"
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


def write_annotated_video(
    frames: list[np.ndarray],
    tracks: list[PersonTrack],
    threshold: float,
    fps: int,
    output_path: Path,
) -> None:
    """Write an annotated MP4 video with progressive track visualization.

    Args:
        frames: List of BGR frames (same order as extraction).
        tracks: All person tracks.
        threshold: Displacement threshold (px/sec) for coloring.
        fps: Frames per second (used for playback speed and displacement calc).
        output_path: Path to write the output MP4.
    """
    if not frames:
        logger.warning("No frames to write video")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        logger.warning(
            "H.264 (avc1) codec unavailable; falling back to mp4v. "
            "Re-encode with: ffmpeg -i %s -c:v libx264 output.mp4",
            output_path,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_idx, frame in enumerate(frames):
        annotated = annotate_frame_progressive(frame, tracks, frame_idx, threshold, fps)
        writer.write(annotated)

    writer.release()
    logger.info("Wrote annotated video: %s (%d frames)", output_path, len(frames))


def write_summary_image(
    frames: list[np.ndarray],
    tracks: list[PersonTrack],
    threshold: float,
    fps: int,
    output_path: Path,
) -> None:
    """Write a static summary image with all tracks drawn on the best frame.

    The "best frame" is the frame of the best point from the fastest track,
    falling back to the middle frame if no tracks exist.

    Args:
        frames: List of BGR frames.
        tracks: All person tracks.
        threshold: Displacement threshold (px/sec) for coloring.
        fps: Frames per second used during extraction.
        output_path: Path to write the output JPEG.
    """
    if not frames:
        logger.warning("No frames to write summary image")
        return

    # Pick the best frame
    if tracks:
        fastest = max(tracks, key=lambda t: t.displacement_per_second(fps))
        best_frame_idx = fastest.best_point.frame_index
    else:
        best_frame_idx = len(frames) // 2

    base_frame = frames[best_frame_idx]
    annotated = base_frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for track in tracks:
        disp = track.displacement_per_second(fps)
        is_fast = len(track.points) >= 2 and disp >= threshold
        color = COLOR_FAST if is_fast else COLOR_SLOW

        # Draw full trajectory
        if len(track.points) >= 2:
            pts = [(int(p.centroid_x), int(p.centroid_y)) for p in track.points]
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], color, 2)

        # Draw bbox + label at best point
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

    cv2.imwrite(str(output_path), annotated)
    logger.info("Wrote summary image: %s", output_path)


def print_track_table(summaries: list[TrackSummary]) -> None:
    """Print a formatted track summary table to stdout."""
    header = f"{'ID':>4} | {'Points':>6} | {'Disp (px/s)':>11} | {'Fast':>4} | {'Best Conf':>9}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for s in summaries:
        fast_label = "YES" if s.above_threshold else "no"
        print(
            f"{s.track_id:>4} | {s.num_points:>6} | {s.displacement_per_sec:>11.1f} "
            f"| {fast_label:>4} | {s.best_confidence:>9.0%}"
        )

    print()


def process_clip(
    clip_path: Path,
    model: str,
    fps: int,
    displacement: float,
    person_confidence: float,
    max_match_distance: float,
    roi_y_start: float,
    roi_y_end: float,
    roi_x_start: float,
    roi_x_end: float,
    output_dir: Path | None = None,
) -> list[TrackSummary]:
    """Process a single clip: extract frames, detect tracks, write outputs.

    Args:
        clip_path: Path to the MP4 clip.
        model: YOLO model weights file.
        fps: Frames per second for extraction.
        displacement: Displacement threshold (px/sec) for coloring.
        person_confidence: Min YOLO person confidence for tracking.
        max_match_distance: Max centroid distance (px/sec) for track matching.
        roi_y_start: ROI vertical start (0.0-1.0).
        roi_y_end: ROI vertical end (0.0-1.0).
        roi_x_start: ROI horizontal start (0.0-1.0).
        roi_x_end: ROI horizontal end (0.0-1.0).
        output_dir: Directory for output files. Defaults to output/tracks/{clip_stem}/.

    Returns:
        List of TrackSummary for the clip.
    """
    clip_stem = clip_path.stem
    if output_dir is None:
        output_dir = Path("output/tracks") / clip_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing clip: %s", clip_path)

    # Extract and crop frames
    mp4_bytes = clip_path.read_bytes()
    raw_frames = extract_frames(mp4_bytes, fps_sample=fps)
    frames = crop_to_roi(
        raw_frames,
        y_start=roi_y_start,
        y_end=roi_y_end,
        x_start=roi_x_start,
        x_end=roi_x_end,
    )
    logger.info("Extracted %d frames at %d FPS", len(frames), fps)

    if not frames:
        logger.warning("No frames extracted from %s", clip_path)
        return []

    # Detect all tracks (displacement_threshold=0.0 to get ALL tracks)
    detector = HSPDetector(
        model_name=model,
        person_confidence=person_confidence,
        displacement_threshold=0.0,
        max_match_distance=max_match_distance,
        fps_sample=fps,
    )
    tracks = detector.detect_all_tracks(frames)
    logger.info("Found %d tracks", len(tracks))

    # Write outputs
    video_path = output_dir / f"{clip_stem}_tracks.mp4"
    write_annotated_video(frames, tracks, displacement, fps, video_path)

    summary_path = output_dir / f"{clip_stem}_summary.jpg"
    write_summary_image(frames, tracks, displacement, fps, summary_path)

    # Summarize
    summaries = summarize_tracks(tracks, fps, displacement)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Visualize HSP person tracks as annotated video and summary image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/visualize_tracks.py --clip clip.mp4\n"
            "  python scripts/visualize_tracks.py --clip a.mp4 b.mp4 --fps 8\n"
            "  python scripts/visualize_tracks.py --clip clip.mp4 "
            "--displacement 320.0 --person-confidence 0.3"
        ),
    )
    parser.add_argument("--clip", type=Path, nargs="+", required=True, help="One or more MP4 paths")
    parser.add_argument("--model", type=str, default=None, help="YOLO model weights file")
    parser.add_argument(
        "--fps", type=int, nargs="+", default=None, help="Frames per second (multi-value sweep)"
    )
    parser.add_argument(
        "--displacement",
        type=float,
        nargs="+",
        default=None,
        help="Displacement threshold px/sec (multi-value sweep)",
    )
    parser.add_argument(
        "--person-confidence",
        type=float,
        nargs="+",
        default=None,
        help="Min YOLO person confidence (multi-value sweep)",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        nargs="+",
        default=None,
        help="Max centroid distance px/sec (multi-value sweep)",
    )
    parser.add_argument(
        "--roi-y-start", type=float, default=None, help="ROI vertical start (0.0-1.0)"
    )
    parser.add_argument("--roi-y-end", type=float, default=None, help="ROI vertical end (0.0-1.0)")
    parser.add_argument(
        "--roi-x-start", type=float, default=None, help="ROI horizontal start (0.0-1.0)"
    )
    parser.add_argument(
        "--roi-x-end", type=float, default=None, help="ROI horizontal end (0.0-1.0)"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL
    fps_values = args.fps or [DEFAULT_FPS]
    displacement_values = (
        args.displacement if args.displacement is not None else [DEFAULT_DISPLACEMENT]
    )
    person_confidence_values = (
        args.person_confidence
        if args.person_confidence is not None
        else [DEFAULT_PERSON_CONFIDENCE]
    )
    max_match_distance_values = (
        args.max_match_distance
        if args.max_match_distance is not None
        else [DEFAULT_MAX_MATCH_DISTANCE]
    )
    roi_y_start = args.roi_y_start if args.roi_y_start is not None else 0.0
    roi_y_end = args.roi_y_end if args.roi_y_end is not None else 1.0
    roi_x_start = args.roi_x_start if args.roi_x_start is not None else 0.0
    roi_x_end = args.roi_x_end if args.roi_x_end is not None else 1.0

    combos = list(
        itertools.product(
            fps_values,
            displacement_values,
            person_confidence_values,
            max_match_distance_values,
        )
    )
    is_sweep = len(combos) > 1

    for clip_path in args.clip:
        if not clip_path.exists():
            logger.error("Clip not found: %s", clip_path)
            continue

        clip_stem = clip_path.stem

        for run_id, (fps, displacement, person_confidence, max_match_distance) in enumerate(
            combos, start=1
        ):
            if is_sweep:
                run_label = (
                    f"fps{fps}_disp{displacement:.0f}"
                    f"_conf{person_confidence:.2f}_match{max_match_distance:.0f}"
                )
                output_dir = Path("output/tracks") / clip_stem / run_label
                print(f"\n[{run_id}/{len(combos)}] {run_label}")
            else:
                output_dir = Path("output/tracks") / clip_stem

            summaries = process_clip(
                clip_path=clip_path,
                model=model,
                fps=fps,
                displacement=displacement,
                person_confidence=person_confidence,
                max_match_distance=max_match_distance,
                roi_y_start=roi_y_start,
                roi_y_end=roi_y_end,
                roi_x_start=roi_x_start,
                roi_x_end=roi_x_end,
                output_dir=output_dir,
            )
            print_track_table(summaries)


if __name__ == "__main__":
    main()
