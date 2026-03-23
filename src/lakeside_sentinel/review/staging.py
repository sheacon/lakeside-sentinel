"""Serializes Detection objects to disk for the review web app."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.notification.html_report import ClipReport

logger = logging.getLogger(__name__)

_STAGING_DIR = Path("output") / "staging"


def _detection_to_dict(
    det: Detection,
    det_id: str,
    section: str,
    source: str,
    event_time_iso: str,
    mp4_filename: str,
    frame_filename: str,
) -> dict[str, object]:
    """Convert a Detection to a JSON-serializable dict."""
    h, w = det.frame.shape[:2]
    return {
        "id": det_id,
        "section": section,
        "source": source,
        "event_time_iso": event_time_iso,
        "mp4_filename": mp4_filename,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": list(det.bbox),
        "frame_filename": frame_filename,
        "frame_height": h,
        "frame_width": w,
        "verification_status": det.verification_status,
        "verification_response": det.verification_response,
        "speed": det.speed,
    }


def stage_detections(
    date_str: str,
    merged_reports: list[ClipReport],
    veh_debug_reports: list[ClipReport],
    hsp_debug_reports: list[ClipReport],
    crop_padding: float,
    total_clips: int | None = None,
) -> Path:
    """Stage detection data to disk for the review web app.

    Saves frame PNGs (deduplicated by id(frame)) and writes staging.json.

    Args:
        date_str: Date string (YYYY-MM-DD).
        merged_reports: Merged VEH+HSP reports (confirmed detections).
        veh_debug_reports: All VEH detections (debug).
        hsp_debug_reports: All HSP detections (debug).
        crop_padding: Padding fraction for cropping.
        total_clips: Total number of video clips analysed (for report stats).

    Returns:
        Path to the staging directory.
    """
    staging_dir = _STAGING_DIR / date_str
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Collect confirmed detection IDs from merged reports
    confirmed_ids: set[str] = set()
    confirmed_dets: dict[str, tuple[Detection, str, str]] = {}

    for report in merged_reports:
        event_time_iso = report.event_time.isoformat()
        for class_name, det in report.class_detections.items():
            if det.verification_status == "confirmed":
                source = "hsp" if class_name == "HSP" else "veh"
                det_id = f"{source}_{id(det)}_{class_name}"
                confirmed_ids.add((report.mp4_filename, class_name))
                confirmed_dets[det_id] = (det, event_time_iso, report.mp4_filename)

    # Frame deduplication: map id(frame) -> filename
    frame_map: dict[int, str] = {}
    frame_counter = 0
    detections_json: list[dict[str, object]] = []

    def _save_frame(frame: np.ndarray) -> str:
        nonlocal frame_counter
        frame_id = id(frame)
        if frame_id in frame_map:
            return frame_map[frame_id]
        filename = f"frame_{frame_counter}.png"
        cv2.imwrite(str(staging_dir / filename), frame)
        frame_map[frame_id] = filename
        frame_counter += 1
        return filename

    # Track which (mp4_filename, class_name) combos are already added
    added: set[tuple[str, str]] = set()

    # 1. Confirmed detections from merged reports
    for report in merged_reports:
        event_time_iso = report.event_time.isoformat()
        for class_name, det in report.class_detections.items():
            if det.verification_status != "confirmed":
                continue
            source = "hsp" if class_name == "HSP" else "veh"
            det_id = f"{source}_{len(detections_json)}_{class_name}"
            frame_filename = _save_frame(det.frame)
            detections_json.append(
                _detection_to_dict(
                    det,
                    det_id,
                    "confirmed",
                    source,
                    event_time_iso,
                    report.mp4_filename,
                    frame_filename,
                )
            )
            added.add((report.mp4_filename, class_name))

    # 2. VEH debug detections (not already confirmed)
    for report in veh_debug_reports:
        event_time_iso = report.event_time.isoformat()
        for class_name, det in report.class_detections.items():
            key = (report.mp4_filename, class_name)
            if key in added:
                continue
            det_id = f"veh_{len(detections_json)}_{class_name}"
            frame_filename = _save_frame(det.frame)
            detections_json.append(
                _detection_to_dict(
                    det,
                    det_id,
                    "veh_debug",
                    "veh",
                    event_time_iso,
                    report.mp4_filename,
                    frame_filename,
                )
            )
            added.add(key)

    # 3. HSP debug detections (not already confirmed)
    for report in hsp_debug_reports:
        event_time_iso = report.event_time.isoformat()
        for class_name, det in report.class_detections.items():
            key = (report.mp4_filename, class_name)
            if key in added:
                continue
            det_id = f"hsp_{len(detections_json)}_{class_name}"
            frame_filename = _save_frame(det.frame)
            detections_json.append(
                _detection_to_dict(
                    det,
                    det_id,
                    "hsp_debug",
                    "hsp",
                    event_time_iso,
                    report.mp4_filename,
                    frame_filename,
                )
            )
            added.add(key)

    staging_data: dict[str, object] = {
        "date_str": date_str,
        "crop_padding": crop_padding,
        "detections": detections_json,
    }
    if total_clips is not None:
        staging_data["total_clips"] = total_clips

    json_path = staging_dir / "staging.json"
    json_path.write_text(json.dumps(staging_data, indent=2))
    logger.info("Staged %d detections to %s", len(detections_json), staging_dir)

    return staging_dir


def discover_unreviewed() -> list[Path]:
    """Scan output/staging/ for unreviewed staging directories.

    Returns:
        List of staging directory paths sorted by date (ascending).
    """
    if not _STAGING_DIR.exists():
        return []
    dirs = []
    for d in _STAGING_DIR.iterdir():
        if d.is_dir() and (d / "staging.json").exists():
            dirs.append(d)
    dirs.sort(key=lambda p: p.name)
    return dirs


def load_staged_detections(staging_dir: Path) -> dict[str, object]:
    """Load staging.json from a staging directory.

    Returns:
        Parsed staging data dict.
    """
    json_path = staging_dir / "staging.json"
    return json.loads(json_path.read_text())


def load_frame(staging_dir: Path, filename: str) -> np.ndarray:
    """Load a staged frame PNG as a numpy array."""
    frame = cv2.imread(str(staging_dir / filename))
    if frame is None:
        msg = f"Failed to load frame: {staging_dir / filename}"
        raise FileNotFoundError(msg)
    return frame


def rebuild_clip_reports(
    staging_dir: Path,
    selected_ids: set[str],
) -> list[ClipReport]:
    """Reconstruct ClipReport objects for selected detections only.

    Args:
        staging_dir: Path to the staging directory.
        selected_ids: Set of detection IDs to include.

    Returns:
        List of ClipReport objects with only selected detections.
    """
    data = load_staged_detections(staging_dir)
    detections = data["detections"]

    # Group selected detections by mp4_filename
    by_clip: dict[str, list[dict[str, object]]] = {}
    for det_dict in detections:
        if det_dict["id"] in selected_ids:
            mp4 = det_dict["mp4_filename"]
            by_clip.setdefault(mp4, []).append(det_dict)

    clip_reports: list[ClipReport] = []
    for mp4_filename, det_dicts in by_clip.items():
        class_detections: dict[str, Detection] = {}
        for dd in det_dicts:
            frame = load_frame(staging_dir, dd["frame_filename"])
            det = Detection(
                frame=frame,
                bbox=tuple(dd["bbox"]),
                confidence=dd["confidence"],
                class_name=dd["class_name"],
                verification_status=dd["verification_status"],
                verification_response=dd["verification_response"],
                speed=dd["speed"],
            )
            class_detections[dd["class_name"]] = det

        # Pick best detection
        confirmed = [d for d in class_detections.values() if d.verification_status == "confirmed"]
        unverified = [d for d in class_detections.values() if d.verification_status is None]
        candidates = confirmed + unverified
        best = max(candidates, key=lambda d: d.confidence) if candidates else None

        event_time = datetime.fromisoformat(det_dicts[0]["event_time_iso"])
        clip_reports.append(
            ClipReport(
                event_time=event_time,
                mp4_filename=mp4_filename,
                best_detection=best,
                class_detections=class_detections,
            )
        )

    return clip_reports


def cleanup_staging(staging_dir: Path) -> None:
    """Delete a staging directory after successful submit."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
        logger.info("Cleaned up staging directory: %s", staging_dir)


def cleanup_videos_for_date(date_str: str) -> None:
    """Delete video clips for a given date (YYYY-MM-DD) from output/video/."""
    video_dir = Path("output") / "video"
    if not video_dir.exists():
        return
    prefix = date_str + "_"
    count = 0
    for filepath in video_dir.iterdir():
        if filepath.name.startswith(prefix) and filepath.suffix == ".mp4":
            filepath.unlink()
            count += 1
    if count:
        logger.info("Cleaned up %d video(s) for %s", count, date_str)
