"""High-speed person (HSP) detection via person tracking and centroid displacement."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from lakeside_sentinel.detection.models import Detection

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


@dataclass
class TrackPoint:
    """A single observation of a tracked person."""

    frame_index: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[float, float, float, float]
    confidence: float
    frame: np.ndarray


@dataclass
class PersonTrack:
    """A sequence of observations tracking a single person across frames."""

    points: list[TrackPoint] = field(default_factory=list)

    def displacement_per_second(self, fps: int) -> float:
        """Average centroid displacement per second.

        Args:
            fps: Frames per second used during extraction.

        Returns 0.0 for tracks with fewer than 2 points.
        """
        if len(self.points) < 2:
            return 0.0
        first = self.points[0]
        last = self.points[-1]
        dx = last.centroid_x - first.centroid_x
        dy = last.centroid_y - first.centroid_y
        total_displacement = math.sqrt(dx * dx + dy * dy)
        return (total_displacement / (len(self.points) - 1)) * fps

    @property
    def best_point(self) -> TrackPoint:
        """Return the TrackPoint with the highest YOLO confidence."""
        return max(self.points, key=lambda p: p.confidence)


class HSPDetector:
    """Detects high-speed persons (HSP) via centroid tracking."""

    def __init__(
        self,
        model_name: str = "yolo26s.pt",
        person_confidence: float = 0.4,
        displacement_threshold: float = 240.0,
        max_match_distance: float = 800.0,
        batch_size: int = 16,
        *,
        fps_sample: int,
    ) -> None:
        self._model = YOLO(model_name)
        self._person_confidence = person_confidence
        self._displacement_threshold = displacement_threshold
        self._max_match_distance = max_match_distance
        self._batch_size = batch_size
        self._fps_sample = fps_sample
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("HSPDetector YOLO device: %s", self._device)

    def _empty_mps_cache(self) -> None:
        """Release unused MPS GPU memory back to the OS (no-op on CPU)."""
        if self._device == "mps":
            torch.mps.empty_cache()

    def _run_batched(
        self, frames: list[np.ndarray], imgsz: tuple[int, int], **kwargs: Any
    ) -> Iterator[tuple[int, np.ndarray, Any]]:
        """Run YOLO inference in batches.

        Yields:
            (frame_index, frame, result) tuples.
        """
        self._empty_mps_cache()
        offset = 0
        for i in range(0, len(frames), self._batch_size):
            batch = frames[i : i + self._batch_size]
            results = self._model(
                batch,
                verbose=False,
                imgsz=imgsz,
                device=self._device,
                classes=[PERSON_CLASS],
                **kwargs,
            )
            for j, (frame, result) in enumerate(zip(batch, results)):
                yield offset + j, frame, result
            offset += len(batch)

    @staticmethod
    def _compute_imgsz(frame_shape: tuple[int, ...], target_width: int = 1280) -> tuple[int, int]:
        """Compute YOLO imgsz preserving the frame's aspect ratio."""
        h, w = frame_shape[:2]
        aspect = h / w
        target_height = int(target_width * aspect)
        target_height = max(32, round(target_height / 32) * 32)
        target_width = max(32, round(target_width / 32) * 32)
        return (target_height, target_width)

    def _build_tracks(self, frames: list[np.ndarray]) -> list[PersonTrack]:
        """Detect persons and build centroid tracks across frames.

        Uses greedy nearest-centroid matching between consecutive frames.
        """
        if not frames:
            return []

        imgsz = self._compute_imgsz(frames[0].shape)

        # Collect per-frame person detections
        frame_detections: dict[int, list[TrackPoint]] = {}
        for frame_idx, frame, result in self._run_batched(frames, imgsz):
            points: list[TrackPoint] = []
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != PERSON_CLASS or conf < self._person_confidence:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                points.append(
                    TrackPoint(
                        frame_index=frame_idx,
                        centroid_x=cx,
                        centroid_y=cy,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        frame=frame,
                    )
                )
            frame_detections[frame_idx] = points

        # Build tracks via greedy nearest-centroid matching
        active_tracks: list[PersonTrack] = []
        finished_tracks: list[PersonTrack] = []
        max_match_distance_per_frame = self._max_match_distance / self._fps_sample

        sorted_indices = sorted(frame_detections.keys())
        for frame_idx in sorted_indices:
            points = frame_detections[frame_idx]
            matched_point_indices: set[int] = set()
            matched_track_indices: set[int] = set()

            # Try to extend each active track with nearest unmatched point
            for t_idx, track in enumerate(active_tracks):
                last = track.points[-1]
                best_dist = max_match_distance_per_frame
                best_p_idx: int | None = None

                for p_idx, point in enumerate(points):
                    if p_idx in matched_point_indices:
                        continue
                    dx = point.centroid_x - last.centroid_x
                    dy = point.centroid_y - last.centroid_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_p_idx = p_idx

                if best_p_idx is not None:
                    track.points.append(points[best_p_idx])
                    matched_point_indices.add(best_p_idx)
                    matched_track_indices.add(t_idx)

            # Finalize unmatched tracks
            new_active: list[PersonTrack] = []
            for t_idx, track in enumerate(active_tracks):
                if t_idx in matched_track_indices:
                    new_active.append(track)
                else:
                    finished_tracks.append(track)
            active_tracks = new_active

            # Start new tracks for unmatched points
            for p_idx, point in enumerate(points):
                if p_idx not in matched_point_indices:
                    active_tracks.append(PersonTrack(points=[point]))

        # All remaining active tracks are finished
        finished_tracks.extend(active_tracks)

        return finished_tracks

    def detect_all_tracks(self, frames: list[np.ndarray]) -> list[PersonTrack]:
        """Return all person tracks (for debug output and threshold tuning).

        Args:
            frames: List of BGR frames.

        Returns:
            All PersonTrack objects, including those below the displacement threshold.
        """
        return self._build_tracks(frames)

    def detect(self, frames: list[np.ndarray]) -> Detection | None:
        """Detect the fastest-moving person above the displacement threshold.

        Args:
            frames: List of BGR frames.

        Returns:
            Detection with class_name="HSP" for the fastest track, or None.
        """
        tracks = self._build_tracks(frames)
        fps = self._fps_sample

        fast_tracks = [
            t
            for t in tracks
            if len(t.points) >= 2 and t.displacement_per_second(fps) >= self._displacement_threshold
        ]

        if not fast_tracks:
            logger.debug("No fast-moving persons in %d frames", len(frames))
            return None

        # Pick the track with highest displacement
        fastest = max(fast_tracks, key=lambda t: t.displacement_per_second(fps))
        best = fastest.best_point

        logger.info(
            "HSP detected: displacement=%.1f px/sec, confidence=%.2f",
            fastest.displacement_per_second(fps),
            best.confidence,
        )

        return Detection(
            frame=best.frame,
            bbox=best.bbox,
            confidence=best.confidence,
            class_name="HSP",
            speed=fastest.displacement_per_second(fps),
        )
