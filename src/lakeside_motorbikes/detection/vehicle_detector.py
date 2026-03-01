import logging

import numpy as np
from ultralytics import YOLO

from lakeside_motorbikes.detection.models import Detection

logger = logging.getLogger(__name__)

VEHICLE_CLASSES: dict[int, str] = {
    1: "Bicycle",
    3: "Motorcycle",
}


class VehicleDetector:
    """Detects vehicles in frames using YOLO."""

    def __init__(self, model_name: str = "yolo26s.pt", confidence_threshold: float = 0.4) -> None:
        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold

    @staticmethod
    def _compute_imgsz(frame_shape: tuple[int, ...], target_width: int = 1280) -> tuple[int, int]:
        """Compute YOLO imgsz that preserves the frame's aspect ratio.

        Args:
            frame_shape: Shape of the input frame (h, w, ...).
            target_width: Desired width for inference.

        Returns:
            (height, width) tuple, both rounded to nearest multiple of 32.
        """
        h, w = frame_shape[:2]
        aspect = h / w
        target_height = int(target_width * aspect)
        # Round to nearest multiple of 32
        target_height = max(32, round(target_height / 32) * 32)
        target_width = max(32, round(target_width / 32) * 32)
        return (target_height, target_width)

    def detect_best(self, frames: list[np.ndarray]) -> Detection | None:
        """Run detection on all frames and return the single best vehicle detection.

        Args:
            frames: List of BGR frames.

        Returns:
            The Detection with highest confidence, or None if no vehicle found.
        """
        if not frames:
            return None

        best: Detection | None = None
        imgsz = self._compute_imgsz(frames[0].shape)

        results = self._model(frames, verbose=False, imgsz=imgsz)

        for frame, result in zip(frames, results):
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in VEHICLE_CLASSES and conf >= self._confidence_threshold:
                    if best is None or conf > best.confidence:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        best = Detection(
                            frame=frame,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_name=VEHICLE_CLASSES[cls],
                        )

        if best:
            logger.info(
                "Best vehicle detection: %s (confidence=%.2f)", best.class_name, best.confidence
            )
        else:
            logger.debug("No vehicle detected in %d frames", len(frames))

        return best

    def detect_detailed(
        self, frames: list[np.ndarray]
    ) -> tuple[Detection | None, dict[str, Detection]]:
        """Run detection and return the best vehicle detection plus per-class best detections.

        Unlike detect_best, the per-class dict includes all vehicle detections regardless
        of the confidence threshold, useful for debugging and tuning.

        Args:
            frames: List of BGR frames.

        Returns:
            Tuple of (best Detection or None, dict mapping class name to best Detection).
        """
        if not frames:
            return None, {}

        best: Detection | None = None
        class_best: dict[str, Detection] = {}
        imgsz = self._compute_imgsz(frames[0].shape)

        # Use a very low YOLO conf so we capture sub-threshold detections
        # for the per-class breakdown (useful for tuning).
        results = self._model(frames, verbose=False, conf=0.01, imgsz=imgsz)

        for frame, result in zip(frames, results):
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls not in VEHICLE_CLASSES:
                    continue

                class_name = VEHICLE_CLASSES[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = Detection(
                    frame=frame,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_name=class_name,
                )

                existing = class_best.get(class_name)
                if existing is None or conf > existing.confidence:
                    class_best[class_name] = det

                if conf >= self._confidence_threshold:
                    if best is None or conf > best.confidence:
                        best = det

        if best:
            logger.info(
                "Best vehicle detection: %s (confidence=%.2f)", best.class_name, best.confidence
            )
        else:
            logger.debug("No vehicle detected in %d frames", len(frames))

        return best, class_best
