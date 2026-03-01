import logging

import numpy as np
from ultralytics import YOLO

from lakeside_motorbikes.detection.models import Detection

logger = logging.getLogger(__name__)

VEHICLE_CLASSES: dict[int, str] = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


class VehicleDetector:
    """Detects vehicles in frames using YOLO."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.4) -> None:
        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold

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

        results = self._model(frames, verbose=False)

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
    ) -> tuple[Detection | None, dict[str, float]]:
        """Run detection and return the best vehicle detection plus per-class max confidences.

        Unlike detect_best, the per-class dict includes all vehicle detections regardless
        of the confidence threshold, useful for debugging and tuning.

        Args:
            frames: List of BGR frames.

        Returns:
            Tuple of (best Detection or None, dict mapping class name to max confidence).
        """
        if not frames:
            return None, {}

        best: Detection | None = None
        class_max: dict[str, float] = {}

        # Use a very low YOLO conf so we capture sub-threshold detections
        # for the per-class breakdown (useful for tuning).
        results = self._model(frames, verbose=False, conf=0.01)

        for frame, result in zip(frames, results):
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls not in VEHICLE_CLASSES:
                    continue

                class_name = VEHICLE_CLASSES[cls]
                class_max[class_name] = max(class_max.get(class_name, 0.0), conf)

                if conf >= self._confidence_threshold:
                    if best is None or conf > best.confidence:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        best = Detection(
                            frame=frame,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_name=class_name,
                        )

        if best:
            logger.info(
                "Best vehicle detection: %s (confidence=%.2f)", best.class_name, best.confidence
            )
        else:
            logger.debug("No vehicle detected in %d frames", len(frames))

        return best, class_max
