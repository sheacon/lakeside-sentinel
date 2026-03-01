from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lakeside_motorbikes.detection.yolo_detector import MOTORCYCLE_CLASS, MotorcycleDetector


def _make_mock_box(cls: int, conf: float, xyxy: list[float]) -> MagicMock:
    """Create a mock YOLO detection box."""
    box = MagicMock()
    box.cls = [cls]
    box.conf = [conf]
    box.xyxy = [MagicMock(tolist=MagicMock(return_value=xyxy))]
    return box


def _make_mock_result(boxes: list[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.boxes = boxes
    return result


@pytest.fixture
def dummy_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestMotorcycleDetector:
    @patch("lakeside_motorbikes.detection.yolo_detector.YOLO")
    def test_no_frames_returns_none(self, mock_yolo_cls: MagicMock) -> None:
        detector = MotorcycleDetector()
        assert detector.detect_best([]) is None

    @patch("lakeside_motorbikes.detection.yolo_detector.YOLO")
    def test_no_motorcycle_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        # Detection is a person (class 0), not a motorcycle
        box = _make_mock_box(cls=0, conf=0.9, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = MotorcycleDetector()
        assert detector.detect_best([dummy_frame]) is None

    @patch("lakeside_motorbikes.detection.yolo_detector.YOLO")
    def test_low_confidence_motorcycle_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box = _make_mock_box(cls=MOTORCYCLE_CLASS, conf=0.2, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = MotorcycleDetector(confidence_threshold=0.4)
        assert detector.detect_best([dummy_frame]) is None

    @patch("lakeside_motorbikes.detection.yolo_detector.YOLO")
    def test_detects_motorcycle_above_threshold(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box = _make_mock_box(cls=MOTORCYCLE_CLASS, conf=0.75, xyxy=[50, 50, 200, 200])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = MotorcycleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.confidence == 0.75
        assert detection.bbox == (50, 50, 200, 200)

    @patch("lakeside_motorbikes.detection.yolo_detector.YOLO")
    def test_returns_highest_confidence(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_low = _make_mock_box(cls=MOTORCYCLE_CLASS, conf=0.5, xyxy=[10, 10, 50, 50])
        box_high = _make_mock_box(cls=MOTORCYCLE_CLASS, conf=0.9, xyxy=[100, 100, 300, 300])

        result1 = _make_mock_result([box_low])
        result2 = _make_mock_result([box_high])
        mock_yolo_cls.return_value.return_value = [result1, result2]

        detector = MotorcycleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame, dummy_frame])

        assert detection is not None
        assert detection.confidence == 0.9
        assert detection.bbox == (100, 100, 300, 300)
