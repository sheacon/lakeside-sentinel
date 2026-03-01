from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lakeside_motorbikes.detection.vehicle_detector import VehicleDetector


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


class TestVehicleDetector:
    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_no_frames_returns_none(self, mock_yolo_cls: MagicMock) -> None:
        detector = VehicleDetector()
        assert detector.detect_best([]) is None

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_no_vehicle_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        # Detection is a person (class 0), not a vehicle
        box = _make_mock_box(cls=0, conf=0.9, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector()
        assert detector.detect_best([dummy_frame]) is None

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_low_confidence_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box = _make_mock_box(cls=3, conf=0.2, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        assert detector.detect_best([dummy_frame]) is None

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detects_motorcycle(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box = _make_mock_box(cls=3, conf=0.75, xyxy=[50, 50, 200, 200])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.confidence == 0.75
        assert detection.bbox == (50, 50, 200, 200)
        assert detection.class_name == "Motorcycle"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detects_car(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=2, conf=0.8, xyxy=[10, 10, 300, 300])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.class_name == "Car"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detects_bicycle(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=1, conf=0.6, xyxy=[10, 10, 200, 200])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.class_name == "Bicycle"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detects_bus(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=5, conf=0.7, xyxy=[10, 10, 400, 300])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.class_name == "Bus"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detects_truck(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=7, conf=0.65, xyxy=[10, 10, 350, 250])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.class_name == "Truck"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detailed_no_frames_returns_none_and_empty_dict(
        self, mock_yolo_cls: MagicMock
    ) -> None:
        detector = VehicleDetector()
        detection, class_max = detector.detect_detailed([])
        assert detection is None
        assert class_max == {}

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detailed_returns_best_and_per_class_best(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_car = _make_mock_box(cls=2, conf=0.82, xyxy=[10, 10, 50, 50])
        box_bike = _make_mock_box(cls=1, conf=0.31, xyxy=[60, 60, 120, 120])
        box_moto = _make_mock_box(cls=3, conf=0.12, xyxy=[200, 200, 300, 300])

        result1 = _make_mock_result([box_car, box_bike])
        result2 = _make_mock_result([box_moto])
        mock_yolo_cls.return_value.return_value = [result1, result2]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection, class_best = detector.detect_detailed([dummy_frame, dummy_frame])

        assert detection is not None
        assert detection.class_name == "Car"
        assert detection.confidence == 0.82
        assert set(class_best.keys()) == {"Car", "Bicycle", "Motorcycle"}
        assert class_best["Car"].confidence == 0.82
        assert class_best["Bicycle"].confidence == 0.31
        assert class_best["Motorcycle"].confidence == 0.12

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detailed_no_vehicles_returns_empty_dict(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_person = _make_mock_box(cls=0, conf=0.95, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box_person])]

        detector = VehicleDetector()
        detection, class_best = detector.detect_detailed([dummy_frame])
        assert detection is None
        assert class_best == {}

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detailed_includes_sub_threshold_in_breakdown(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_car_high = _make_mock_box(cls=2, conf=0.7, xyxy=[10, 10, 50, 50])
        box_truck_low = _make_mock_box(cls=7, conf=0.15, xyxy=[60, 60, 120, 120])
        mock_yolo_cls.return_value.return_value = [
            _make_mock_result([box_car_high, box_truck_low])
        ]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection, class_best = detector.detect_detailed([dummy_frame])

        # Best detection is only above-threshold
        assert detection is not None
        assert detection.class_name == "Car"
        # But class_best includes the sub-threshold truck
        assert set(class_best.keys()) == {"Car", "Truck"}
        assert class_best["Car"].confidence == 0.7
        assert class_best["Truck"].confidence == 0.15

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_returns_highest_confidence_across_types(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_car = _make_mock_box(cls=2, conf=0.5, xyxy=[10, 10, 50, 50])
        box_motorcycle = _make_mock_box(cls=3, conf=0.9, xyxy=[100, 100, 300, 300])

        result1 = _make_mock_result([box_car])
        result2 = _make_mock_result([box_motorcycle])
        mock_yolo_cls.return_value.return_value = [result1, result2]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame, dummy_frame])

        assert detection is not None
        assert detection.confidence == 0.9
        assert detection.bbox == (100, 100, 300, 300)
        assert detection.class_name == "Motorcycle"
