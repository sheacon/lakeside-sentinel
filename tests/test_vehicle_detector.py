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
    def test_detects_motorcycle(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=3, conf=0.75, xyxy=[50, 50, 200, 200])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.confidence == 0.75
        assert detection.bbox == (50, 50, 200, 200)
        assert detection.class_name == "Motorcycle"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_ignores_car(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=2, conf=0.8, xyxy=[10, 10, 300, 300])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        assert detector.detect_best([dummy_frame]) is None

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detects_bicycle(self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray) -> None:
        box = _make_mock_box(cls=1, conf=0.6, xyxy=[10, 10, 200, 200])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame])

        assert detection is not None
        assert detection.class_name == "Bicycle"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detailed_no_frames_returns_none_and_empty_dict(self, mock_yolo_cls: MagicMock) -> None:
        detector = VehicleDetector()
        detection, class_max = detector.detect_detailed([])
        assert detection is None
        assert class_max == {}

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_detailed_returns_best_and_per_class_best(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_moto = _make_mock_box(cls=3, conf=0.82, xyxy=[10, 10, 50, 50])
        box_bike = _make_mock_box(cls=1, conf=0.31, xyxy=[60, 60, 120, 120])

        result1 = _make_mock_result([box_moto, box_bike])
        result2 = _make_mock_result([])
        mock_yolo_cls.return_value.return_value = [result1, result2]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection, class_best = detector.detect_detailed([dummy_frame, dummy_frame])

        assert detection is not None
        assert detection.class_name == "Motorcycle"
        assert detection.confidence == 0.82
        assert set(class_best.keys()) == {"Bicycle", "Motorcycle"}
        assert class_best["Motorcycle"].confidence == 0.82
        assert class_best["Bicycle"].confidence == 0.31

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
        box_moto_high = _make_mock_box(cls=3, conf=0.7, xyxy=[10, 10, 50, 50])
        box_bike_low = _make_mock_box(cls=1, conf=0.15, xyxy=[60, 60, 120, 120])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box_moto_high, box_bike_low])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection, class_best = detector.detect_detailed([dummy_frame])

        # Best detection is only above-threshold
        assert detection is not None
        assert detection.class_name == "Motorcycle"
        # But class_best includes the sub-threshold bicycle
        assert set(class_best.keys()) == {"Motorcycle", "Bicycle"}
        assert class_best["Motorcycle"].confidence == 0.7
        assert class_best["Bicycle"].confidence == 0.15

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_returns_highest_confidence_across_types(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        box_bike = _make_mock_box(cls=1, conf=0.5, xyxy=[10, 10, 50, 50])
        box_motorcycle = _make_mock_box(cls=3, conf=0.9, xyxy=[100, 100, 300, 300])

        result1 = _make_mock_result([box_bike])
        result2 = _make_mock_result([box_motorcycle])
        mock_yolo_cls.return_value.return_value = [result1, result2]

        detector = VehicleDetector(confidence_threshold=0.4)
        detection = detector.detect_best([dummy_frame, dummy_frame])

        assert detection is not None
        assert detection.confidence == 0.9
        assert detection.bbox == (100, 100, 300, 300)
        assert detection.class_name == "Motorcycle"

    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_batched_inference_splits_frames(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """20 frames with batch_size=8 should call _model 3 times (8+8+4)."""
        box = _make_mock_box(cls=3, conf=0.85, xyxy=[10, 10, 100, 100])
        mock_model = mock_yolo_cls.return_value
        mock_model.return_value = [_make_mock_result([box])]

        # Make _model return one result per frame in each batch
        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result([box]) for _ in frames]

        mock_model.side_effect = side_effect

        detector = VehicleDetector(confidence_threshold=0.4, batch_size=8)
        frames = [dummy_frame] * 20
        detection = detector.detect_best(frames)

        assert detection is not None
        assert detection.confidence == 0.85
        assert detection.class_name == "Motorcycle"
        assert mock_model.call_count == 3
        # Verify batch sizes: 8, 8, 4
        batch_sizes = [len(call.args[0]) for call in mock_model.call_args_list]
        assert batch_sizes == [8, 8, 4]


class TestEmptyMpsCache:
    @patch("lakeside_motorbikes.detection.vehicle_detector.torch")
    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_calls_empty_cache_on_mps(
        self, mock_yolo_cls: MagicMock, mock_torch: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        mock_torch.backends.mps.is_available.return_value = True

        box = _make_mock_box(cls=3, conf=0.8, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detector.detect_best([dummy_frame])

        mock_torch.mps.empty_cache.assert_called_once()

    @patch("lakeside_motorbikes.detection.vehicle_detector.torch")
    @patch("lakeside_motorbikes.detection.vehicle_detector.YOLO")
    def test_skips_empty_cache_on_cpu(
        self, mock_yolo_cls: MagicMock, mock_torch: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        mock_torch.backends.mps.is_available.return_value = False

        box = _make_mock_box(cls=3, conf=0.8, xyxy=[10, 10, 100, 100])
        mock_yolo_cls.return_value.return_value = [_make_mock_result([box])]

        detector = VehicleDetector(confidence_threshold=0.4)
        detector.detect_best([dummy_frame])

        mock_torch.mps.empty_cache.assert_not_called()


class TestComputeImgsz:
    def test_square_frame(self) -> None:
        h, w = VehicleDetector._compute_imgsz((640, 640, 3))
        assert h == w == 1280
        assert h % 32 == 0

    def test_wide_frame_preserves_aspect(self) -> None:
        # 1920x360 strip (top-third ROI of 1080p)
        h, w = VehicleDetector._compute_imgsz((360, 1920, 3))
        assert w == 1280
        assert h % 32 == 0
        # Aspect ratio should be roughly preserved
        expected_h = int(1280 * (360 / 1920))  # 240
        assert abs(h - expected_h) <= 16  # within one rounding step

    def test_results_are_multiples_of_32(self) -> None:
        h, w = VehicleDetector._compute_imgsz((100, 700, 3))
        assert h % 32 == 0
        assert w % 32 == 0

    def test_custom_target_width(self) -> None:
        h, w = VehicleDetector._compute_imgsz((480, 640, 3), target_width=640)
        assert w == 640
        assert h == 480
        assert h % 32 == 0
