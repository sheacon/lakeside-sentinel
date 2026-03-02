from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lakeside_sentinel.detection.hsp_detector import (
    HSPDetector,
    PersonTrack,
    TrackPoint,
)


def _make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_track_point(
    frame_index: int = 0,
    cx: float = 100.0,
    cy: float = 100.0,
    confidence: float = 0.8,
) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        centroid_x=cx,
        centroid_y=cy,
        bbox=(cx - 20, cy - 40, cx + 20, cy + 40),
        confidence=confidence,
        frame=_make_frame(),
    )


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
    return _make_frame()


class TestPersonTrack:
    def test_single_point_displacement_zero(self) -> None:
        track = PersonTrack(points=[_make_track_point()])
        assert track.displacement_per_second(fps=4) == 0.0

    def test_two_points_displacement(self) -> None:
        p1 = _make_track_point(frame_index=0, cx=100.0, cy=100.0)
        p2 = _make_track_point(frame_index=1, cx=130.0, cy=140.0)
        track = PersonTrack(points=[p1, p2])
        # sqrt(30^2 + 40^2) / 1 interval * 4 fps = 50.0 * 4 = 200.0 px/sec
        assert track.displacement_per_second(fps=4) == 200.0

    def test_multiple_points_average_displacement(self) -> None:
        p1 = _make_track_point(frame_index=0, cx=0.0, cy=0.0)
        p2 = _make_track_point(frame_index=1, cx=10.0, cy=0.0)
        p3 = _make_track_point(frame_index=2, cx=30.0, cy=0.0)
        track = PersonTrack(points=[p1, p2, p3])
        # sqrt(30^2 + 0^2) / 2 intervals * 4 fps = 15.0 * 4 = 60.0 px/sec
        assert track.displacement_per_second(fps=4) == 60.0

    def test_best_point_returns_highest_confidence(self) -> None:
        p1 = _make_track_point(confidence=0.5)
        p2 = _make_track_point(confidence=0.9)
        p3 = _make_track_point(confidence=0.7)
        track = PersonTrack(points=[p1, p2, p3])
        assert track.best_point.confidence == 0.9

    def test_empty_track_displacement_zero(self) -> None:
        track = PersonTrack(points=[])
        assert track.displacement_per_second(fps=4) == 0.0

    def test_fps_invariance(self) -> None:
        """Same physical motion at different FPS should yield equal displacement_per_second."""
        # At 4 FPS: person moves 30px per frame interval over 2 intervals
        p1_4fps = _make_track_point(frame_index=0, cx=0.0, cy=0.0)
        p2_4fps = _make_track_point(frame_index=1, cx=30.0, cy=0.0)
        p3_4fps = _make_track_point(frame_index=2, cx=60.0, cy=0.0)
        track_4fps = PersonTrack(points=[p1_4fps, p2_4fps, p3_4fps])

        # At 8 FPS: same motion = 15px per frame interval over 4 intervals
        p1_8fps = _make_track_point(frame_index=0, cx=0.0, cy=0.0)
        p2_8fps = _make_track_point(frame_index=1, cx=15.0, cy=0.0)
        p3_8fps = _make_track_point(frame_index=2, cx=30.0, cy=0.0)
        p4_8fps = _make_track_point(frame_index=3, cx=45.0, cy=0.0)
        p5_8fps = _make_track_point(frame_index=4, cx=60.0, cy=0.0)
        track_8fps = PersonTrack(points=[p1_8fps, p2_8fps, p3_8fps, p4_8fps, p5_8fps])

        disp_4fps = track_4fps.displacement_per_second(fps=4)
        disp_8fps = track_8fps.displacement_per_second(fps=8)

        assert disp_4fps == pytest.approx(disp_8fps, abs=0.01)


class TestTrackBuilding:
    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_single_person_across_frames_one_track(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """A person detected in 3 consecutive frames should form 1 track."""
        mock_model = mock_yolo_cls.return_value

        # Person moves right: x=100, x=130, x=160
        boxes = [
            [_make_mock_box(cls=0, conf=0.8, xyxy=[80, 60, 120, 140])],
            [_make_mock_box(cls=0, conf=0.85, xyxy=[110, 60, 150, 140])],
            [_make_mock_box(cls=0, conf=0.82, xyxy=[140, 60, 180, 140])],
        ]

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result(boxes[i]) for i in range(len(frames))]

        mock_model.side_effect = side_effect

        detector = HSPDetector(person_confidence=0.4, max_match_distance=800.0, fps_sample=4)
        frames = [dummy_frame] * 3
        tracks = detector.detect_all_tracks(frames)

        assert len(tracks) == 1
        assert len(tracks[0].points) == 3

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_two_people_tracked_independently(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Two people in each frame should form 2 separate tracks."""
        mock_model = mock_yolo_cls.return_value

        # Person A at left, Person B at right — both stationary
        person_a = _make_mock_box(cls=0, conf=0.8, xyxy=[10, 10, 50, 100])
        person_b = _make_mock_box(cls=0, conf=0.7, xyxy=[400, 10, 440, 100])

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result([person_a, person_b]) for _ in frames]

        mock_model.side_effect = side_effect

        detector = HSPDetector(person_confidence=0.4, max_match_distance=800.0, fps_sample=4)
        tracks = detector.detect_all_tracks([dummy_frame, dummy_frame])

        assert len(tracks) == 2

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_person_disappears_track_finalized(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Person in frame 0, gone in frame 1 — track is finalized with 1 point."""
        mock_model = mock_yolo_cls.return_value

        frame0_box = _make_mock_box(cls=0, conf=0.8, xyxy=[80, 60, 120, 140])

        def side_effect(frames: list, **kwargs: object) -> list:
            results = []
            for i in range(len(frames)):
                if i == 0:
                    results.append(_make_mock_result([frame0_box]))
                else:
                    results.append(_make_mock_result([]))
            return results

        mock_model.side_effect = side_effect

        detector = HSPDetector(person_confidence=0.4, fps_sample=4)
        tracks = detector.detect_all_tracks([dummy_frame, dummy_frame])

        assert len(tracks) == 1
        assert len(tracks[0].points) == 1

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_centroids_too_far_separate_tracks(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Centroids beyond max_match_distance produce separate tracks."""
        mock_model = mock_yolo_cls.return_value

        # Frame 0: person at x=100. Frame 1: person at x=500 (400px away)
        # max_match_distance=400 px/sec at 4fps = 100 px/frame, so 400px gap -> separate
        box0 = _make_mock_box(cls=0, conf=0.8, xyxy=[80, 60, 120, 140])
        box1 = _make_mock_box(cls=0, conf=0.8, xyxy=[480, 60, 520, 140])

        def side_effect(frames: list, **kwargs: object) -> list:
            results = []
            for i in range(len(frames)):
                if i == 0:
                    results.append(_make_mock_result([box0]))
                else:
                    results.append(_make_mock_result([box1]))
            return results

        mock_model.side_effect = side_effect

        # 400 px/sec / 4 fps = 100 px/frame; 400px gap > 100 -> separate tracks
        detector = HSPDetector(person_confidence=0.4, max_match_distance=400.0, fps_sample=4)
        tracks = detector.detect_all_tracks([dummy_frame, dummy_frame])

        assert len(tracks) == 2
        assert all(len(t.points) == 1 for t in tracks)

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_empty_frames_no_tracks(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Frames with no detections produce no tracks."""
        mock_model = mock_yolo_cls.return_value

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result([]) for _ in frames]

        mock_model.side_effect = side_effect

        detector = HSPDetector(person_confidence=0.4, fps_sample=4)
        tracks = detector.detect_all_tracks([dummy_frame, dummy_frame])

        assert len(tracks) == 0

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_no_frames_returns_empty(self, mock_yolo_cls: MagicMock) -> None:
        detector = HSPDetector(fps_sample=4)
        assert detector.detect_all_tracks([]) == []


class TestHSPDetector:
    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_no_frames_returns_none(self, mock_yolo_cls: MagicMock) -> None:
        detector = HSPDetector(fps_sample=4)
        assert detector.detect([]) is None

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_no_persons_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        mock_model = mock_yolo_cls.return_value

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result([]) for _ in frames]

        mock_model.side_effect = side_effect

        detector = HSPDetector(person_confidence=0.4, fps_sample=4)
        assert detector.detect([dummy_frame, dummy_frame]) is None

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_slow_person_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Person moving slowly (below threshold) should not be flagged."""
        mock_model = mock_yolo_cls.return_value

        # Person moves 5px per frame at 4fps = 20 px/sec (below 240.0 threshold)
        boxes = [
            [_make_mock_box(cls=0, conf=0.8, xyxy=[95, 60, 105, 140])],
            [_make_mock_box(cls=0, conf=0.8, xyxy=[100, 60, 110, 140])],
            [_make_mock_box(cls=0, conf=0.8, xyxy=[105, 60, 115, 140])],
        ]

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result(boxes[i]) for i in range(len(frames))]

        mock_model.side_effect = side_effect

        detector = HSPDetector(
            person_confidence=0.4,
            displacement_threshold=240.0,
            max_match_distance=800.0,
            fps_sample=4,
        )
        assert detector.detect([dummy_frame] * 3) is None

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_fast_person_detected_as_hsp(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Person moving fast should be flagged as HSP."""
        mock_model = mock_yolo_cls.return_value

        # Person moves 60px per frame at 4fps = 240 px/sec (>= 240.0 threshold)
        boxes = [
            [_make_mock_box(cls=0, conf=0.75, xyxy=[80, 60, 120, 140])],
            [_make_mock_box(cls=0, conf=0.85, xyxy=[140, 60, 180, 140])],
            [_make_mock_box(cls=0, conf=0.80, xyxy=[200, 60, 240, 140])],
        ]

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result(boxes[i]) for i in range(len(frames))]

        mock_model.side_effect = side_effect

        detector = HSPDetector(
            person_confidence=0.4,
            displacement_threshold=240.0,
            max_match_distance=800.0,
            fps_sample=4,
        )
        detection = detector.detect([dummy_frame] * 3)

        assert detection is not None
        assert detection.class_name == "HSP"
        assert detection.confidence == 0.85  # best confidence in track

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_multiple_people_only_fast_one_flagged(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Only the fast person should be detected, not the slow one."""
        mock_model = mock_yolo_cls.return_value

        # Slow person (left): barely moves. Fast person (right): moves 60px/frame = 240 px/sec
        frame_boxes = [
            [
                _make_mock_box(cls=0, conf=0.9, xyxy=[10, 60, 50, 140]),  # slow
                _make_mock_box(cls=0, conf=0.7, xyxy=[400, 60, 440, 140]),  # fast
            ],
            [
                _make_mock_box(cls=0, conf=0.9, xyxy=[12, 60, 52, 140]),  # slow
                _make_mock_box(cls=0, conf=0.75, xyxy=[460, 60, 500, 140]),  # fast
            ],
            [
                _make_mock_box(cls=0, conf=0.9, xyxy=[14, 60, 54, 140]),  # slow
                _make_mock_box(cls=0, conf=0.8, xyxy=[520, 60, 560, 140]),  # fast
            ],
        ]

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result(frame_boxes[i]) for i in range(len(frames))]

        mock_model.side_effect = side_effect

        detector = HSPDetector(
            person_confidence=0.4,
            displacement_threshold=240.0,
            max_match_distance=800.0,
            fps_sample=4,
        )
        detection = detector.detect([dummy_frame] * 3)

        assert detection is not None
        assert detection.class_name == "HSP"
        # The fast person's bbox should be around x=400-560 range
        assert detection.bbox[0] >= 400

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_single_frame_person_returns_none(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """A person seen in only one frame can't have displacement computed."""
        mock_model = mock_yolo_cls.return_value

        box = _make_mock_box(cls=0, conf=0.9, xyxy=[80, 60, 120, 140])

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result([box]) for _ in frames]

        mock_model.side_effect = side_effect

        detector = HSPDetector(
            person_confidence=0.4,
            displacement_threshold=240.0,
            max_match_distance=800.0,
            fps_sample=4,
        )
        # Only one frame → track has 1 point → can't compute displacement
        assert detector.detect([dummy_frame]) is None

    @patch("lakeside_sentinel.detection.hsp_detector.YOLO")
    def test_low_confidence_person_filtered(
        self, mock_yolo_cls: MagicMock, dummy_frame: np.ndarray
    ) -> None:
        """Persons below person_confidence threshold should be ignored."""
        mock_model = mock_yolo_cls.return_value

        # Fast-moving but low confidence
        boxes = [
            [_make_mock_box(cls=0, conf=0.2, xyxy=[80, 60, 120, 140])],
            [_make_mock_box(cls=0, conf=0.2, xyxy=[200, 60, 240, 140])],
        ]

        def side_effect(frames: list, **kwargs: object) -> list:
            return [_make_mock_result(boxes[i]) for i in range(len(frames))]

        mock_model.side_effect = side_effect

        detector = HSPDetector(person_confidence=0.4, displacement_threshold=240.0, fps_sample=4)
        assert detector.detect([dummy_frame, dummy_frame]) is None
