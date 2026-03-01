from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lakeside_motorbikes.detection.models import Detection
from lakeside_motorbikes.notification.html_report import ClipReport, generate_report


def _make_detection(
    class_name: str = "Car",
    confidence: float = 0.85,
) -> Detection:
    """Create a Detection with a small dummy frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return Detection(
        frame=frame,
        bbox=(10.0, 10.0, 90.0, 90.0),
        confidence=confidence,
        class_name=class_name,
    )


def _make_clip_report(
    hour: int = 12,
    best: Detection | None = None,
    class_detections: dict[str, Detection] | None = None,
) -> ClipReport:
    event_time = datetime(2026, 2, 28, hour, 0, 0, tzinfo=timezone.utc)
    local = event_time.astimezone()
    mp4_fn = local.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
    return ClipReport(
        event_time=event_time,
        mp4_filename=mp4_fn,
        best_detection=best,
        class_detections=class_detections or {},
    )


class TestGenerateReport:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        reports = [_make_clip_report(hour=10)]
        path = generate_report(reports, tmp_path)
        assert path == tmp_path / "report.html"
        assert path.exists()

    def test_contains_video_tags(self, tmp_path: Path) -> None:
        report = _make_clip_report(hour=14)
        path = generate_report([report], tmp_path)
        html = path.read_text()
        assert "<video" in html
        assert report.mp4_filename in html

    def test_contains_base64_images_for_detections(self, tmp_path: Path) -> None:
        car_det = _make_detection("Car", 0.85)
        bike_det = _make_detection("Bicycle", 0.30)
        report = _make_clip_report(
            hour=12,
            best=car_det,
            class_detections={"Car": car_det, "Bicycle": bike_det},
        )
        path = generate_report([report], tmp_path)
        html = path.read_text()
        assert "data:image/png;base64," in html
        assert "Car" in html
        assert "Bicycle" in html

    def test_handles_no_detection_clips(self, tmp_path: Path) -> None:
        report = _make_clip_report(hour=8, best=None, class_detections={})
        path = generate_report([report], tmp_path)
        html = path.read_text()
        assert "No detection above threshold" in html
        assert "<video" in html

    def test_empty_clip_list(self, tmp_path: Path) -> None:
        path = generate_report([], tmp_path)
        html = path.read_text()
        assert "0 clips analysed" in html
        assert "0 with detections" in html

    def test_confidence_formatted_as_percentage(self, tmp_path: Path) -> None:
        det = _make_detection("Truck", 0.731)
        report = _make_clip_report(
            hour=15,
            best=det,
            class_detections={"Truck": det},
        )
        path = generate_report([report], tmp_path)
        html = path.read_text()
        assert "73%" in html

    def test_summary_stats_correct(self, tmp_path: Path) -> None:
        det = _make_detection("Car", 0.9)
        reports = [
            _make_clip_report(hour=10, best=det, class_detections={"Car": det}),
            _make_clip_report(hour=11, best=None),
            _make_clip_report(hour=12, best=det, class_detections={"Car": det}),
        ]
        path = generate_report(reports, tmp_path)
        html = path.read_text()
        assert "3 clips analysed" in html
        assert "2 with detections" in html
