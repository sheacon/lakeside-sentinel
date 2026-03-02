from datetime import datetime, timezone

import numpy as np

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.notification.html_report import ClipReport, generate_report


def _make_detection(
    class_name: str = "Car",
    confidence: float = 0.85,
    speed: float | None = None,
) -> Detection:
    """Create a Detection with a small dummy frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return Detection(
        frame=frame,
        bbox=(10.0, 10.0, 90.0, 90.0),
        confidence=confidence,
        class_name=class_name,
        speed=speed,
    )


def _make_clip_report(
    hour: int = 12,
    best: Detection | None = None,
    class_detections: dict[str, Detection] | None = None,
) -> ClipReport:
    event_time = datetime(2026, 2, 28, hour, 0, 0, tzinfo=timezone.utc)
    local = event_time.astimezone()
    mp4_fn = "video/" + local.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
    return ClipReport(
        event_time=event_time,
        mp4_filename=mp4_fn,
        best_detection=best,
        class_detections=class_detections or {},
    )


class TestGenerateReport:
    def test_returns_html_string(self) -> None:
        det = _make_detection("motorcycle", 0.5)
        reports = [_make_clip_report(hour=10, best=det, class_detections={"motorcycle": det})]
        html = generate_report(reports)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_video_tags_by_default(self) -> None:
        det = _make_detection("motorcycle", 0.5)
        report = _make_clip_report(hour=14, best=det, class_detections={"motorcycle": det})
        html = generate_report([report])
        assert "<video" in html
        assert report.mp4_filename in html

    def test_no_video_tags_when_include_video_false(self) -> None:
        det = _make_detection("motorcycle", 0.5)
        report = _make_clip_report(hour=14, best=det, class_detections={"motorcycle": det})
        html = generate_report([report], include_video=False)
        assert "<video" not in html
        assert report.mp4_filename in html  # filename still in heading

    def test_contains_base64_images_for_detections(self) -> None:
        car_det = _make_detection("Car", 0.85)
        bike_det = _make_detection("Bicycle", 0.30)
        report = _make_clip_report(
            hour=12,
            best=car_det,
            class_detections={"Car": car_det, "Bicycle": bike_det},
        )
        html = generate_report([report])
        assert "data:image/png;base64," in html
        assert "Car" in html
        assert "Bicycle" in html

    def test_clips_with_no_detections_are_excluded(self) -> None:
        report = _make_clip_report(hour=8, best=None, class_detections={})
        html = generate_report([report])
        assert "<video" not in html
        assert "1 clips analysed" in html
        assert "0 with detections" in html

    def test_empty_clip_list(self) -> None:
        html = generate_report([])
        assert "0 clips analysed" in html
        assert "0 with detections" in html

    def test_confidence_formatted_as_percentage(self) -> None:
        det = _make_detection("Truck", 0.731)
        report = _make_clip_report(
            hour=15,
            best=det,
            class_detections={"Truck": det},
        )
        html = generate_report([report])
        assert "73%" in html

    def test_summary_stats_correct(self) -> None:
        det = _make_detection("Car", 0.9)
        reports = [
            _make_clip_report(hour=10, best=det, class_detections={"Car": det}),
            _make_clip_report(hour=11, best=None),
            _make_clip_report(hour=12, best=det, class_detections={"Car": det}),
        ]
        html = generate_report(reports)
        assert "3 clips analysed" in html
        assert "2 with detections" in html

    def test_sorted_by_motorcycle_confidence(self) -> None:
        low = _make_detection("Motorcycle", 0.3)
        high = _make_detection("Motorcycle", 0.9)
        mid = _make_detection("Motorcycle", 0.6)
        reports = [
            _make_clip_report(hour=10, best=low, class_detections={"Motorcycle": low}),
            _make_clip_report(hour=11, best=high, class_detections={"Motorcycle": high}),
            _make_clip_report(hour=12, best=mid, class_detections={"Motorcycle": mid}),
        ]
        html = generate_report(reports)
        # 90% should appear before 60% which should appear before 30%
        pos_90 = html.index("90%")
        pos_60 = html.index("60%")
        pos_30 = html.index("30%")
        assert pos_90 < pos_60 < pos_30

    def test_non_motorcycle_clips_sorted_after_motorcycle(self) -> None:
        moto = _make_detection("Motorcycle", 0.4)
        bike = _make_detection("Bicycle", 0.9)
        reports = [
            _make_clip_report(hour=10, best=bike, class_detections={"Bicycle": bike}),
            _make_clip_report(hour=11, best=moto, class_detections={"Motorcycle": moto}),
        ]
        html = generate_report(reports)
        # Motorcycle clip (hour=11) should appear before bicycle-only clip (hour=10)
        pos_moto = html.index("Motorcycle")
        pos_bike = html.index("Bicycle")
        assert pos_moto < pos_bike

    def test_secondary_sort_by_bicycle(self) -> None:
        """When motorcycle confidence is equal, sort by bicycle confidence."""
        bike_high = _make_detection("Bicycle", 0.8)
        bike_low = _make_detection("Bicycle", 0.3)
        reports = [
            _make_clip_report(hour=10, best=bike_low, class_detections={"Bicycle": bike_low}),
            _make_clip_report(hour=11, best=bike_high, class_detections={"Bicycle": bike_high}),
        ]
        html = generate_report(reports)
        # 80% should appear before 30%
        pos_80 = html.index("80%")
        pos_30 = html.index("30%")
        assert pos_80 < pos_30

    def test_report_default_title(self) -> None:
        html = generate_report([])
        assert "Detection Report" in html

    def test_report_custom_title(self) -> None:
        html = generate_report([], title="VEH Detection Report")
        assert "VEH Detection Report" in html

    def test_verified_badge_shown_for_confirmed(self) -> None:
        det = _make_detection("Motorcycle", 0.9)
        det.verification_status = "confirmed"
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report])
        assert "Claude verified" in html
        assert "Claude rejected" not in html

    def test_rejected_badge_shown_for_rejected(self) -> None:
        det = _make_detection("Bicycle", 0.7)
        det.verification_status = "rejected"
        report = _make_clip_report(hour=10, best=det, class_detections={"Bicycle": det})
        html = generate_report([report])
        assert "Claude rejected" in html
        assert "Claude verified" not in html

    def test_no_badge_when_not_verified(self) -> None:
        det = _make_detection("Motorcycle", 0.8)
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report])
        assert "Claude verified" not in html
        assert "Claude rejected" not in html

    def test_verification_response_shown_when_set(self) -> None:
        det = _make_detection("Motorcycle", 0.9)
        det.verification_status = "confirmed"
        det.verification_response = "yes"
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report])
        assert "Claude:" in html
        assert "&ldquo;yes&rdquo;" in html

    def test_verification_response_not_shown_when_none(self) -> None:
        det = _make_detection("Motorcycle", 0.8)
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report])
        assert "Claude:" not in html

    def test_hsp_sorted_by_speed(self) -> None:
        slow = _make_detection("HSP", 0.5, speed=160.0)
        fast = _make_detection("HSP", 0.4, speed=400.0)
        mid = _make_detection("HSP", 0.6, speed=280.0)
        reports = [
            _make_clip_report(hour=10, best=slow, class_detections={"HSP": slow}),
            _make_clip_report(hour=11, best=fast, class_detections={"HSP": fast}),
            _make_clip_report(hour=12, best=mid, class_detections={"HSP": mid}),
        ]
        html = generate_report(reports, mode="hsp")
        # 400 px/sec should appear before 280 before 160
        pos_400 = html.index("400 px/sec")
        pos_280 = html.index("280 px/sec")
        pos_160 = html.index("160 px/sec")
        assert pos_400 < pos_280 < pos_160

    def test_hsp_footer_says_sorted_by_speed(self) -> None:
        det = _make_detection("HSP", 0.5, speed=300.0)
        reports = [_make_clip_report(hour=10, best=det, class_detections={"HSP": det})]
        html = generate_report(reports, mode="hsp")
        assert "sorted by speed" in html
        assert "sorted by motorcycle confidence" not in html

    def test_veh_footer_says_sorted_by_motorcycle_confidence(self) -> None:
        det = _make_detection("Motorcycle", 0.8)
        reports = [_make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})]
        html = generate_report(reports, mode="veh")
        assert "sorted by motorcycle confidence" in html

    def test_hsp_card_shows_speed_instead_of_confidence(self) -> None:
        det = _make_detection("HSP", 0.5, speed=320.0)
        report = _make_clip_report(hour=10, best=det, class_detections={"HSP": det})
        html = generate_report([report], mode="hsp")
        assert "320 px/sec" in html


class TestPresentModeReport:
    def test_sorted_chronologically(self) -> None:
        early = _make_detection("Motorcycle", 0.9)
        late = _make_detection("Motorcycle", 0.3)
        reports = [
            _make_clip_report(hour=14, best=late, class_detections={"Motorcycle": late}),
            _make_clip_report(hour=10, best=early, class_detections={"Motorcycle": early}),
        ]
        html = generate_report(reports, mode="present")
        # Times display in local time; get the expected local time strings
        early_time = datetime(2026, 2, 28, 10, 0, 0, tzinfo=timezone.utc)
        late_time = datetime(2026, 2, 28, 14, 0, 0, tzinfo=timezone.utc)
        early_str = early_time.astimezone().strftime("%H:%M:%S")
        late_str = late_time.astimezone().strftime("%H:%M:%S")
        pos_early = html.index(early_str)
        pos_late = html.index(late_str)
        assert pos_early < pos_late

    def test_sort_label_says_chronologically(self) -> None:
        det = _make_detection("Motorcycle", 0.8)
        reports = [_make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})]
        html = generate_report(reports, mode="present")
        assert "sorted chronologically" in html
        assert "sorted by motorcycle confidence" not in html

    def test_hides_actual_class_names(self) -> None:
        moto = _make_detection("Motorcycle", 0.9)
        hsp = _make_detection("HSP", 0.5, speed=300.0)
        report = _make_clip_report(
            hour=10,
            best=moto,
            class_detections={"Motorcycle": moto, "HSP": hsp},
        )
        html = generate_report([report], mode="present")
        assert "Potential Motorized Vehicle" in html
        assert ">Motorcycle<" not in html
        assert ">HSP<" not in html

    def test_hides_confidence_percentages(self) -> None:
        det = _make_detection("Motorcycle", 0.85)
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report], mode="present")
        assert "85%" not in html

    def test_hides_speed_metrics(self) -> None:
        det = _make_detection("HSP", 0.5, speed=320.0)
        report = _make_clip_report(hour=10, best=det, class_detections={"HSP": det})
        html = generate_report([report], mode="present")
        assert "px/sec" not in html

    def test_hides_claude_badges(self) -> None:
        det = _make_detection("Motorcycle", 0.9)
        det.verification_status = "confirmed"
        det.verification_response = "yes this is a motorcycle"
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report], mode="present")
        assert "Claude verified" not in html
        assert "Claude rejected" not in html
        assert "Claude:" not in html

    def test_best_detection_shows_generic_label(self) -> None:
        det = _make_detection("Motorcycle", 0.9)
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report], mode="present")
        # Best detection summary should show generic label, not class name
        assert "Potential Motorized Vehicle" in html

    def test_no_detection_shows_generic_message(self) -> None:
        det = _make_detection("Motorcycle", 0.5)
        # Create a report where best_detection is None but has class_detections
        report = _make_clip_report(hour=10, best=None, class_detections={"Motorcycle": det})
        html = generate_report([report], mode="present")
        assert "No detection" in html
        assert "No detection above threshold" not in html

    def test_shows_cropped_images(self) -> None:
        det = _make_detection("Motorcycle", 0.85)
        report = _make_clip_report(hour=10, best=det, class_detections={"Motorcycle": det})
        html = generate_report([report], mode="present")
        assert "data:image/png;base64," in html

    def test_mixed_veh_and_hsp_in_one_clip(self) -> None:
        moto = _make_detection("Motorcycle", 0.9)
        hsp = _make_detection("HSP", 0.5, speed=300.0)
        report = _make_clip_report(
            hour=10,
            best=moto,
            class_detections={"Motorcycle": moto, "HSP": hsp},
        )
        html = generate_report([report], mode="present")
        # Should have two cards, both labeled "Potential Motorized Vehicle"
        assert html.count("Potential Motorized Vehicle") >= 2
