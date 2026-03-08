import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.notification.html_report import ClipReport
from lakeside_sentinel.review.server import (
    _create_app,
    _group_by_video,
    _sort_detections,
    _video_review_score,
)
from lakeside_sentinel.review.staging import stage_detections


def _make_detection(
    class_name: str = "Motorcycle",
    confidence: float = 0.9,
    verification_status: str | None = "confirmed",
    frame: np.ndarray | None = None,
) -> Detection:
    if frame is None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
    return Detection(
        frame=frame,
        bbox=(10.0, 20.0, 90.0, 80.0),
        confidence=confidence,
        class_name=class_name,
        verification_status=verification_status,
    )


def _make_clip_report(
    class_detections: dict[str, Detection] | None = None,
    mp4_filename: str = "video/2026-03-06_12-00-00.mp4",
) -> ClipReport:
    if class_detections is None:
        class_detections = {}
    best = None
    if class_detections:
        best = max(class_detections.values(), key=lambda d: d.confidence)
    return ClipReport(
        event_time=datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc),
        mp4_filename=mp4_filename,
        best_detection=best,
        class_detections=class_detections,
    )


@pytest.fixture
def staged_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a staged detection and return Flask test client."""
    monkeypatch.chdir(tmp_path)

    det = _make_detection()
    merged = [_make_clip_report({"Motorcycle": det})]
    stage_detections("2026-03-06", merged, [], [], 0.2)

    app = _create_app()
    app.config["TESTING"] = True
    return app.test_client()


class TestIndex:
    def test_redirects_to_first_day(self, staged_app) -> None:
        resp = staged_app.get("/")
        assert resp.status_code == 302
        assert "/2026-03-06" in resp.headers["Location"]

    def test_no_data_returns_message(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        app = _create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"No staged data" in resp.data


class TestReviewDay:
    def test_renders_page_with_video_groups(self, staged_app) -> None:
        resp = staged_app.get("/2026-03-06")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "2026-03-06" in html
        assert "Motorcycle" in html
        assert "video-group" in html
        assert "video-header" in html

    def test_missing_date_returns_404(self, staged_app) -> None:
        resp = staged_app.get("/2099-01-01")
        assert resp.status_code == 404


class TestServeCrop:
    def test_returns_jpeg(self, staged_app) -> None:
        # First get the detection IDs from the page
        resp = staged_app.get("/2026-03-06")
        html = resp.data.decode()

        # Find a detection ID from the page
        import re

        match = re.search(r'data-id="([^"]+)"', html)
        assert match
        det_id = match.group(1)

        resp = staged_app.get(f"/crop/2026-03-06/{det_id}")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"

    def test_missing_detection_returns_404(self, staged_app) -> None:
        resp = staged_app.get("/crop/2026-03-06/nonexistent_id")
        assert resp.status_code == 404


class TestSubmit:
    def test_submit_returns_ok(self, staged_app) -> None:
        payload = {
            "days": {
                "2026-03-06": {
                    "selected": ["veh_0_Motorcycle"],
                    "classifications": {"veh_0_Motorcycle": "motorbike"},
                }
            }
        }
        resp = staged_app.post(
            "/submit",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"

    def test_submit_invalid_payload(self, staged_app) -> None:
        resp = staged_app.post(
            "/submit",
            data=json.dumps({"invalid": True}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestExit:
    def test_exit_returns_ok(self, staged_app) -> None:
        resp = staged_app.post("/exit")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"


class TestServeFrame:
    def test_serves_frame_png(self, staged_app) -> None:
        resp = staged_app.get("/frame/2026-03-06/frame_0.png")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_missing_frame_returns_404(self, staged_app) -> None:
        resp = staged_app.get("/frame/2026-03-06/nonexistent.png")
        assert resp.status_code == 404


class TestServeVideo:
    def test_serves_mp4(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        video_dir = tmp_path / "output" / "video"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "2026-03-06_12-00-00.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x1cftypisom")  # minimal MP4 header

        app = _create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/video/video/2026-03-06_12-00-00.mp4")
        assert resp.status_code == 200
        assert resp.content_type == "video/mp4"

    def test_missing_video_returns_404(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        app = _create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/video/video/nonexistent.mp4")
        assert resp.status_code == 404


class TestSortDetections:
    def test_confirmed_before_debug(self) -> None:
        detections = [
            {"section": "veh_debug", "class_name": "Motorcycle", "confidence": 0.9, "speed": None},
            {"section": "confirmed", "class_name": "Motorcycle", "confidence": 0.8, "speed": None},
        ]
        result = _sort_detections(detections)
        assert result[0]["section"] == "confirmed"
        assert result[1]["section"] == "veh_debug"

    def test_veh_debug_motorcycle_before_bicycle(self) -> None:
        detections = [
            {"section": "veh_debug", "class_name": "Bicycle", "confidence": 0.9, "speed": None},
            {"section": "veh_debug", "class_name": "Motorcycle", "confidence": 0.5, "speed": None},
        ]
        result = _sort_detections(detections)
        assert result[0]["class_name"] == "Motorcycle"
        assert result[1]["class_name"] == "Bicycle"

    def test_veh_debug_sorted_by_confidence_within_class(self) -> None:
        detections = [
            {"section": "veh_debug", "class_name": "Motorcycle", "confidence": 0.5, "speed": None},
            {"section": "veh_debug", "class_name": "Motorcycle", "confidence": 0.9, "speed": None},
            {"section": "veh_debug", "class_name": "Motorcycle", "confidence": 0.7, "speed": None},
        ]
        result = _sort_detections(detections)
        confidences = [d["confidence"] for d in result]
        assert confidences == [0.9, 0.7, 0.5]

    def test_hsp_debug_sorted_by_speed_descending(self) -> None:
        detections = [
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.5, "speed": 100.0},
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.9, "speed": 300.0},
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.7, "speed": 200.0},
        ]
        result = _sort_detections(detections)
        speeds = [d["speed"] for d in result]
        assert speeds == [300.0, 200.0, 100.0]

    def test_hsp_debug_none_speed_treated_as_zero(self) -> None:
        detections = [
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.5, "speed": None},
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.9, "speed": 150.0},
        ]
        result = _sort_detections(detections)
        assert result[0]["speed"] == 150.0
        assert result[1]["speed"] is None

    def test_full_ordering_confirmed_then_veh_then_hsp(self) -> None:
        detections = [
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.8, "speed": 200.0},
            {"section": "veh_debug", "class_name": "Motorcycle", "confidence": 0.7, "speed": None},
            {"section": "confirmed", "class_name": "Motorcycle", "confidence": 0.9, "speed": None},
            {"section": "veh_debug", "class_name": "Bicycle", "confidence": 0.6, "speed": None},
            {"section": "hsp_debug", "class_name": "HSP", "confidence": 0.5, "speed": 300.0},
        ]
        result = _sort_detections(detections)
        sections = [d["section"] for d in result]
        assert sections == ["confirmed", "veh_debug", "veh_debug", "hsp_debug", "hsp_debug"]
        # VEH: Motorcycle before Bicycle
        assert result[1]["class_name"] == "Motorcycle"
        assert result[2]["class_name"] == "Bicycle"
        # HSP: 300 before 200
        assert result[3]["speed"] == 300.0
        assert result[4]["speed"] == 200.0


class TestVideoReviewScore:
    def test_motorcycle_score(self) -> None:
        dets = [{"source": "veh", "class_name": "Motorcycle", "confidence": 0.8, "speed": None}]
        assert _video_review_score(dets) == pytest.approx(1.8)

    def test_bicycle_score(self) -> None:
        dets = [{"source": "veh", "class_name": "Bicycle", "confidence": 0.9, "speed": None}]
        assert _video_review_score(dets) == pytest.approx(1.4)

    def test_hsp_score(self) -> None:
        dets = [{"source": "hsp", "class_name": "HSP", "confidence": 0.5, "speed": 450.0}]
        assert _video_review_score(dets) == pytest.approx(1.5)

    def test_none_confidence_treated_as_zero(self) -> None:
        dets = [{"source": "veh", "class_name": "Motorcycle", "confidence": None, "speed": None}]
        assert _video_review_score(dets) == pytest.approx(1.0)

    def test_none_speed_treated_as_zero(self) -> None:
        dets = [{"source": "hsp", "class_name": "HSP", "confidence": 0.5, "speed": None}]
        assert _video_review_score(dets) == pytest.approx(0.0)

    def test_max_across_detections(self) -> None:
        dets = [
            {"source": "veh", "class_name": "Bicycle", "confidence": 0.5, "speed": None},
            {"source": "veh", "class_name": "Motorcycle", "confidence": 0.8, "speed": None},
        ]
        assert _video_review_score(dets) == pytest.approx(1.8)


class TestGroupByVideo:
    def test_groups_by_mp4_filename(self) -> None:
        detections = [
            {
                "id": "1",
                "mp4_filename": "a.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "veh_debug",
                "source": "veh",
                "class_name": "Motorcycle",
                "confidence": 0.8,
                "speed": None,
            },
            {
                "id": "2",
                "mp4_filename": "a.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "hsp_debug",
                "source": "hsp",
                "class_name": "HSP",
                "confidence": 0.5,
                "speed": 300.0,
            },
            {
                "id": "3",
                "mp4_filename": "b.mp4",
                "event_time_iso": "2026-03-06T12:05:00",
                "section": "veh_debug",
                "source": "veh",
                "class_name": "Bicycle",
                "confidence": 0.6,
                "speed": None,
            },
        ]
        groups = _group_by_video(detections)
        assert len(groups) == 2
        mp4s = [g["mp4_filename"] for g in groups]
        assert "a.mp4" in mp4s
        assert "b.mp4" in mp4s

    def test_confirmed_groups_first(self) -> None:
        detections = [
            {
                "id": "1",
                "mp4_filename": "unconfirmed.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "veh_debug",
                "source": "veh",
                "class_name": "Motorcycle",
                "confidence": 0.9,
                "speed": None,
            },
            {
                "id": "2",
                "mp4_filename": "confirmed.mp4",
                "event_time_iso": "2026-03-06T12:05:00",
                "section": "confirmed",
                "source": "veh",
                "class_name": "Motorcycle",
                "confidence": 0.5,
                "speed": None,
            },
        ]
        groups = _group_by_video(detections)
        assert groups[0]["mp4_filename"] == "confirmed.mp4"
        assert groups[0]["has_confirmed"] is True
        assert groups[1]["mp4_filename"] == "unconfirmed.mp4"
        assert groups[1]["has_confirmed"] is False

    def test_score_based_sorting(self) -> None:
        detections = [
            {
                "id": "1",
                "mp4_filename": "low.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "veh_debug",
                "source": "veh",
                "class_name": "Bicycle",
                "confidence": 0.3,
                "speed": None,
            },
            {
                "id": "2",
                "mp4_filename": "high.mp4",
                "event_time_iso": "2026-03-06T12:05:00",
                "section": "veh_debug",
                "source": "veh",
                "class_name": "Motorcycle",
                "confidence": 0.9,
                "speed": None,
            },
        ]
        groups = _group_by_video(detections)
        assert groups[0]["mp4_filename"] == "high.mp4"
        assert groups[1]["mp4_filename"] == "low.mp4"

    def test_within_group_ordering(self) -> None:
        detections = [
            {
                "id": "1",
                "mp4_filename": "a.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "hsp_debug",
                "source": "hsp",
                "class_name": "HSP",
                "confidence": 0.5,
                "speed": 200.0,
            },
            {
                "id": "2",
                "mp4_filename": "a.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "confirmed",
                "source": "veh",
                "class_name": "Motorcycle",
                "confidence": 0.8,
                "speed": None,
            },
        ]
        groups = _group_by_video(detections)
        assert len(groups) == 1
        # Confirmed should come first within the group
        assert groups[0]["detections"][0]["section"] == "confirmed"
        assert groups[0]["detections"][1]["section"] == "hsp_debug"

    def test_empty_input(self) -> None:
        groups = _group_by_video([])
        assert groups == []

    def test_mixed_veh_and_hsp(self) -> None:
        detections = [
            {
                "id": "1",
                "mp4_filename": "mix.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "veh_debug",
                "source": "veh",
                "class_name": "Motorcycle",
                "confidence": 0.7,
                "speed": None,
            },
            {
                "id": "2",
                "mp4_filename": "mix.mp4",
                "event_time_iso": "2026-03-06T12:00:00",
                "section": "hsp_debug",
                "source": "hsp",
                "class_name": "HSP",
                "confidence": 0.5,
                "speed": 350.0,
            },
        ]
        groups = _group_by_video(detections)
        assert len(groups) == 1
        assert len(groups[0]["detections"]) == 2
        # VEH debug sorts before HSP debug
        assert groups[0]["detections"][0]["source"] == "veh"
        assert groups[0]["detections"][1]["source"] == "hsp"
