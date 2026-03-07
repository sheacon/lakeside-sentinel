import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.notification.html_report import ClipReport
from lakeside_sentinel.review.server import _create_app
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
) -> ClipReport:
    if class_detections is None:
        class_detections = {}
    best = None
    if class_detections:
        best = max(class_detections.values(), key=lambda d: d.confidence)
    return ClipReport(
        event_time=datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc),
        mp4_filename="video/2026-03-06_12-00-00.mp4",
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
    def test_renders_page_with_sections(self, staged_app) -> None:
        resp = staged_app.get("/2026-03-06")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "2026-03-06" in html
        assert "Motorcycle" in html

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
