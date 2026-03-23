from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.notification.html_report import ClipReport
from lakeside_sentinel.review.staging import (
    cleanup_staging,
    cleanup_videos_for_date,
    discover_unreviewed,
    load_frame,
    load_staged_detections,
    rebuild_clip_reports,
    stage_detections,
)


def _make_detection(
    class_name: str = "Motorcycle",
    confidence: float = 0.9,
    verification_status: str | None = None,
    speed: float | None = None,
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
        speed=speed,
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


class TestStagingRoundtrip:
    def test_stage_and_load_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]
        veh_debug = [_make_clip_report({"Motorcycle": det})]
        hsp_debug = [_make_clip_report()]

        staging_dir = stage_detections("2026-03-06", merged, veh_debug, hsp_debug, 0.2)

        assert staging_dir.exists()
        assert (staging_dir / "staging.json").exists()

        data = load_staged_detections(staging_dir)
        assert data["date_str"] == "2026-03-06"
        assert data["crop_padding"] == 0.2
        assert len(data["detections"]) >= 1

    def test_detection_fields_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(
            class_name="Bicycle",
            confidence=0.75,
            verification_status="confirmed",
            speed=None,
        )
        merged = [_make_clip_report({"Bicycle": det})]

        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2)
        data = load_staged_detections(staging_dir)

        d = data["detections"][0]
        assert d["class_name"] == "Bicycle"
        assert d["confidence"] == 0.75
        assert d["verification_status"] == "confirmed"
        assert d["bbox"] == [10.0, 20.0, 90.0, 80.0]
        assert d["frame_height"] == 100
        assert d["frame_width"] == 200


class TestTotalClips:
    def test_total_clips_stored_in_staging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]
        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2, total_clips=47)

        data = load_staged_detections(staging_dir)
        assert data["total_clips"] == 47

    def test_total_clips_none_omits_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]
        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2, total_clips=None)

        data = load_staged_detections(staging_dir)
        assert "total_clips" not in data

    def test_old_staging_without_total_clips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]
        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2)

        data = load_staged_detections(staging_dir)
        assert data.get("total_clips") is None


class TestSectionAssignment:
    def test_confirmed_detection_in_confirmed_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]

        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2)
        data = load_staged_detections(staging_dir)

        confirmed = [d for d in data["detections"] if d["section"] == "confirmed"]
        assert len(confirmed) == 1
        assert confirmed[0]["class_name"] == "Motorcycle"

    def test_unconfirmed_veh_in_debug_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status=None)
        veh_debug = [_make_clip_report({"Motorcycle": det})]

        staging_dir = stage_detections("2026-03-06", [], veh_debug, [], 0.2)
        data = load_staged_detections(staging_dir)

        veh_debug_dets = [d for d in data["detections"] if d["section"] == "veh_debug"]
        assert len(veh_debug_dets) == 1

    def test_hsp_debug_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(class_name="HSP", speed=350.0)
        hsp_debug = [_make_clip_report({"HSP": det})]

        staging_dir = stage_detections("2026-03-06", [], [], hsp_debug, 0.2)
        data = load_staged_detections(staging_dir)

        hsp_debug_dets = [d for d in data["detections"] if d["section"] == "hsp_debug"]
        assert len(hsp_debug_dets) == 1
        assert hsp_debug_dets[0]["source"] == "hsp"


class TestFrameDeduplication:
    def test_same_frame_saved_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        shared_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        det1 = _make_detection(class_name="Motorcycle", frame=shared_frame)
        det2 = _make_detection(class_name="Bicycle", confidence=0.6, frame=shared_frame)

        merged = [_make_clip_report({"Motorcycle": det1})]
        veh_debug = [_make_clip_report({"Motorcycle": det1, "Bicycle": det2})]

        staging_dir = stage_detections("2026-03-06", merged, veh_debug, [], 0.2)

        # Count PNG files
        png_files = list(staging_dir.glob("*.png"))
        assert len(png_files) == 1


class TestRebuildClipReports:
    def test_rebuild_filters_by_selected_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det1 = _make_detection(class_name="Motorcycle", verification_status="confirmed")
        det2 = _make_detection(class_name="Bicycle", confidence=0.6)
        merged = [_make_clip_report({"Motorcycle": det1})]
        veh_debug = [_make_clip_report({"Motorcycle": det1, "Bicycle": det2})]

        staging_dir = stage_detections("2026-03-06", merged, veh_debug, [], 0.2)
        data = load_staged_detections(staging_dir)

        # Select only the confirmed detection
        confirmed_ids = {d["id"] for d in data["detections"] if d["section"] == "confirmed"}
        reports = rebuild_clip_reports(staging_dir, confirmed_ids)

        assert len(reports) == 1
        assert "Motorcycle" in reports[0].class_detections
        assert "Bicycle" not in reports[0].class_detections

    def test_rebuild_returns_empty_for_no_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]

        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2)
        reports = rebuild_clip_reports(staging_dir, set())

        assert len(reports) == 0


class TestDiscoverUnreviewed:
    def test_finds_staging_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]
        stage_detections("2026-03-05", merged, [], [], 0.2)
        stage_detections("2026-03-06", merged, [], [], 0.2)

        dirs = discover_unreviewed()
        assert len(dirs) == 2
        assert dirs[0].name == "2026-03-05"
        assert dirs[1].name == "2026-03-06"

    def test_returns_empty_when_no_staging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        dirs = discover_unreviewed()
        assert dirs == []


class TestCleanupStaging:
    def test_removes_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        det = _make_detection(verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]
        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2)

        assert staging_dir.exists()
        cleanup_staging(staging_dir)
        assert not staging_dir.exists()


class TestCleanupVideosForDate:
    def test_deletes_matching_videos(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        video_dir = tmp_path / "output" / "video"
        video_dir.mkdir(parents=True)

        # Create videos for the target date and another date
        (video_dir / "2026-03-06_12-00-00.mp4").write_bytes(b"clip1")
        (video_dir / "2026-03-06_14-30-00.mp4").write_bytes(b"clip2")
        (video_dir / "2026-03-07_09-00-00.mp4").write_bytes(b"other")

        cleanup_videos_for_date("2026-03-06")

        assert not (video_dir / "2026-03-06_12-00-00.mp4").exists()
        assert not (video_dir / "2026-03-06_14-30-00.mp4").exists()
        assert (video_dir / "2026-03-07_09-00-00.mp4").exists()

    def test_no_op_when_no_video_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        # Should not raise
        cleanup_videos_for_date("2026-03-06")

    def test_no_op_when_no_matching_videos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        video_dir = tmp_path / "output" / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "2026-03-07_09-00-00.mp4").write_bytes(b"other")

        cleanup_videos_for_date("2026-03-06")

        assert (video_dir / "2026-03-07_09-00-00.mp4").exists()


class TestLoadFrame:
    def test_loads_saved_frame(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        frame = np.full((50, 100, 3), 128, dtype=np.uint8)
        det = _make_detection(frame=frame, verification_status="confirmed")
        merged = [_make_clip_report({"Motorcycle": det})]

        staging_dir = stage_detections("2026-03-06", merged, [], [], 0.2)
        data = load_staged_detections(staging_dir)
        loaded = load_frame(staging_dir, data["detections"][0]["frame_filename"])

        assert loaded.shape == (50, 100, 3)
