import logging
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lakeside_sentinel.camera.models import CameraEvent
from lakeside_sentinel.config import Settings
from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.main import (
    Monitor,
    _cleanup_old_files,
    _print_settings,
    _setup_file_logging,
    _warn_expiring_staging,
)
from lakeside_sentinel.notification.email_sender import EmailSender


def _make_event(hour: int = 12) -> CameraEvent:
    return CameraEvent(
        start_time=datetime(2026, 2, 28, hour, 0, 0, tzinfo=timezone.utc),
        duration=timedelta(seconds=10),
    )


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        google_master_token="token",
        google_username="user@gmail.com",
        nest_device_id="device-id",
        resend_api_key="re_test",
        alert_email_to="to@example.com",
        alert_email_from="from@example.com",
        camera_latitude=51.5074,
        camera_longitude=-0.1278,
    )


@pytest.fixture(autouse=True)
def _mock_main_yolo() -> Iterator[MagicMock]:
    """Prevent Monitor.__init__ from loading real YOLO weights in tests.

    Monitor builds a single YOLO instance that it shares with VEHDetector and
    HSPDetector. The detectors themselves are patched per-test, but the YOLO
    construction in main.py still needs to be intercepted to avoid touching
    the filesystem / network.
    """
    with patch("lakeside_sentinel.main.YOLO") as m:
        yield m


class TestPrintSettings:
    def test_prints_all_settings_fields(
        self, mock_settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _print_settings(mock_settings)
        for field_name in Settings.model_fields:
            label = field_name.replace("_", " ").title()
            assert label in caplog.text

    def test_masks_sensitive_values(
        self, mock_settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _print_settings(mock_settings)
        assert mock_settings.google_master_token not in caplog.text
        assert mock_settings.resend_api_key not in caplog.text
        assert "****" in caplog.text

    def test_shows_non_sensitive_values(
        self, mock_settings: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _print_settings(mock_settings)
        assert str(mock_settings.yolo_model) in caplog.text
        assert str(mock_settings.veh_fps_sample) in caplog.text
        assert mock_settings.alert_email_to in caplog.text


class TestRunCache:
    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_cached_files_are_read_from_disk(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"downloaded_data"

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})

        # Pre-populate the cache directory with the expected file
        cache_dir = tmp_path / "output" / "video"
        cache_dir.mkdir(parents=True)
        local_time = event.start_time.astimezone()
        filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        (cache_dir / filename).write_bytes(b"cached_data")

        monitor = Monitor(mock_settings)
        monitor.run_debug_veh()

        # download_clip should NOT have been called since the file was cached
        mock_api.download_clip.assert_not_called()

    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_missing_files_are_downloaded_and_written(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        event = _make_event(hour=14)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"fresh_download"

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})

        monitor = Monitor(mock_settings)
        monitor.run_debug_veh()

        # download_clip should have been called since no cached file exists
        mock_api.download_clip.assert_called_once_with(event)

        # File should have been written to disk
        local_time = event.start_time.astimezone()
        filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        cached_file = tmp_path / "output" / "video" / filename
        assert cached_file.exists()
        assert cached_file.read_bytes() == b"fresh_download"


class TestThresholdFiltering:
    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value=("<html></html>", []))
    def test_sub_threshold_detections_excluded_from_clip_report(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        sub_threshold = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.15,
            class_name="Bicycle",
        )
        above_threshold = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.85,
            class_name="Motorcycle",
        )
        mock_veh_cls.return_value.detect_detailed.return_value = (
            above_threshold,
            {"Bicycle": sub_threshold, "Motorcycle": above_threshold},
        )

        monitor = Monitor(mock_settings)
        monitor.run_debug_veh()

        # generate_report is called once (no email), check the first call
        clip_reports = mock_generate_report.call_args_list[0][0][0]
        assert len(clip_reports) == 1
        report = clip_reports[0]

        # Sub-threshold Bicycle (0.15 < 0.4 default) should be filtered out
        assert "Bicycle" not in report.class_detections
        # Above-threshold Motorcycle should remain
        assert "Motorcycle" in report.class_detections


class TestClaudeVerification:
    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value=("<html></html>", []))
    def test_claude_verification_filters_rejected(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        moto_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.85,
            class_name="Motorcycle",
        )
        bike_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.65,
            class_name="Bicycle",
        )
        mock_veh_cls.return_value.detect_detailed.return_value = (
            moto_det,
            {"Motorcycle": moto_det, "Bicycle": bike_det},
        )

        # Claude verifier confirms motorcycle, rejects bicycle
        def mock_verify_detections(detections: dict[str, Detection]) -> dict[str, Detection]:
            for name, det in detections.items():
                if name == "Motorcycle":
                    det.verification_status = "confirmed"
                else:
                    det.verification_status = "rejected"
            return {"Motorcycle": detections["Motorcycle"]}

        mock_verifier_cls.return_value.verify_detections.side_effect = mock_verify_detections

        monitor = Monitor(mock_settings)
        monitor.run_debug_veh(use_claude=True)

        clip_reports = mock_generate_report.call_args_list[0][0][0]
        assert len(clip_reports) == 1
        report = clip_reports[0]

        # Bicycle should be filtered out by Claude verification
        assert "Bicycle" not in report.class_detections
        assert "Motorcycle" in report.class_detections
        assert report.best_detection is not None
        assert report.best_detection.verification_status == "confirmed"

    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value=("<html></html>", []))
    def test_claude_keep_rejected_preserves_all(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        moto_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.85,
            class_name="Motorcycle",
        )
        bike_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.65,
            class_name="Bicycle",
        )
        mock_veh_cls.return_value.detect_detailed.return_value = (
            moto_det,
            {"Motorcycle": moto_det, "Bicycle": bike_det},
        )

        # Claude rejects bicycle but keep-rejected is on
        def mock_verify_detections(detections: dict[str, Detection]) -> dict[str, Detection]:
            for name, det in detections.items():
                if name == "Motorcycle":
                    det.verification_status = "confirmed"
                else:
                    det.verification_status = "rejected"
            return {"Motorcycle": detections["Motorcycle"]}

        mock_verifier_cls.return_value.verify_detections.side_effect = mock_verify_detections

        monitor = Monitor(mock_settings)
        monitor.run_debug_veh(use_claude=True, claude_keep_rejected=True)

        clip_reports = mock_generate_report.call_args_list[0][0][0]
        report = clip_reports[0]

        # Both should be preserved with keep-rejected
        assert "Motorcycle" in report.class_detections
        assert "Bicycle" in report.class_detections
        assert report.best_detection is not None
        assert report.best_detection.class_name == "Motorcycle"


class TestDefaultMode:
    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_default_mode_runs_both_detectors_and_stages(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        moto_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.85,
            class_name="Motorcycle",
        )
        mock_veh_cls.return_value.detect_detailed.return_value = (
            moto_det,
            {"Motorcycle": moto_det},
        )

        hsp_det = Detection(
            frame=dummy_frame,
            bbox=(20.0, 20.0, 80.0, 80.0),
            confidence=0.70,
            class_name="HSP",
            speed=300.0,
        )
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = hsp_det

        # Claude confirms everything
        def mock_verify(detections: dict[str, Detection]) -> dict[str, Detection]:
            for det in detections.values():
                det.verification_status = "confirmed"
            return detections

        mock_verifier_cls.return_value.verify_detections.side_effect = mock_verify

        monitor = Monitor(mock_settings)
        monitor.run(target_date=datetime(2026, 2, 28).date())

        # Both VEH and HSP detectors should have been called
        mock_veh_cls.return_value.detect_detailed.assert_called_once()
        mock_hsp_cls.return_value.detect.assert_called_once()

        # Claude verification should have been called
        mock_verifier_cls.return_value.verify_detections.assert_called()

        # Staging dir should exist
        staging_dir = tmp_path / "output" / "staging" / "2026-02-28"
        assert staging_dir.exists()

        # No reports should be generated (staging only)
        report_path = tmp_path / "output" / "report-2026-02-28.html"
        assert not report_path.exists()

        # No email should be sent
        mock_email_cls.return_value.send_report.assert_not_called()

    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_default_mode_skips_existing_staging(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        # Pre-create staging dir
        staging_dir = tmp_path / "output" / "staging" / "2026-02-28"
        staging_dir.mkdir(parents=True)

        monitor = Monitor(mock_settings)
        monitor.run(target_date=datetime(2026, 2, 28).date())

        # Pipeline should NOT have run (no events fetched)
        mock_api_cls.return_value.get_events.assert_not_called()


class TestDefaultModeNoReportsOrEmail:
    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_no_email_sent_in_default_mode(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None

        mock_verifier_cls.return_value.verify_detections.return_value = {}

        monitor = Monitor(mock_settings)
        monitor.run(target_date=datetime(2026, 2, 28).date())

        # No email should be sent in default mode
        mock_email_cls.return_value.send_report.assert_not_called()

    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_no_reports_generated_in_default_mode(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None

        mock_verifier_cls.return_value.verify_detections.return_value = {}

        monitor = Monitor(mock_settings)
        monitor.run(target_date=datetime(2026, 2, 28).date())

        # No report files should exist
        assert not (tmp_path / "output" / "report-2026-02-28.html").exists()
        assert not (tmp_path / "output" / "report-veh-2026-02-28.html").exists()
        assert not (tmp_path / "output" / "report-hsp-2026-02-28.html").exists()


class TestMergeClipReports:
    def test_merge_disjoint_detections(self) -> None:
        from lakeside_sentinel.notification.html_report import ClipReport

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event_time = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
        mp4_fn = "video/2026-02-28_12-00-00.mp4"

        moto_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.85,
            class_name="Motorcycle",
        )
        hsp_det = Detection(
            frame=dummy_frame,
            bbox=(20.0, 20.0, 80.0, 80.0),
            confidence=0.70,
            class_name="HSP",
            speed=300.0,
        )

        veh_reports = [
            ClipReport(
                event_time=event_time,
                mp4_filename=mp4_fn,
                best_detection=moto_det,
                class_detections={"Motorcycle": moto_det},
            )
        ]
        hsp_reports = [
            ClipReport(
                event_time=event_time,
                mp4_filename=mp4_fn,
                best_detection=hsp_det,
                class_detections={"HSP": hsp_det},
            )
        ]

        merged = Monitor._merge_clip_reports(veh_reports, hsp_reports)
        assert len(merged) == 1
        assert "Motorcycle" in merged[0].class_detections
        assert "HSP" in merged[0].class_detections
        # Best should be highest-confidence unverified detection
        assert merged[0].best_detection is not None
        assert merged[0].best_detection.confidence == 0.85

    def test_merge_veh_only_clip(self) -> None:
        from lakeside_sentinel.notification.html_report import ClipReport

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event_time = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)

        moto_det = Detection(
            frame=dummy_frame,
            bbox=(10.0, 10.0, 90.0, 90.0),
            confidence=0.85,
            class_name="Motorcycle",
        )

        veh_reports = [
            ClipReport(
                event_time=event_time,
                mp4_filename="video/a.mp4",
                best_detection=moto_det,
                class_detections={"Motorcycle": moto_det},
            )
        ]
        hsp_reports = [
            ClipReport(
                event_time=event_time,
                mp4_filename="video/b.mp4",
                best_detection=None,
                class_detections={},
            )
        ]

        merged = Monitor._merge_clip_reports(veh_reports, hsp_reports)
        assert len(merged) == 1
        # Since HSP has a different filename, VEH report is returned as-is
        assert "Motorcycle" in merged[0].class_detections
        assert "HSP" not in merged[0].class_detections


class TestStepLabels:
    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_default_mode_step_labels(
        self,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_verifier_cls: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_settings.anthropic_api_key = "test-key"

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None
        mock_verifier_cls.return_value.verify_detections.return_value = {}

        monitor = Monitor(mock_settings)
        with caplog.at_level(logging.INFO):
            monitor.run(target_date=datetime(2026, 2, 28).date())

        assert "[1/5]" in caplog.text
        assert "[2/5]" in caplog.text
        assert "[3/5]" in caplog.text
        assert "[4/5]" in caplog.text

    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value=("<html></html>", []))
    def test_debug_veh_without_claude_has_4_steps(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})

        monitor = Monitor(mock_settings)
        with caplog.at_level(logging.INFO):
            monitor.run_debug_veh()

        assert "[1/4]" in caplog.text
        assert "[2/4]" in caplog.text
        assert "[3/4]" in caplog.text
        assert "[4/4]" in caplog.text
        assert "[3.5/4]" not in caplog.text


class TestTimestampsAndDuration:
    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value=("<html></html>", []))
    def test_done_in_appears_in_output(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_veh_cls: MagicMock,
        mock_hsp_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
        mock_extract_frames: MagicMock,
        mock_crop_to_roi: MagicMock,
        mock_webbrowser_open: MagicMock,
        mock_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        event = _make_event(hour=12)
        mock_api = mock_api_cls.return_value
        mock_api.get_events.return_value = [event]
        mock_api.download_clip.return_value = b"video_data"

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract_frames.return_value = [dummy_frame]
        mock_crop_to_roi.return_value = [dummy_frame]

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})

        monitor = Monitor(mock_settings)
        with caplog.at_level(logging.INFO):
            monitor.run_debug_veh()

        assert "Done in" in caplog.text


class TestFileLogging:
    def test_setup_file_logging_creates_log_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        try:
            _setup_file_logging()
            log_dir = tmp_path / "output" / "logs"
            assert log_dir.exists()
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1

            # Verify the handler writes content
            test_logger = logging.getLogger("test_file_logging")
            test_logger.setLevel(logging.INFO)
            root_logger.setLevel(logging.INFO)
            test_logger.info("test message")
            # Flush all handlers to ensure content is written
            for handler in root_logger.handlers:
                handler.flush()
            content = log_files[0].read_text()
            assert "test message" in content
        finally:
            # Clean up added handlers
            for handler in root_logger.handlers[:]:
                if handler not in original_handlers:
                    root_logger.removeHandler(handler)
                    handler.close()

    def test_cleanup_old_files_deletes_old(self, tmp_path: Path) -> None:
        old_file = tmp_path / "old.log"
        old_file.write_text("old")
        import os

        old_mtime = time.time() - (8 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        new_file = tmp_path / "new.log"
        new_file.write_text("new")

        _cleanup_old_files(tmp_path, ".log", 7)

        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_old_files_ignores_wrong_suffix(self, tmp_path: Path) -> None:
        old_file = tmp_path / "old.txt"
        old_file.write_text("old")
        import os

        old_mtime = time.time() - (8 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        _cleanup_old_files(tmp_path, ".log", 7)

        assert old_file.exists()

    def test_cleanup_old_files_nonexistent_directory(self) -> None:
        _cleanup_old_files(Path("/nonexistent/directory"), ".log", 7)


class TestExpiryWarning:
    def _create_staging_dir(
        self, tmp_path: Path, date_str: str, age_days: int, det_count: int = 3
    ) -> Path:
        """Create a staging dir with staging.json and set its mtime."""
        import json
        import os

        staging_dir = tmp_path / "output" / "staging" / date_str
        staging_dir.mkdir(parents=True, exist_ok=True)
        detections = [{"id": f"det_{i}"} for i in range(det_count)]
        (staging_dir / "staging.json").write_text(
            json.dumps({"date_str": date_str, "detections": detections})
        )
        old_mtime = time.time() - (age_days * 86400)
        os.utime(staging_dir, (old_mtime, old_mtime))
        return staging_dir

    @patch.object(EmailSender, "send_report", return_value="email-id-123")
    def test_sends_email_for_expiring_staging(
        self,
        mock_send: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._create_staging_dir(tmp_path, "2026-02-28", age_days=12, det_count=5)

        sender = EmailSender(api_key="test", from_email="from@test.com", to_email="to@test.com")
        _warn_expiring_staging(
            tmp_path / "output" / "staging",
            max_age_days=14,
            warning_days=3,
            email_sender=sender,
        )

        mock_send.assert_called_once()
        html = mock_send.call_args[0][0]
        assert "2026-02-28" in html
        assert "5" in html
        assert "2" in html  # days remaining: 14 - 12 = 2

    @patch.object(EmailSender, "send_report", return_value="email-id-123")
    def test_skips_fresh_staging(
        self,
        mock_send: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._create_staging_dir(tmp_path, "2026-03-10", age_days=2, det_count=3)

        sender = EmailSender(api_key="test", from_email="from@test.com", to_email="to@test.com")
        _warn_expiring_staging(
            tmp_path / "output" / "staging",
            max_age_days=14,
            warning_days=3,
            email_sender=sender,
        )

        mock_send.assert_not_called()

    @patch.object(EmailSender, "send_report", return_value="email-id-123")
    def test_skips_nonexistent_dir(
        self,
        mock_send: MagicMock,
        tmp_path: Path,
    ) -> None:
        sender = EmailSender(api_key="test", from_email="from@test.com", to_email="to@test.com")
        _warn_expiring_staging(
            tmp_path / "nonexistent" / "staging",
            max_age_days=14,
            warning_days=3,
            email_sender=sender,
        )

        mock_send.assert_not_called()

    @patch.object(EmailSender, "send_report", return_value="email-id-123")
    def test_skips_dirs_without_staging_json(
        self,
        mock_send: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Create dir without staging.json
        staging_dir = tmp_path / "output" / "staging" / "2026-02-28"
        staging_dir.mkdir(parents=True, exist_ok=True)
        import os

        old_mtime = time.time() - (12 * 86400)
        os.utime(staging_dir, (old_mtime, old_mtime))

        sender = EmailSender(api_key="test", from_email="from@test.com", to_email="to@test.com")
        _warn_expiring_staging(
            tmp_path / "output" / "staging",
            max_age_days=14,
            warning_days=3,
            email_sender=sender,
        )

        mock_send.assert_not_called()

    @patch.object(EmailSender, "send_report", return_value=None)
    def test_handles_email_failure(
        self,
        mock_send: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._create_staging_dir(tmp_path, "2026-02-28", age_days=12, det_count=2)

        sender = EmailSender(api_key="test", from_email="from@test.com", to_email="to@test.com")
        _warn_expiring_staging(
            tmp_path / "output" / "staging",
            max_age_days=14,
            warning_days=3,
            email_sender=sender,
        )

        mock_send.assert_called_once()
        # No exception raised
