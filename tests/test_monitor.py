from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lakeside_sentinel.camera.models import CameraEvent
from lakeside_sentinel.config import Settings
from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.main import Monitor, _print_settings


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


class TestPrintSettings:
    def test_prints_all_settings_fields(
        self, mock_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_settings(mock_settings)
        output = capsys.readouterr().out
        for field_name in Settings.model_fields:
            label = field_name.replace("_", " ").title()
            assert label in output

    def test_masks_sensitive_values(
        self, mock_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_settings(mock_settings)
        output = capsys.readouterr().out
        assert mock_settings.google_master_token not in output
        assert mock_settings.resend_api_key not in output
        assert "****" in output

    def test_shows_non_sensitive_values(
        self, mock_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_settings(mock_settings)
        output = capsys.readouterr().out
        assert str(mock_settings.yolo_model) in output
        assert str(mock_settings.veh_fps_sample) in output
        assert mock_settings.alert_email_to in output


class TestRunCache:
    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_cached_files_are_read_from_disk(
        self,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
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

        mock_detector_cls.return_value.detect_detailed.return_value = (None, {})

        # Pre-populate the cache directory with the expected file
        cache_dir = tmp_path / "output" / "video"
        cache_dir.mkdir(parents=True)
        local_time = event.start_time.astimezone()
        filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        (cache_dir / filename).write_bytes(b"cached_data")

        monitor = Monitor(mock_settings)
        monitor.run()

        # download_clip should NOT have been called since the file was cached
        mock_api.download_clip.assert_not_called()

    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_missing_files_are_downloaded_and_written(
        self,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
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

        mock_detector_cls.return_value.detect_detailed.return_value = (None, {})

        monitor = Monitor(mock_settings)
        monitor.run()

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
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value="<html></html>")
    def test_sub_threshold_detections_excluded_from_clip_report(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
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
        mock_detector_cls.return_value.detect_detailed.return_value = (
            above_threshold,
            {"Bicycle": sub_threshold, "Motorcycle": above_threshold},
        )

        monitor = Monitor(mock_settings)
        monitor.run()

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
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value="<html></html>")
    def test_claude_verification_filters_rejected(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
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
        mock_detector_cls.return_value.detect_detailed.return_value = (
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
        monitor.run(use_claude=True)

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
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    @patch("lakeside_sentinel.main.generate_report", return_value="<html></html>")
    def test_claude_keep_rejected_preserves_all(
        self,
        mock_generate_report: MagicMock,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
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
        mock_detector_cls.return_value.detect_detailed.return_value = (
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
        monitor.run(use_claude=True, claude_keep_rejected=True)

        clip_reports = mock_generate_report.call_args_list[0][0][0]
        report = clip_reports[0]

        # Both should be preserved with keep-rejected
        assert "Motorcycle" in report.class_detections
        assert "Bicycle" in report.class_detections
        assert report.best_detection is not None
        assert report.best_detection.class_name == "Motorcycle"
