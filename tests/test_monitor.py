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


class TestPresentMode:
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
    def test_present_mode_runs_both_detectors_with_claude(
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
        monitor.run_present()

        # Both VEH and HSP detectors should have been called
        mock_veh_cls.return_value.detect_detailed.assert_called_once()
        mock_hsp_cls.return_value.detect.assert_called_once()

        # Claude verification should have been called
        mock_verifier_cls.return_value.verify_detections.assert_called()

        # generate_report called 3 times: present + veh debug + hsp debug
        assert mock_generate_report.call_count == 3

        # First call is the present report
        call_kwargs = mock_generate_report.call_args_list[0][1]
        assert call_kwargs["mode"] == "present"

        # Second call is the VEH debug report
        call_kwargs = mock_generate_report.call_args_list[1][1]
        assert call_kwargs["mode"] == "veh"

        # Third call is the HSP debug report
        call_kwargs = mock_generate_report.call_args_list[2][1]
        assert call_kwargs["mode"] == "hsp"

    @patch("lakeside_sentinel.main.webbrowser.open")
    @patch("lakeside_sentinel.main.ClaudeVerifier")
    @patch("lakeside_sentinel.main.crop_to_roi")
    @patch("lakeside_sentinel.main.extract_frames")
    @patch("lakeside_sentinel.main.NestCameraAPI")
    @patch("lakeside_sentinel.main.NestAuth")
    @patch("lakeside_sentinel.main.HSPDetector")
    @patch("lakeside_sentinel.main.VEHDetector")
    @patch("lakeside_sentinel.main.EmailSender")
    def test_present_mode_report_filename(
        self,
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

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None

        # Claude verifier (nothing to verify)
        mock_verifier_cls.return_value.verify_detections.return_value = {}

        monitor = Monitor(mock_settings)
        monitor.run_present(target_date=datetime(2026, 2, 28).date())

        # Report filename should be report-{date}.html (no mode prefix)
        report_path = tmp_path / "output" / "report-2026-02-28.html"
        assert report_path.exists()

        # Debug reports should also exist
        veh_debug_path = tmp_path / "output" / "report-veh-2026-02-28.html"
        hsp_debug_path = tmp_path / "output" / "report-hsp-2026-02-28.html"
        assert veh_debug_path.exists()
        assert hsp_debug_path.exists()

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
    def test_present_mode_merges_veh_and_hsp(
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
        monitor.run_present()

        # generate_report called 3 times: present + veh debug + hsp debug
        assert mock_generate_report.call_count == 3

        # The merged present report should contain both Motorcycle and HSP detections
        clip_reports = mock_generate_report.call_args_list[0][0][0]
        assert len(clip_reports) == 1
        report = clip_reports[0]
        assert "Motorcycle" in report.class_detections
        assert "HSP" in report.class_detections


class TestPresentModeDebugReports:
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
    def test_veh_debug_report_includes_below_threshold_detections(
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

        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None

        def mock_verify(detections: dict[str, Detection]) -> dict[str, Detection]:
            for det in detections.values():
                det.verification_status = "confirmed"
            return detections

        mock_verifier_cls.return_value.verify_detections.side_effect = mock_verify

        monitor = Monitor(mock_settings)
        monitor.run_present(target_date=datetime(2026, 2, 28).date())

        # Call 0 = present, call 1 = VEH debug, call 2 = HSP debug
        assert mock_generate_report.call_count == 3

        # Present report should NOT have below-threshold Bicycle
        present_reports = mock_generate_report.call_args_list[0][0][0]
        assert "Bicycle" not in present_reports[0].class_detections

        # VEH debug report should include below-threshold Bicycle
        veh_debug_reports = mock_generate_report.call_args_list[1][0][0]
        assert len(veh_debug_reports) == 1
        assert "Bicycle" in veh_debug_reports[0].class_detections
        assert "Motorcycle" in veh_debug_reports[0].class_detections

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
    def test_hsp_debug_report_includes_below_threshold_tracks(
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

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})

        # HSP: below-threshold track (detect returns None, but detect_all_tracks
        # returns a track). We mock a PersonTrack with displacement below threshold.
        from lakeside_sentinel.detection.hsp_detector import PersonTrack, TrackPoint

        slow_track = PersonTrack(
            points=[
                TrackPoint(
                    frame_index=0,
                    centroid_x=100.0,
                    centroid_y=100.0,
                    bbox=(80.0, 80.0, 120.0, 120.0),
                    confidence=0.75,
                    frame=dummy_frame,
                ),
                TrackPoint(
                    frame_index=1,
                    centroid_x=110.0,
                    centroid_y=100.0,
                    bbox=(90.0, 80.0, 130.0, 120.0),
                    confidence=0.80,
                    frame=dummy_frame,
                ),
            ]
        )
        mock_hsp_cls.return_value.detect_all_tracks.return_value = [slow_track]
        mock_hsp_cls.return_value.detect.return_value = None  # below threshold

        def mock_verify(detections: dict[str, Detection]) -> dict[str, Detection]:
            for det in detections.values():
                det.verification_status = "confirmed"
            return detections

        mock_verifier_cls.return_value.verify_detections.side_effect = mock_verify

        monitor = Monitor(mock_settings)
        monitor.run_present(target_date=datetime(2026, 2, 28).date())

        # Call 0 = present, call 1 = VEH debug, call 2 = HSP debug
        assert mock_generate_report.call_count == 3

        # Present report should NOT have HSP (below threshold)
        present_reports = mock_generate_report.call_args_list[0][0][0]
        assert "HSP" not in present_reports[0].class_detections

        # HSP debug report should include the below-threshold track
        hsp_debug_reports = mock_generate_report.call_args_list[2][0][0]
        assert len(hsp_debug_reports) == 1
        assert "HSP" in hsp_debug_reports[0].class_detections
        assert hsp_debug_reports[0].best_detection is not None
        assert hsp_debug_reports[0].best_detection.speed is not None

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
    def test_debug_reports_not_emailed(
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

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None

        mock_verifier_cls.return_value.verify_detections.return_value = {}
        mock_email_cls.return_value.send_report.return_value = "email-id-123"

        monitor = Monitor(mock_settings)
        monitor.run_present(send_email=True, target_date=datetime(2026, 2, 28).date())

        # Email should only be sent once (for the present report, not debug reports)
        mock_email_cls.return_value.send_report.assert_called_once()

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
    def test_debug_reports_not_opened_in_browser(
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

        mock_veh_cls.return_value.detect_detailed.return_value = (None, {})
        mock_hsp_cls.return_value.detect_all_tracks.return_value = []
        mock_hsp_cls.return_value.detect.return_value = None

        mock_verifier_cls.return_value.verify_detections.return_value = {}

        monitor = Monitor(mock_settings)
        monitor.run_present(target_date=datetime(2026, 2, 28).date())

        # Browser should only be opened once (for the present report)
        mock_webbrowser_open.assert_called_once()

        # The opened URL should be the present report, not a debug report
        opened_url = mock_webbrowser_open.call_args[0][0]
        assert "report-2026-02-28.html" in opened_url
        assert "report-veh-" not in opened_url
        assert "report-hsp-" not in opened_url


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
