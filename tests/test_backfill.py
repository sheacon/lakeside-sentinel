from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lakeside_motorbikes.camera.models import CameraEvent
from lakeside_motorbikes.config import Settings
from lakeside_motorbikes.main import Monitor


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


class TestBackfillCache:
    @patch("lakeside_motorbikes.main.NestCameraAPI")
    @patch("lakeside_motorbikes.main.NestAuth")
    @patch("lakeside_motorbikes.main.VehicleDetector")
    @patch("lakeside_motorbikes.main.EmailSender")
    def test_cached_files_are_read_from_disk(
        self,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
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
        mock_email_cls.return_value.send_backfill_summary.return_value = None

        # Pre-populate the cache directory with the expected file
        cache_dir = tmp_path / "output" / "backfill"
        cache_dir.mkdir(parents=True)
        local_time = event.start_time.astimezone()
        filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        (cache_dir / filename).write_bytes(b"cached_data")

        monitor = Monitor(mock_settings)
        monitor.backfill(debug_dump=True)

        # download_clip should NOT have been called since the file was cached
        mock_api.download_clip.assert_not_called()

    @patch("lakeside_motorbikes.main.NestCameraAPI")
    @patch("lakeside_motorbikes.main.NestAuth")
    @patch("lakeside_motorbikes.main.VehicleDetector")
    @patch("lakeside_motorbikes.main.EmailSender")
    def test_missing_files_are_downloaded_and_written(
        self,
        mock_email_cls: MagicMock,
        mock_detector_cls: MagicMock,
        mock_auth_cls: MagicMock,
        mock_api_cls: MagicMock,
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
        mock_email_cls.return_value.send_backfill_summary.return_value = None

        monitor = Monitor(mock_settings)
        monitor.backfill(debug_dump=True)

        # download_clip should have been called since no cached file exists
        mock_api.download_clip.assert_called_once_with(event)

        # File should have been written to disk
        local_time = event.start_time.astimezone()
        filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        cached_file = tmp_path / "output" / "backfill" / filename
        assert cached_file.exists()
        assert cached_file.read_bytes() == b"fresh_download"
