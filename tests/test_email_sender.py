from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lakeside_motorbikes.notification.email_sender import EmailSender


@pytest.fixture
def sender() -> EmailSender:
    return EmailSender(
        api_key="re_test_key",
        from_email="alerts@xeroshot.org",
        to_email="user@example.com",
    )


@pytest.fixture
def dummy_image() -> np.ndarray:
    return np.zeros((100, 150, 3), dtype=np.uint8)


class TestEmailSender:
    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_send_alert_calls_resend(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "email_123"}
        event_time = datetime(2026, 2, 28, 14, 32, 5, tzinfo=timezone.utc)

        result = sender.send_alert(dummy_image, 0.85, event_time, class_name="Motorcycle")

        assert result == "email_123"
        mock_resend.Emails.send.assert_called_once()
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["to"] == ["user@example.com"]
        assert call_args["from"] == "alerts@xeroshot.org"
        assert "Motorcycle Detected" in call_args["subject"]
        assert len(call_args["attachments"]) == 1
        assert call_args["attachments"][0]["filename"] == "detection.png"

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_send_alert_includes_confidence_in_body(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "email_456"}
        event_time = datetime(2026, 2, 28, 14, 32, 5, tzinfo=timezone.utc)

        sender.send_alert(dummy_image, 0.85, event_time, class_name="Car")

        call_args = mock_resend.Emails.send.call_args[0][0]
        assert "85%" in call_args["html"]
        assert "Car Detected" in call_args["html"]

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_send_alert_default_class_name(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "email_789"}
        event_time = datetime(2026, 2, 28, 14, 32, 5, tzinfo=timezone.utc)

        sender.send_alert(dummy_image, 0.85, event_time)

        call_args = mock_resend.Emails.send.call_args[0][0]
        assert "Vehicle Detected" in call_args["subject"]

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_send_alert_handles_exception(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.side_effect = Exception("API error")
        event_time = datetime(2026, 2, 28, 14, 32, 5, tzinfo=timezone.utc)

        result = sender.send_alert(dummy_image, 0.85, event_time)

        assert result is None


class TestBackfillSummary:
    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_empty_detections_returns_none(
        self, mock_resend: MagicMock, sender: EmailSender
    ) -> None:
        result = sender.send_backfill_summary([])
        assert result is None
        mock_resend.Emails.send.assert_not_called()

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_multiple_detections_produce_one_email(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "summary_001"}
        detections = [
            (dummy_image, 0.85, "Motorcycle", datetime(2026, 2, 28, 10, 0, 0, tzinfo=timezone.utc)),
            (dummy_image, 0.72, "Car", datetime(2026, 2, 28, 11, 30, 0, tzinfo=timezone.utc)),
            (dummy_image, 0.65, "Truck", datetime(2026, 2, 28, 14, 15, 0, tzinfo=timezone.utc)),
        ]

        result = sender.send_backfill_summary(detections)

        assert result == "summary_001"
        mock_resend.Emails.send.assert_called_once()
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert "3 vehicles detected" in call_args["subject"]
        assert len(call_args["attachments"]) == 3
        assert call_args["attachments"][0]["filename"] == "detection_1.png"
        assert call_args["attachments"][1]["filename"] == "detection_2.png"
        assert call_args["attachments"][2]["filename"] == "detection_3.png"

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_summary_html_contains_detection_details(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "summary_002"}
        detections = [
            (dummy_image, 0.85, "Motorcycle", datetime(2026, 2, 28, 10, 0, 0, tzinfo=timezone.utc)),
            (dummy_image, 0.72, "Car", datetime(2026, 2, 28, 11, 30, 0, tzinfo=timezone.utc)),
        ]

        sender.send_backfill_summary(detections)

        call_args = mock_resend.Emails.send.call_args[0][0]
        html = call_args["html"]
        assert "Motorcycle" in html
        assert "Car" in html
        assert "85%" in html
        assert "72%" in html

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_summary_handles_exception(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.side_effect = Exception("API error")
        detections = [
            (dummy_image, 0.85, "Motorcycle", datetime(2026, 2, 28, 10, 0, 0, tzinfo=timezone.utc)),
        ]

        result = sender.send_backfill_summary(detections)
        assert result is None
