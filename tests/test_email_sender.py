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

        result = sender.send_alert(dummy_image, 0.85, event_time)

        assert result == "email_123"
        mock_resend.Emails.send.assert_called_once()
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["to"] == ["user@example.com"]
        assert call_args["from"] == "alerts@xeroshot.org"
        assert "Motorbike Detected" in call_args["subject"]
        assert len(call_args["attachments"]) == 1
        assert call_args["attachments"][0]["filename"] == "motorbike.png"

    @patch("lakeside_motorbikes.notification.email_sender.resend")
    def test_send_alert_includes_confidence_in_body(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
        dummy_image: np.ndarray,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "email_456"}
        event_time = datetime(2026, 2, 28, 14, 32, 5, tzinfo=timezone.utc)

        sender.send_alert(dummy_image, 0.85, event_time)

        call_args = mock_resend.Emails.send.call_args[0][0]
        assert "85%" in call_args["html"]

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
