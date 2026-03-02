from unittest.mock import MagicMock, patch

import pytest

from lakeside_sentinel.notification.email_sender import EmailSender


@pytest.fixture
def sender() -> EmailSender:
    return EmailSender(
        api_key="re_test_key",
        from_email="alerts@xeroshot.org",
        to_email="user@example.com",
    )


class TestSendReport:
    @patch("lakeside_sentinel.notification.email_sender.resend")
    def test_send_report_calls_resend(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
    ) -> None:
        mock_resend.Emails.send.return_value = {"id": "email_123"}

        result = sender.send_report("<h1>Report</h1>", "Daily Detection Report")

        assert result == "email_123"
        mock_resend.Emails.send.assert_called_once()
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["to"] == ["user@example.com"]
        assert call_args["from"] == "alerts@xeroshot.org"
        assert call_args["subject"] == "Daily Detection Report"
        assert call_args["html"] == "<h1>Report</h1>"
        assert "attachments" not in call_args

    @patch("lakeside_sentinel.notification.email_sender.resend")
    def test_send_report_returns_id_from_object(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
    ) -> None:
        result_obj = MagicMock()
        result_obj.id = "email_obj_456"
        # Make it not a dict so getattr branch is used
        mock_resend.Emails.send.return_value = result_obj

        result = sender.send_report("<p>Test</p>", "Test Subject")

        assert result == "email_obj_456"

    @patch("lakeside_sentinel.notification.email_sender.resend")
    def test_send_report_handles_exception(
        self,
        mock_resend: MagicMock,
        sender: EmailSender,
    ) -> None:
        mock_resend.Emails.send.side_effect = Exception("API error")

        result = sender.send_report("<h1>Report</h1>", "Test Subject")

        assert result is None
