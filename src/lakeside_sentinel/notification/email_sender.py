import logging

import resend

logger = logging.getLogger(__name__)


class EmailSender:
    """Sends detection report emails via Resend."""

    def __init__(self, api_key: str, from_email: str, to_email: str) -> None:
        resend.api_key = api_key
        self._from_email = from_email
        self._to_email = to_email

    def send_report(
        self,
        html_body: str,
        subject: str,
        attachments: list[dict[str, str | bytes]] | None = None,
    ) -> str | None:
        """Send a pre-built HTML report via Resend.

        Args:
            html_body: Full HTML content.
            subject: Email subject line.
            attachments: Optional list of Resend attachment dicts (CID inline images).

        Returns:
            The email ID if sent successfully, None otherwise.
        """
        try:
            payload: dict[str, object] = {
                "from": self._from_email,
                "to": [self._to_email],
                "subject": subject,
                "html": html_body,
            }
            if attachments:
                payload["attachments"] = attachments
            result = resend.Emails.send(payload)
            email_id = result.get("id") if isinstance(result, dict) else getattr(result, "id", None)
            logger.info("Report email sent: %s", email_id)
            return email_id
        except Exception:
            logger.exception("Failed to send report email")
            return None
