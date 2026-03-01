import base64
import logging
from datetime import datetime

import cv2
import numpy as np
import resend

logger = logging.getLogger(__name__)


class EmailSender:
    """Sends motorbike detection alert emails via Resend."""

    def __init__(self, api_key: str, from_email: str, to_email: str) -> None:
        resend.api_key = api_key
        self._from_email = from_email
        self._to_email = to_email

    def send_alert(
        self,
        cropped_image: np.ndarray,
        confidence: float,
        event_time: datetime,
    ) -> str | None:
        """Send an alert email with the cropped motorbike image.

        Returns:
            The email ID if sent successfully, None otherwise.
        """
        _, png_bytes = cv2.imencode(".png", cropped_image)
        image_b64 = base64.b64encode(png_bytes.tobytes()).decode()

        formatted_time = event_time.strftime("%-d %b %Y at %H:%M:%S")
        subject = f"Motorbike Detected — {event_time.strftime('%-d %b %Y %H:%M')}"

        body = (
            f"<h2>Motorbike Detected</h2>"
            f"<p><strong>Time:</strong> {formatted_time}</p>"
            f"<p><strong>Confidence:</strong> {confidence:.0%}</p>"
            f"<p>See attached image.</p>"
        )

        try:
            result = resend.Emails.send(
                {
                    "from": self._from_email,
                    "to": [self._to_email],
                    "subject": subject,
                    "html": body,
                    "attachments": [
                        {
                            "filename": "motorbike.png",
                            "content": image_b64,
                            "content_type": "image/png",
                        }
                    ],
                }
            )
            email_id = result.get("id") if isinstance(result, dict) else getattr(result, "id", None)
            logger.info("Alert email sent: %s", email_id)
            return email_id
        except Exception:
            logger.exception("Failed to send alert email")
            return None
