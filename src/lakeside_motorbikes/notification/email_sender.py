import base64
import logging
from datetime import datetime

import cv2
import numpy as np
import resend

logger = logging.getLogger(__name__)


class EmailSender:
    """Sends vehicle detection alert emails via Resend."""

    def __init__(self, api_key: str, from_email: str, to_email: str) -> None:
        resend.api_key = api_key
        self._from_email = from_email
        self._to_email = to_email

    def send_alert(
        self,
        cropped_image: np.ndarray,
        confidence: float,
        event_time: datetime,
        class_name: str = "Vehicle",
    ) -> str | None:
        """Send an alert email with the cropped vehicle image.

        Returns:
            The email ID if sent successfully, None otherwise.
        """
        _, png_bytes = cv2.imencode(".png", cropped_image)
        image_b64 = base64.b64encode(png_bytes.tobytes()).decode()

        formatted_time = event_time.strftime("%-d %b %Y at %H:%M:%S")
        subject = f"{class_name} Detected — {event_time.strftime('%-d %b %Y %H:%M')}"

        body = (
            f"<h2>{class_name} Detected</h2>"
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
                            "filename": "detection.png",
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

    def send_backfill_summary(
        self,
        detections: list[tuple[np.ndarray, float, str, datetime]],
    ) -> str | None:
        """Send a single summary email with all backfill detections.

        Args:
            detections: List of (cropped_image, confidence, class_name, event_time) tuples.

        Returns:
            The email ID if sent successfully, None otherwise.
        """
        if not detections:
            return None

        count = len(detections)
        plural = "s" if count != 1 else ""
        subject = f"Backfill Summary — {count} vehicle{plural} detected"

        rows = []
        attachments = []
        for i, (image, confidence, class_name, event_time) in enumerate(detections, 1):
            _, png_bytes = cv2.imencode(".png", image)
            image_b64 = base64.b64encode(png_bytes.tobytes()).decode()

            filename = f"detection_{i}.png"
            attachments.append(
                {
                    "filename": filename,
                    "content": image_b64,
                    "content_type": "image/png",
                }
            )

            formatted_time = event_time.strftime("%-d %b %Y at %H:%M:%S")
            rows.append(
                f"<tr>"
                f"<td style='padding:8px;border:1px solid #ddd'>{i}</td>"
                f"<td style='padding:8px;border:1px solid #ddd'>{formatted_time}</td>"
                f"<td style='padding:8px;border:1px solid #ddd'>{class_name}</td>"
                f"<td style='padding:8px;border:1px solid #ddd'>{confidence:.0%}</td>"
                f"<td style='padding:8px;border:1px solid #ddd'>{filename}</td>"
                f"</tr>"
            )

        body = (
            f"<h2>Backfill Summary</h2>"
            f"<p>{count} vehicle{plural} detected in the last 24 hours.</p>"
            f"<table style='border-collapse:collapse;width:100%'>"
            f"<tr>"
            f"<th style='padding:8px;border:1px solid #ddd'>#</th>"
            f"<th style='padding:8px;border:1px solid #ddd'>Time</th>"
            f"<th style='padding:8px;border:1px solid #ddd'>Type</th>"
            f"<th style='padding:8px;border:1px solid #ddd'>Confidence</th>"
            f"<th style='padding:8px;border:1px solid #ddd'>Attachment</th>"
            f"</tr>"
            f"{''.join(rows)}"
            f"</table>"
        )

        try:
            result = resend.Emails.send(
                {
                    "from": self._from_email,
                    "to": [self._to_email],
                    "subject": subject,
                    "html": body,
                    "attachments": attachments,
                }
            )
            email_id = result.get("id") if isinstance(result, dict) else getattr(result, "id", None)
            logger.info("Backfill summary email sent: %s", email_id)
            return email_id
        except Exception:
            logger.exception("Failed to send backfill summary email")
            return None
