"""Claude Vision verification for YOLO detections."""

from __future__ import annotations

import base64
import logging
import re

import anthropic
import cv2
import numpy as np

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.utils.image import crop_to_bbox

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Is there a motorized two-wheeled vehicle in this image — such as a motorcycle, motorbike, "
    "scooter, moped, or e-bike? People may be riding or standing near it. "
    "Do NOT count baby strollers, prams, pushchairs, wagons, wheelchairs, "
    "shopping carts, or any non-motorized wheeled object. "
    'Answer "yes" or "no" only. '
    'Answer "yes" only if a motorized two-wheeled vehicle is clearly visible. '
    'Answer "no" for everything else.'
)


class ClaudeVerifier:
    """Verifies YOLO detections using Claude Vision."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        crop_padding: float = 0.2,
        timeout: float = 30.0,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._crop_padding = crop_padding
        self._prompt = prompt

    def _encode_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        padding: float,
    ) -> str:
        """Crop detection and return base64-encoded JPEG string."""
        cropped = crop_to_bbox(frame, bbox, padding=padding)
        _, jpg_bytes = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(jpg_bytes.tobytes()).decode()

    def verify_detection(self, detection: Detection) -> str:
        """Verify a single detection with Claude Vision.

        Returns:
            "confirmed", "rejected", or "error".
        """
        try:
            image_b64 = self._encode_frame(detection.frame, detection.bbox, self._crop_padding)
            response = self._client.messages.create(
                model=self._model,
                max_tokens=16,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": self._prompt,
                            },
                        ],
                    }
                ],
            )
            raw_text = response.content[0].text.strip()
            logger.info(
                "Claude response for %s (%.0f%%): %r",
                detection.class_name,
                detection.confidence * 100,
                raw_text,
            )
            detection.verification_response = raw_text
            if re.search(r"yes", raw_text, re.IGNORECASE):
                detection.verification_status = "confirmed"
                return "confirmed"
            else:
                detection.verification_status = "rejected"
                return "rejected"
        except Exception:
            logger.exception("Claude Vision verification failed")
            return "error"

    def verify_detections(self, detections: dict[str, Detection]) -> dict[str, Detection]:
        """Verify all detections, returning only confirmed ones (plus errors, fail-open).

        Args:
            detections: Mapping of class name to Detection.

        Returns:
            Filtered dict with only confirmed and error (unverified) detections.
        """
        if not detections:
            return {}

        result: dict[str, Detection] = {}
        for class_name, detection in detections.items():
            status = self.verify_detection(detection)
            if status in ("confirmed", "error"):
                result[class_name] = detection
            else:
                logger.info(
                    "Claude rejected %s (confidence=%.0f%%)",
                    class_name,
                    detection.confidence * 100,
                )
        return result
