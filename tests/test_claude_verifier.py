from unittest.mock import MagicMock, patch

import numpy as np

from lakeside_sentinel.detection.claude_verifier import ClaudeVerifier
from lakeside_sentinel.detection.models import Detection


def _make_detection(
    class_name: str = "Motorcycle",
    confidence: float = 0.85,
) -> Detection:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return Detection(
        frame=frame,
        bbox=(10.0, 10.0, 90.0, 90.0),
        confidence=confidence,
        class_name=class_name,
    )


def _mock_response(text: str) -> MagicMock:
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


class TestVerifyDetection:
    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_confirmed_when_claude_says_yes(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.return_value = _mock_response("yes")

        verifier = ClaudeVerifier(api_key="test-key")
        det = _make_detection()
        result = verifier.verify_detection(det)

        assert result == "confirmed"
        assert det.verification_status == "confirmed"

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_rejected_when_claude_says_no(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.return_value = _mock_response("no")

        verifier = ClaudeVerifier(api_key="test-key")
        det = _make_detection()
        result = verifier.verify_detection(det)

        assert result == "rejected"
        assert det.verification_status == "rejected"

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_confirmed_when_response_starts_with_yes(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.return_value = _mock_response("Yes, it is a motorcycle.")

        verifier = ClaudeVerifier(api_key="test-key")
        det = _make_detection()
        result = verifier.verify_detection(det)

        assert result == "confirmed"
        assert det.verification_status == "confirmed"

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_error_returns_error_and_status_none(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.side_effect = Exception("API timeout")

        verifier = ClaudeVerifier(api_key="test-key")
        det = _make_detection()
        result = verifier.verify_detection(det)

        assert result == "error"
        assert det.verification_status is None

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_sends_jpeg_image_to_api(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.return_value = _mock_response("yes")

        verifier = ClaudeVerifier(api_key="test-key", model="claude-sonnet-4-20250514")
        det = _make_detection()
        verifier.verify_detection(det)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 16

        content = call_kwargs["messages"][0]["content"]
        image_block = content[0]
        assert image_block["type"] == "image"
        assert image_block["source"]["media_type"] == "image/jpeg"
        assert image_block["source"]["type"] == "base64"
        # Verify the data is valid base64
        import base64

        decoded = base64.b64decode(image_block["source"]["data"])
        assert len(decoded) > 0


class TestVerifyDetections:
    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_filters_rejected_detections(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        # First call: yes (Motorcycle), second call: no (Bicycle)
        mock_client.messages.create.side_effect = [
            _mock_response("yes"),
            _mock_response("no"),
        ]

        verifier = ClaudeVerifier(api_key="test-key")
        detections = {
            "Motorcycle": _make_detection("Motorcycle", 0.9),
            "Bicycle": _make_detection("Bicycle", 0.7),
        }
        result = verifier.verify_detections(detections)

        assert "Motorcycle" in result
        assert "Bicycle" not in result

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_keeps_detection_on_error(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.side_effect = Exception("API error")

        verifier = ClaudeVerifier(api_key="test-key")
        detections = {"Motorcycle": _make_detection("Motorcycle", 0.9)}
        result = verifier.verify_detections(detections)

        # Fail-open: detection is kept
        assert "Motorcycle" in result
        assert result["Motorcycle"].verification_status is None

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_empty_dict_returns_empty(self, mock_anthropic_cls: MagicMock) -> None:
        verifier = ClaudeVerifier(api_key="test-key")
        result = verifier.verify_detections({})

        assert result == {}
        mock_anthropic_cls.return_value.messages.create.assert_not_called()

    @patch("lakeside_sentinel.detection.claude_verifier.anthropic.Anthropic")
    def test_encode_frame_returns_base64_string(self, mock_anthropic_cls: MagicMock) -> None:
        verifier = ClaudeVerifier(api_key="test-key")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = verifier._encode_frame(frame, (10.0, 10.0, 90.0, 90.0), 0.2)

        assert isinstance(result, str)
        # Should be valid base64
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0
