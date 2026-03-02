"""Tests for the verification diagnostic script."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from test_verification import build_parser, encode_crop, run_verification


class TestBuildParser:
    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4"])
        assert args.clip == Path("test.mp4")
        assert args.runs == 3
        assert args.temperature == 0.0
        assert args.model == "claude-sonnet-4-20250514"
        assert args.yolo_model == "yolo26s.pt"
        assert args.fps == 2
        assert args.confidence == 0.4
        assert args.crop_padding == 0.2
        assert args.roi_y_start == 0.0
        assert args.roi_y_end == 1.0
        assert args.roi_x_start == 0.0
        assert args.roi_x_end == 1.0
        assert args.prompt is None
        assert args.save_crops is False

    def test_clip_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_optional_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--clip",
                "test.mp4",
                "--runs",
                "5",
                "--temperature",
                "1.0",
                "--model",
                "claude-opus-4-20250514",
                "--yolo-model",
                "yolo26m.pt",
                "--fps",
                "4",
                "--confidence",
                "0.3",
                "--crop-padding",
                "0.3",
                "--roi-y-start",
                "0.1",
                "--roi-y-end",
                "0.5",
                "--roi-x-start",
                "0.2",
                "--roi-x-end",
                "0.8",
                "--prompt",
                "Is this a vehicle?",
                "--save-crops",
            ]
        )
        assert args.runs == 5
        assert args.temperature == 1.0
        assert args.model == "claude-opus-4-20250514"
        assert args.yolo_model == "yolo26m.pt"
        assert args.fps == 4
        assert args.confidence == 0.3
        assert args.crop_padding == 0.3
        assert args.roi_y_start == 0.1
        assert args.roi_y_end == 0.5
        assert args.roi_x_start == 0.2
        assert args.roi_x_end == 0.8
        assert args.prompt == "Is this a vehicle?"
        assert args.save_crops is True


class TestEncodeCrop:
    def test_returns_valid_base64(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_crop(frame, (10.0, 10.0, 90.0, 90.0), padding=0.2)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_different_frames_produce_different_output(self) -> None:
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bbox = (10.0, 10.0, 90.0, 90.0)
        result_black = encode_crop(black, bbox)
        result_white = encode_crop(white, bbox)
        assert result_black != result_white


class TestRunVerification:
    def test_confirmed_verdict(self) -> None:
        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "yes"
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create.return_value = mock_response

        verdict, raw = run_verification(
            mock_client, "fake_b64", "claude-sonnet-4-20250514", 0.0, "Is this a vehicle?"
        )
        assert verdict == "confirmed"
        assert raw == "yes"

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_rejected_verdict(self) -> None:
        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "No"
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create.return_value = mock_response

        verdict, raw = run_verification(
            mock_client, "fake_b64", "claude-sonnet-4-20250514", 0.0, "prompt"
        )
        assert verdict == "rejected"
        assert raw == "no"

    def test_temperature_passed_through(self) -> None:
        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "yes"
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create.return_value = mock_response

        run_verification(mock_client, "b64", "model", 1.0, "prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0
