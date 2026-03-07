import sys
from unittest.mock import patch

import pytest

from lakeside_sentinel.cli import parse_args


class TestCLIValidation:
    def test_no_flags_is_present_mode(self) -> None:
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert not args.debug
        assert not args.veh
        assert not args.hsp

    def test_debug_requires_veh_or_hsp(self) -> None:
        with patch.object(sys, "argv", ["prog", "--debug"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_debug_veh_is_valid(self) -> None:
        with patch.object(sys, "argv", ["prog", "--debug", "--veh"]):
            args = parse_args()
        assert args.debug
        assert args.veh

    def test_debug_hsp_is_valid(self) -> None:
        with patch.object(sys, "argv", ["prog", "--debug", "--hsp"]):
            args = parse_args()
        assert args.debug
        assert args.hsp

    def test_debug_veh_and_hsp_mutually_exclusive(self) -> None:
        with patch.object(sys, "argv", ["prog", "--debug", "--veh", "--hsp"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_veh_without_debug_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--veh"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_hsp_without_debug_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--hsp"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_claude_without_debug_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--claude"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_claude_keep_rejected_without_debug_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--claude-keep-rejected"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_debug_veh_with_claude(self) -> None:
        with patch.object(sys, "argv", ["prog", "--debug", "--veh", "--claude"]):
            args = parse_args()
        assert args.debug
        assert args.veh
        assert args.claude

    def test_review_flag(self) -> None:
        with patch.object(sys, "argv", ["prog", "--review"]):
            args = parse_args()
        assert args.review
        assert not args.debug

    def test_review_with_debug_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--review", "--debug", "--veh"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_date_without_debug(self) -> None:
        with patch.object(sys, "argv", ["prog", "--date", "2026-03-01"]):
            args = parse_args()
        assert args.date == "2026-03-01"
        assert not args.debug
