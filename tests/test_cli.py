import sys
from unittest.mock import patch

import pytest

from lakeside_sentinel.cli import parse_args


class TestCLIValidation:
    def test_no_flags_is_present_mode(self) -> None:
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert not args.verbose
        assert not args.veh
        assert not args.hsp

    def test_veh_standalone(self) -> None:
        with patch.object(sys, "argv", ["prog", "--veh"]):
            args = parse_args()
        assert args.veh
        assert not args.hsp

    def test_hsp_standalone(self) -> None:
        with patch.object(sys, "argv", ["prog", "--hsp"]):
            args = parse_args()
        assert args.hsp
        assert not args.veh

    def test_veh_and_hsp_mutually_exclusive(self) -> None:
        with patch.object(sys, "argv", ["prog", "--veh", "--hsp"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_claude_requires_veh_or_hsp(self) -> None:
        with patch.object(sys, "argv", ["prog", "--claude"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_claude_keep_rejected_requires_veh_or_hsp(self) -> None:
        with patch.object(sys, "argv", ["prog", "--claude-keep-rejected"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_veh_with_claude(self) -> None:
        with patch.object(sys, "argv", ["prog", "--veh", "--claude"]):
            args = parse_args()
        assert args.veh
        assert args.claude

    def test_review_flag(self) -> None:
        with patch.object(sys, "argv", ["prog", "--review"]):
            args = parse_args()
        assert args.review
        assert not args.verbose

    def test_review_with_veh_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--review", "--veh"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_review_with_hsp_is_error(self) -> None:
        with patch.object(sys, "argv", ["prog", "--review", "--hsp"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_date_without_verbose(self) -> None:
        with patch.object(sys, "argv", ["prog", "--date", "2026-03-01"]):
            args = parse_args()
        assert args.date == "2026-03-01"
        assert not args.verbose

    def test_verbose_standalone(self) -> None:
        with patch.object(sys, "argv", ["prog", "--verbose"]):
            args = parse_args()
        assert args.verbose

    def test_verbose_with_review(self) -> None:
        with patch.object(sys, "argv", ["prog", "--review", "--verbose"]):
            args = parse_args()
        assert args.review
        assert args.verbose

    def test_verbose_with_veh(self) -> None:
        with patch.object(sys, "argv", ["prog", "--verbose", "--veh"]):
            args = parse_args()
        assert args.verbose
        assert args.veh
