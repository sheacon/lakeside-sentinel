"""Tests for the HSP track visualization script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from visualize_tracks import (
    TrackSummary,
    annotate_frame_progressive,
    build_parser,
    print_track_table,
    summarize_tracks,
)

from lakeside_sentinel.detection.hsp_detector import PersonTrack, TrackPoint

# --- Helpers ---


def _make_frame(h: int = 100, w: int = 200) -> np.ndarray:
    """Create a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_track_point(
    frame_index: int = 0,
    cx: float = 50.0,
    cy: float = 50.0,
    confidence: float = 0.8,
) -> TrackPoint:
    """Create a TrackPoint with a dummy frame."""
    return TrackPoint(
        frame_index=frame_index,
        centroid_x=cx,
        centroid_y=cy,
        bbox=(cx - 10, cy - 20, cx + 10, cy + 20),
        confidence=confidence,
        frame=_make_frame(),
    )


def _make_track(points: list[TrackPoint]) -> PersonTrack:
    """Create a PersonTrack from a list of TrackPoints."""
    return PersonTrack(points=points)


# --- TestBuildParser ---


class TestBuildParser:
    def test_single_clip(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4"])
        assert args.clip == [Path("test.mp4")]

    def test_multiple_clips(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "a.mp4", "b.mp4", "c.mp4"])
        assert args.clip == [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]

    def test_clip_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--clip",
                "test.mp4",
                "--model",
                "yolo26m.pt",
                "--fps",
                "8",
                "--displacement",
                "320.0",
                "--person-confidence",
                "0.3",
                "--max-match-distance",
                "600.0",
                "--roi-y-start",
                "0.1",
                "--roi-y-end",
                "0.5",
                "--roi-x-start",
                "0.2",
                "--roi-x-end",
                "0.8",
            ]
        )
        assert args.model == "yolo26m.pt"
        assert args.fps == 8
        assert args.displacement == 320.0
        assert args.person_confidence == 0.3
        assert args.max_match_distance == 600.0
        assert args.roi_y_start == 0.1
        assert args.roi_y_end == 0.5
        assert args.roi_x_start == 0.2
        assert args.roi_x_end == 0.8

    def test_defaults_are_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4"])
        assert args.model is None
        assert args.fps is None
        assert args.displacement is None
        assert args.person_confidence is None
        assert args.max_match_distance is None
        assert args.roi_y_start is None
        assert args.roi_y_end is None
        assert args.roi_x_start is None
        assert args.roi_x_end is None


# --- TestSummarizeTracks ---


class TestSummarizeTracks:
    def test_empty_tracks(self) -> None:
        result = summarize_tracks([], fps=4, threshold=240.0)
        assert result == []

    def test_fast_track(self) -> None:
        # Two points far apart → high displacement
        track = _make_track(
            [
                _make_track_point(frame_index=0, cx=0.0, cy=0.0, confidence=0.9),
                _make_track_point(frame_index=1, cx=100.0, cy=0.0, confidence=0.85),
            ]
        )
        result = summarize_tracks([track], fps=4, threshold=200.0)
        assert len(result) == 1
        s = result[0]
        assert s.track_id == 1
        assert s.num_points == 2
        assert s.displacement_per_sec == pytest.approx(400.0)  # 100px * 4fps
        assert s.above_threshold is True
        assert s.best_confidence == 0.9

    def test_slow_track(self) -> None:
        # Two points close together → low displacement
        track = _make_track(
            [
                _make_track_point(frame_index=0, cx=50.0, cy=50.0, confidence=0.7),
                _make_track_point(frame_index=1, cx=52.0, cy=50.0, confidence=0.6),
            ]
        )
        result = summarize_tracks([track], fps=4, threshold=240.0)
        assert len(result) == 1
        s = result[0]
        assert s.above_threshold is False
        assert s.displacement_per_sec == pytest.approx(8.0)  # 2px * 4fps

    def test_single_point_track(self) -> None:
        track = _make_track([_make_track_point(frame_index=0, confidence=0.5)])
        result = summarize_tracks([track], fps=4, threshold=240.0)
        assert len(result) == 1
        s = result[0]
        assert s.num_points == 1
        assert s.displacement_per_sec == 0.0
        assert s.above_threshold is False

    def test_sequential_ids(self) -> None:
        tracks = [
            _make_track([_make_track_point(frame_index=0)]),
            _make_track([_make_track_point(frame_index=1)]),
            _make_track([_make_track_point(frame_index=2)]),
        ]
        result = summarize_tracks(tracks, fps=4, threshold=240.0)
        assert [s.track_id for s in result] == [1, 2, 3]


# --- TestAnnotateFrameProgressive ---


class TestAnnotateFrameProgressive:
    def test_no_tracks_returns_copy(self) -> None:
        frame = _make_frame()
        result = annotate_frame_progressive(
            frame, [], current_frame_index=0, threshold=240.0, fps=4
        )
        np.testing.assert_array_equal(result, frame)
        assert result is not frame  # must be a copy

    def test_future_track_not_drawn(self) -> None:
        # Track starts at frame 5, but we're at frame 0
        track = _make_track(
            [
                _make_track_point(frame_index=5, cx=50.0, cy=50.0),
                _make_track_point(frame_index=6, cx=60.0, cy=50.0),
            ]
        )
        frame = _make_frame()
        result = annotate_frame_progressive(
            frame, [track], current_frame_index=0, threshold=240.0, fps=4
        )
        np.testing.assert_array_equal(result, frame)

    def test_active_track_modifies_frame(self) -> None:
        track = _make_track(
            [
                _make_track_point(frame_index=0, cx=50.0, cy=50.0),
                _make_track_point(frame_index=1, cx=60.0, cy=50.0),
            ]
        )
        frame = _make_frame()
        result = annotate_frame_progressive(
            frame, [track], current_frame_index=1, threshold=240.0, fps=4
        )
        # Frame should be modified (not equal to blank)
        assert not np.array_equal(result, frame)

    def test_progressive_reveal(self) -> None:
        # Track with 3 points — more annotation at frame 2 than frame 0
        track = _make_track(
            [
                _make_track_point(frame_index=0, cx=30.0, cy=50.0),
                _make_track_point(frame_index=1, cx=60.0, cy=50.0),
                _make_track_point(frame_index=2, cx=90.0, cy=50.0),
            ]
        )
        frame = _make_frame(h=100, w=200)

        result_0 = annotate_frame_progressive(
            frame, [track], current_frame_index=0, threshold=240.0, fps=4
        )
        result_2 = annotate_frame_progressive(
            frame, [track], current_frame_index=2, threshold=240.0, fps=4
        )

        # Frame 2 should have more non-zero pixels than frame 0
        # (frame 0 has only bbox, frame 2 has bbox + 2 line segments)
        pixels_0 = np.count_nonzero(result_0)
        pixels_2 = np.count_nonzero(result_2)
        assert pixels_2 > pixels_0


# --- TestPrintTrackTable ---


class TestPrintTrackTable:
    def test_header_and_rows_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        summaries = [
            TrackSummary(
                track_id=1,
                num_points=5,
                displacement_per_sec=300.0,
                above_threshold=True,
                best_confidence=0.92,
            ),
            TrackSummary(
                track_id=2,
                num_points=2,
                displacement_per_sec=50.0,
                above_threshold=False,
                best_confidence=0.45,
            ),
        ]
        print_track_table(summaries)
        output = capsys.readouterr().out
        assert "ID" in output
        assert "Points" in output
        assert "Disp (px/s)" in output
        assert "Fast" in output
        assert "Best Conf" in output
        assert "300.0" in output
        assert "50.0" in output
        assert "YES" in output
        assert "92%" in output

    def test_empty_summaries(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_track_table([])
        output = capsys.readouterr().out
        # Should still print header
        assert "ID" in output
        # No data rows — just header + separator + blank line
        lines = [line for line in output.strip().split("\n") if line.strip()]
        assert len(lines) == 2  # header + separator
