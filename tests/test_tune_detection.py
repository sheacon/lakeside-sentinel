"""Tests for the detection tuning harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tune_detection import (
    apply_threshold,
    build_hsp_configs,
    build_parser,
    build_vehicle_configs,
)

from lakeside_sentinel.detection.models import Detection

# --- Fixtures ---


def _make_detection(class_name: str, confidence: float) -> Detection:
    """Create a Detection with a dummy frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return Detection(
        frame=frame,
        bbox=(10.0, 20.0, 50.0, 60.0),
        confidence=confidence,
        class_name=class_name,
    )


# --- build_vehicle_configs tests ---


class TestBuildVehicleConfigs:
    def test_single_values(self) -> None:
        configs = build_vehicle_configs(
            models=["yolo26s.pt"],
            fps_values=[2],
            confidences=[0.4],
            roi_y_start=0.0,
            roi_y_end=1.0,
            roi_x_start=0.0,
            roi_x_end=1.0,
        )
        assert len(configs) == 1
        c = configs[0]
        assert c.run_id == 1
        assert c.mode == "vehicle"
        assert c.model_name == "yolo26s.pt"
        assert c.fps_sample == 2
        assert c.confidence_threshold == 0.4

    def test_cartesian_product(self) -> None:
        configs = build_vehicle_configs(
            models=["yolo26s.pt", "yolo26m.pt"],
            fps_values=[2, 4],
            confidences=[0.3, 0.5],
            roi_y_start=0.0,
            roi_y_end=0.28,
            roi_x_start=0.33,
            roi_x_end=1.0,
        )
        # 2 models × 2 fps × 2 confidences = 8
        assert len(configs) == 8
        assert configs[0].run_id == 1
        assert configs[-1].run_id == 8

    def test_roi_propagated(self) -> None:
        configs = build_vehicle_configs(
            models=["yolo26s.pt"],
            fps_values=[2],
            confidences=[0.4],
            roi_y_start=0.1,
            roi_y_end=0.5,
            roi_x_start=0.2,
            roi_x_end=0.8,
        )
        c = configs[0]
        assert c.roi_y_start == 0.1
        assert c.roi_y_end == 0.5
        assert c.roi_x_start == 0.2
        assert c.roi_x_end == 0.8

    def test_run_ids_sequential(self) -> None:
        configs = build_vehicle_configs(
            models=["a.pt", "b.pt", "c.pt"],
            fps_values=[1],
            confidences=[0.4],
            roi_y_start=0.0,
            roi_y_end=1.0,
            roi_x_start=0.0,
            roi_x_end=1.0,
        )
        assert [c.run_id for c in configs] == [1, 2, 3]


# --- build_hsp_configs tests ---


class TestBuildHSPConfigs:
    def test_single_values(self) -> None:
        configs = build_hsp_configs(
            models=["yolo26s.pt"],
            fps_values=[4],
            person_confidence_thresholds=[0.4],
            displacements=[60.0],
            max_match_distances=[200.0],
            roi_y_start=0.0,
            roi_y_end=1.0,
            roi_x_start=0.0,
            roi_x_end=1.0,
        )
        assert len(configs) == 1
        c = configs[0]
        assert c.mode == "hsp"
        assert c.hsp_displacement == 60.0
        assert c.hsp_person_confidence_threshold == 0.4
        assert c.hsp_max_match_distance == 200.0

    def test_cartesian_product(self) -> None:
        configs = build_hsp_configs(
            models=["yolo26s.pt"],
            fps_values=[4, 8],
            person_confidence_thresholds=[0.3, 0.4],
            displacements=[40.0, 60.0, 80.0],
            max_match_distances=[200.0],
            roi_y_start=0.0,
            roi_y_end=1.0,
            roi_x_start=0.0,
            roi_x_end=1.0,
        )
        # 1 model × 2 fps × 2 pconf × 3 disp × 1 maxd = 12
        assert len(configs) == 12

    def test_all_params_swept(self) -> None:
        configs = build_hsp_configs(
            models=["a.pt", "b.pt"],
            fps_values=[4, 8],
            person_confidence_thresholds=[0.3, 0.4],
            displacements=[40.0, 60.0],
            max_match_distances=[150.0, 200.0],
            roi_y_start=0.0,
            roi_y_end=1.0,
            roi_x_start=0.0,
            roi_x_end=1.0,
        )
        # 2 × 2 × 2 × 2 × 2 = 32
        assert len(configs) == 32


# --- apply_threshold tests ---


class TestApplyThreshold:
    def test_all_above_threshold(self) -> None:
        class_best = {
            "Bicycle": _make_detection("Bicycle", 0.8),
            "Motorcycle": _make_detection("Motorcycle", 0.6),
        }
        best, sub = apply_threshold(class_best, 0.4)
        assert best is not None
        assert best.class_name == "Bicycle"
        assert best.confidence == 0.8
        assert sub == {}

    def test_all_below_threshold(self) -> None:
        class_best = {
            "Bicycle": _make_detection("Bicycle", 0.2),
            "Motorcycle": _make_detection("Motorcycle", 0.1),
        }
        best, sub = apply_threshold(class_best, 0.4)
        assert best is None
        assert len(sub) == 2
        assert "Bicycle" in sub
        assert "Motorcycle" in sub

    def test_mixed_threshold(self) -> None:
        class_best = {
            "Bicycle": _make_detection("Bicycle", 0.5),
            "Motorcycle": _make_detection("Motorcycle", 0.2),
        }
        best, sub = apply_threshold(class_best, 0.4)
        assert best is not None
        assert best.class_name == "Bicycle"
        assert len(sub) == 1
        assert "Motorcycle" in sub

    def test_empty_input(self) -> None:
        best, sub = apply_threshold({}, 0.4)
        assert best is None
        assert sub == {}

    def test_exact_threshold(self) -> None:
        class_best = {
            "Bicycle": _make_detection("Bicycle", 0.4),
        }
        best, sub = apply_threshold(class_best, 0.4)
        assert best is not None
        assert best.class_name == "Bicycle"
        assert sub == {}


# --- Argparse tests ---


class TestArgparse:
    def test_vehicle_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4"])
        assert args.clip == Path("test.mp4")
        assert args.hsp is False
        assert args.model == ["yolo26s.pt"]
        assert args.fps is None  # defaults applied in main()
        assert args.confidence == [0.4]
        assert args.roi_y_start == 0.0
        assert args.roi_y_end == 1.0
        assert args.roi_x_start == 0.0
        assert args.roi_x_end == 1.0

    def test_hsp_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4", "--hsp"])
        assert args.hsp is True

    def test_multi_value_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4", "--model", "a.pt", "b.pt", "c.pt"])
        assert args.model == ["a.pt", "b.pt", "c.pt"]

    def test_multi_value_fps(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4", "--fps", "2", "4", "8"])
        assert args.fps == [2, 4, 8]

    def test_multi_value_confidence(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--clip", "test.mp4", "--confidence", "0.3", "0.4", "0.5"])
        assert args.confidence == [0.3, 0.4, 0.5]

    def test_hsp_params(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--clip",
                "test.mp4",
                "--hsp",
                "--hsp-displacement",
                "40.0",
                "60.0",
                "--hsp-person-confidence",
                "0.3",
                "0.4",
                "--hsp-max-match-distance",
                "150.0",
                "200.0",
            ]
        )
        assert args.hsp_displacement == [40.0, 60.0]
        assert args.hsp_person_confidence == [0.3, 0.4]
        assert args.hsp_max_match_distance == [150.0, 200.0]

    def test_roi_params(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--clip",
                "test.mp4",
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
        assert args.roi_y_start == 0.1
        assert args.roi_y_end == 0.5
        assert args.roi_x_start == 0.2
        assert args.roi_x_end == 0.8

    def test_clip_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
