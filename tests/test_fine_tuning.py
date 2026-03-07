from pathlib import Path

import numpy as np
import pytest

from lakeside_sentinel.review.fine_tuning import (
    FINE_TUNING_CLASSES,
    ensure_data_yaml,
    save_annotation,
    save_other,
)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((200, 400, 3), dtype=np.uint8)


@pytest.fixture
def bbox() -> tuple[float, float, float, float]:
    return (100.0, 50.0, 300.0, 150.0)


class TestSaveAnnotation:
    def test_creates_image_and_label(
        self, tmp_path: Path, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> None:
        save_annotation(frame, bbox, "motorbike", "test_001", tmp_path)

        image_path = tmp_path / "images" / "train" / "test_001.jpg"
        label_path = tmp_path / "labels" / "train" / "test_001.txt"

        assert image_path.exists()
        assert label_path.exists()

        label_content = label_path.read_text().strip()
        parts = label_content.split()
        assert parts[0] == str(FINE_TUNING_CLASSES["motorbike"])
        assert len(parts) == 5

    def test_bbox_normalization(
        self, tmp_path: Path, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> None:
        save_annotation(frame, bbox, "bicycle", "test_002", tmp_path)

        label_path = tmp_path / "labels" / "train" / "test_002.txt"
        parts = label_path.read_text().strip().split()

        # bbox = (100, 50, 300, 150), frame = 400x200
        # x_center = (100+300)/2 / 400 = 0.5
        # y_center = (50+150)/2 / 200 = 0.5
        # width = (300-100) / 400 = 0.5
        # height = (150-50) / 200 = 0.5
        assert float(parts[1]) == pytest.approx(0.5, abs=1e-4)
        assert float(parts[2]) == pytest.approx(0.5, abs=1e-4)
        assert float(parts[3]) == pytest.approx(0.5, abs=1e-4)
        assert float(parts[4]) == pytest.approx(0.5, abs=1e-4)

    def test_appends_to_existing_label(self, tmp_path: Path, frame: np.ndarray) -> None:
        bbox1 = (10.0, 10.0, 50.0, 50.0)
        bbox2 = (100.0, 100.0, 200.0, 200.0)

        save_annotation(frame, bbox1, "person", "test_003", tmp_path)
        save_annotation(frame, bbox2, "dog", "test_003", tmp_path)

        label_path = tmp_path / "labels" / "train" / "test_003.txt"
        lines = label_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith(str(FINE_TUNING_CLASSES["person"]))
        assert lines[1].startswith(str(FINE_TUNING_CLASSES["dog"]))

    def test_unknown_class_produces_no_files(
        self, tmp_path: Path, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> None:
        save_annotation(frame, bbox, "unknown_class", "test_004", tmp_path)

        assert not (tmp_path / "images" / "train" / "test_004.jpg").exists()
        assert not (tmp_path / "labels" / "train" / "test_004.txt").exists()

    def test_all_classes_have_ids(self) -> None:
        expected = {"bicycle", "chair", "dog", "motorbike", "person", "scooter", "stroller"}
        assert set(FINE_TUNING_CLASSES.keys()) == expected
        # IDs should be 0-6
        assert sorted(FINE_TUNING_CLASSES.values()) == list(range(7))


class TestSaveOther:
    def test_creates_image_and_json(
        self, tmp_path: Path, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> None:
        save_other(frame, bbox, "test_other_001", tmp_path)

        image_path = tmp_path / "other" / "test_other_001.jpg"
        json_path = tmp_path / "other" / "test_other_001.json"

        assert image_path.exists()
        assert json_path.exists()

        import json

        metadata = json.loads(json_path.read_text())
        assert metadata["bbox"] == list(bbox)
        assert metadata["frame_width"] == 400
        assert metadata["frame_height"] == 200


class TestEnsureDataYaml:
    def test_creates_data_yaml(self, tmp_path: Path) -> None:
        yaml_path = ensure_data_yaml(tmp_path)
        assert yaml_path.exists()

        content = yaml_path.read_text()
        assert "bicycle" in content
        assert "motorbike" in content
        assert "stroller" in content

    def test_idempotent(self, tmp_path: Path) -> None:
        path1 = ensure_data_yaml(tmp_path)
        content1 = path1.read_text()

        path2 = ensure_data_yaml(tmp_path)
        content2 = path2.read_text()

        assert content1 == content2


class TestSkipClassification:
    def test_skip_produces_no_files(
        self, tmp_path: Path, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> None:
        # "skip" is not in FINE_TUNING_CLASSES, so save_annotation won't save
        save_annotation(frame, bbox, "", "test_skip", tmp_path)
        assert not (tmp_path / "images" / "train").exists()
        assert not (tmp_path / "labels" / "train").exists()
