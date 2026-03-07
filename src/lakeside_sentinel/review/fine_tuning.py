"""YOLO-format annotation writer for fine-tuning dataset collection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FINE_TUNING_CLASSES = {
    "bicycle": 0,
    "chair": 1,
    "dog": 2,
    "motorbike": 3,
    "person": 4,
    "scooter": 5,
    "stroller": 6,
}

_DATA_YAML_CONTENT = """path: .
train: images/train
val: images/train

names:
  0: bicycle
  1: chair
  2: dog
  3: motorbike
  4: person
  5: scooter
  6: stroller
"""


def _normalize_bbox(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel bbox (x1, y1, x2, y2) to YOLO normalized format.

    Returns:
        (x_center, y_center, width, height) normalized to 0.0-1.0.
    """
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / frame_width
    y_center = ((y1 + y2) / 2) / frame_height
    width = (x2 - x1) / frame_width
    height = (y2 - y1) / frame_height
    return x_center, y_center, width, height


def save_annotation(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    class_label: str,
    image_id: str,
    output_dir: Path,
) -> None:
    """Save a YOLO-format annotation (image + label file).

    Args:
        frame: Full scene image (ROI-cropped frame).
        bbox: Bounding box in pixel coordinates (x1, y1, x2, y2).
        class_label: Class label (must be in FINE_TUNING_CLASSES).
        image_id: Unique identifier for this image.
        output_dir: Base output directory (e.g. output/fine-tuning).
    """
    if class_label not in FINE_TUNING_CLASSES:
        logger.warning("Unknown class label: %s", class_label)
        return

    class_id = FINE_TUNING_CLASSES[class_label]
    h, w = frame.shape[:2]

    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    image_path = images_dir / f"{image_id}.jpg"
    if not image_path.exists():
        cv2.imwrite(str(image_path), frame)

    # Write/append label
    x_center, y_center, bw, bh = _normalize_bbox(bbox, w, h)
    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"
    label_path = labels_dir / f"{image_id}.txt"
    with open(label_path, "a") as f:
        f.write(label_line)

    logger.info("Saved annotation: %s -> %s (class=%s)", image_id, class_label, class_id)


def save_other(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    image_id: str,
    output_dir: Path,
) -> None:
    """Save an 'other' classified image for later manual classification.

    Args:
        frame: Full scene image.
        bbox: Bounding box in pixel coordinates.
        image_id: Unique identifier.
        output_dir: Base output directory (e.g. output/fine-tuning).
    """
    other_dir = output_dir / "other"
    other_dir.mkdir(parents=True, exist_ok=True)

    image_path = other_dir / f"{image_id}.jpg"
    cv2.imwrite(str(image_path), frame)

    metadata = {
        "bbox": list(bbox),
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
    }
    json_path = other_dir / f"{image_id}.json"
    json_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved 'other' image: %s", image_id)


def ensure_data_yaml(output_dir: Path) -> Path:
    """Create data.yaml with class mapping if it doesn't exist.

    Args:
        output_dir: Base output directory (e.g. output/fine-tuning).

    Returns:
        Path to the data.yaml file.
    """
    yaml_path = output_dir / "data.yaml"
    if not yaml_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(_DATA_YAML_CONTENT)
        logger.info("Created data.yaml at %s", yaml_path)
    return yaml_path
