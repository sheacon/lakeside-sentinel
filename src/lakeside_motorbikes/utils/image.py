import numpy as np


def crop_to_roi(
    frames: list[np.ndarray],
    y_start: float = 0.0,
    y_end: float = 1.0,
) -> list[np.ndarray]:
    """Crop frames to a vertical region of interest.

    Args:
        frames: List of BGR images as numpy arrays.
        y_start: Fraction (0.0–1.0) for the top of the ROI.
        y_end: Fraction (0.0–1.0) for the bottom of the ROI.

    Returns:
        List of cropped frames.
    """
    if y_start == 0.0 and y_end == 1.0:
        return frames

    cropped: list[np.ndarray] = []
    for frame in frames:
        h = frame.shape[0]
        y1 = max(0, int(h * y_start))
        y2 = min(h, int(h * y_end))
        cropped.append(frame[y1:y2, :].copy())
    return cropped


def crop_to_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: float = 0.2,
) -> np.ndarray:
    """Crop a frame to a bounding box with padding.

    Args:
        frame: BGR image as numpy array.
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        padding: Fraction of bbox dimensions to add as padding on each side.

    Returns:
        Cropped image as numpy array.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
