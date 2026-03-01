from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single motorcycle detection result."""

    frame: np.ndarray
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
