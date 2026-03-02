from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single detection result (shared by VEH and HSP modes)."""

    frame: np.ndarray
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    verification_status: str | None = field(default=None)
