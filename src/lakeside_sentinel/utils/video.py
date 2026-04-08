import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(mp4_path: Path, fps_sample: int = 1) -> list[np.ndarray]:
    """Extract frames from an MP4 file at the given sample rate (frames per second).

    Args:
        mp4_path: Path to the MP4 file on disk.
        fps_sample: How many frames to extract per second of video.

    Returns:
        List of BGR frames as numpy arrays.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", mp4_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    frame_interval = max(1, int(video_fps / fps_sample))
    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    logger.debug(
        "Extracted %d frames from video (%d total, interval=%d)",
        len(frames),
        frame_idx,
        frame_interval,
    )
    return frames
