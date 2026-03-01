import logging
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(mp4_bytes: bytes, fps_sample: int = 1) -> list[np.ndarray]:
    """Extract frames from MP4 bytes at the given sample rate (frames per second).

    Args:
        mp4_bytes: Raw MP4 video bytes.
        fps_sample: How many frames to extract per second of video.

    Returns:
        List of BGR frames as numpy arrays.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(mp4_bytes)
        tmp.flush()
        return _read_frames(tmp.name, fps_sample)


def _read_frames(path: str, fps_sample: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", path)
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
