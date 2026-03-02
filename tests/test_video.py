import tempfile
from pathlib import Path

import cv2
import numpy as np

from lakeside_sentinel.utils.video import extract_frames


def _create_test_mp4(num_frames: int = 30, fps: float = 30.0) -> bytes:
    """Create a minimal MP4 file with solid-color frames for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, fps, (320, 240))

        for i in range(num_frames):
            frame = np.full((240, 320, 3), fill_value=i % 256, dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return Path(tmp.name).read_bytes()


class TestExtractFrames:
    def test_extracts_one_frame_per_second(self) -> None:
        # 30 frames at 30fps = 1 second → expect 1 frame
        mp4 = _create_test_mp4(num_frames=30, fps=30.0)
        frames = extract_frames(mp4, fps_sample=1)
        assert len(frames) == 1

    def test_extracts_multiple_frames(self) -> None:
        # 90 frames at 30fps = 3 seconds → expect 3 frames at 1fps
        mp4 = _create_test_mp4(num_frames=90, fps=30.0)
        frames = extract_frames(mp4, fps_sample=1)
        assert len(frames) == 3

    def test_frames_are_correct_shape(self) -> None:
        mp4 = _create_test_mp4(num_frames=30, fps=30.0)
        frames = extract_frames(mp4, fps_sample=1)
        assert frames[0].shape == (240, 320, 3)

    def test_extracts_two_frames_per_second(self) -> None:
        # 90 frames at 30fps = 3 seconds → expect 6 frames at 2fps
        mp4 = _create_test_mp4(num_frames=90, fps=30.0)
        frames = extract_frames(mp4, fps_sample=2)
        assert len(frames) == 6

    def test_empty_bytes_returns_empty_list(self) -> None:
        frames = extract_frames(b"not a video")
        assert frames == []
