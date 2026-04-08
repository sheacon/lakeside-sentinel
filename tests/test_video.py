from pathlib import Path

import cv2
import numpy as np

from lakeside_sentinel.utils.video import extract_frames


def _create_test_mp4(path: Path, num_frames: int = 30, fps: float = 30.0) -> Path:
    """Create a minimal MP4 file with solid-color frames for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))

    for i in range(num_frames):
        frame = np.full((240, 320, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path


class TestExtractFrames:
    def test_extracts_one_frame_per_second(self, tmp_path: Path) -> None:
        # 30 frames at 30fps = 1 second → expect 1 frame
        mp4 = _create_test_mp4(tmp_path / "clip.mp4", num_frames=30, fps=30.0)
        frames = extract_frames(mp4, fps_sample=1)
        assert len(frames) == 1

    def test_extracts_multiple_frames(self, tmp_path: Path) -> None:
        # 90 frames at 30fps = 3 seconds → expect 3 frames at 1fps
        mp4 = _create_test_mp4(tmp_path / "clip.mp4", num_frames=90, fps=30.0)
        frames = extract_frames(mp4, fps_sample=1)
        assert len(frames) == 3

    def test_frames_are_correct_shape(self, tmp_path: Path) -> None:
        mp4 = _create_test_mp4(tmp_path / "clip.mp4", num_frames=30, fps=30.0)
        frames = extract_frames(mp4, fps_sample=1)
        assert frames[0].shape == (240, 320, 3)

    def test_extracts_two_frames_per_second(self, tmp_path: Path) -> None:
        # 90 frames at 30fps = 3 seconds → expect 6 frames at 2fps
        mp4 = _create_test_mp4(tmp_path / "clip.mp4", num_frames=90, fps=30.0)
        frames = extract_frames(mp4, fps_sample=2)
        assert len(frames) == 6

    def test_invalid_file_returns_empty_list(self, tmp_path: Path) -> None:
        bad = tmp_path / "not_a_video.mp4"
        bad.write_bytes(b"not a video")
        frames = extract_frames(bad)
        assert frames == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        frames = extract_frames(tmp_path / "does_not_exist.mp4")
        assert frames == []
