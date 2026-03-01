import numpy as np

from lakeside_motorbikes.utils.image import crop_to_bbox, crop_to_roi


class TestCropToBbox:
    def test_basic_crop(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 255  # white square

        cropped = crop_to_bbox(frame, (100, 100, 200, 200), padding=0.0)
        assert cropped.shape == (100, 100, 3)

    def test_crop_with_padding(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cropped = crop_to_bbox(frame, (100, 100, 200, 200), padding=0.2)
        # 100px box + 20% padding = 20px each side → 140x140
        assert cropped.shape == (140, 140, 3)

    def test_crop_clamps_to_frame_bounds(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox near edges with large padding
        cropped = crop_to_bbox(frame, (0, 0, 90, 90), padding=0.5)
        assert cropped.shape[0] <= 100
        assert cropped.shape[1] <= 100

    def test_crop_returns_copy(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cropped = crop_to_bbox(frame, (50, 50, 100, 100), padding=0.0)
        cropped[:] = 255
        assert frame[75, 75, 0] == 0  # original unchanged


class TestCropToRoi:
    def test_crops_to_top_third(self) -> None:
        frames = [np.zeros((300, 640, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, y_start=0.0, y_end=1 / 3)
        assert result[0].shape == (100, 640, 3)

    def test_full_frame_with_defaults(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, y_start=0.0, y_end=1.0)
        assert result is frames  # returns same list, no copy

    def test_clamps_to_valid_bounds(self) -> None:
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, y_start=-0.5, y_end=1.5)
        assert result[0].shape[0] == 100
        assert result[0].shape[1] == 100

    def test_returns_copies(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = crop_to_roi([frame], y_start=0.0, y_end=0.5)
        result[0][:] = 255
        assert frame[0, 0, 0] == 0  # original unchanged

    def test_multiple_frames(self) -> None:
        frames = [np.zeros((400, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = crop_to_roi(frames, y_start=0.25, y_end=0.75)
        assert len(result) == 5
        for r in result:
            assert r.shape == (200, 640, 3)

    def test_horizontal_crop(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, x_start=0.25, x_end=0.75)
        assert result[0].shape == (480, 320, 3)

    def test_combined_x_and_y_crop(self) -> None:
        frames = [np.zeros((400, 800, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, y_start=0.0, y_end=0.5, x_start=0.25, x_end=0.75)
        assert result[0].shape == (200, 400, 3)

    def test_horizontal_only_does_not_short_circuit(self) -> None:
        frames = [np.zeros((300, 600, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, x_start=0.0, x_end=0.5)
        assert result is not frames
        assert result[0].shape == (300, 300, 3)

    def test_all_defaults_short_circuits(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = crop_to_roi(frames, y_start=0.0, y_end=1.0, x_start=0.0, x_end=1.0)
        assert result is frames
