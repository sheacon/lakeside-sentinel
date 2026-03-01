import numpy as np

from lakeside_motorbikes.utils.image import crop_to_bbox


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
