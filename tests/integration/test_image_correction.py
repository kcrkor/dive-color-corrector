import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from dive_color_corrector.core.models import SESR_AVAILABLE
from dive_color_corrector.core.processing.image import correct_image


class TestImageCorrection:
    def test_gopro_image_simple_correction(self, gopro_image: Path) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        correct_image(str(gopro_image), output_path, use_deep=False)

        assert Path(output_path).exists()
        corrected = cv2.imread(output_path)
        original = cv2.imread(str(gopro_image))

        assert corrected.shape == original.shape
        assert not np.array_equal(corrected, original)

        avg_red_original = np.mean(original[:, :, 2])
        avg_red_corrected = np.mean(corrected[:, :, 2])
        assert avg_red_corrected > avg_red_original

        Path(output_path).unlink()

    @pytest.mark.skipif(not SESR_AVAILABLE, reason="SESR not available")
    def test_gopro_image_sesr_correction(self, gopro_image: Path) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        correct_image(str(gopro_image), output_path, use_deep=True)

        assert Path(output_path).exists()
        corrected = cv2.imread(output_path)
        original = cv2.imread(str(gopro_image))

        assert corrected.shape == original.shape
        assert not np.array_equal(corrected, original)

        Path(output_path).unlink()

    def test_correction_returns_preview_bytes(self, gopro_image: Path) -> None:
        preview = correct_image(str(gopro_image), None, use_deep=False)

        assert isinstance(preview, bytes)
        assert len(preview) > 0

        preview_arr = np.frombuffer(preview, dtype=np.uint8)
        preview_img = cv2.imdecode(preview_arr, cv2.IMREAD_COLOR)
        assert preview_img is not None

    def test_metadata_preserved(self, gopro_image: Path) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        correct_image(str(gopro_image), output_path, use_deep=False)

        with Image.open(str(gopro_image)) as original:
            original_exif = original.info.get("exif")

        with Image.open(output_path) as corrected:
            corrected_exif = corrected.info.get("exif")

        if original_exif:
            assert corrected_exif is not None
            assert len(corrected_exif) > 0

        Path(output_path).unlink()
