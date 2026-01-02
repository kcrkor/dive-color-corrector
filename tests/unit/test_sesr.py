from pathlib import Path

import numpy as np
import pytest

from dive_color_corrector.core.models.sesr import DeepSESR


def test_model_initialization() -> None:
    model = DeepSESR()
    assert model.input_size == (240, 320)
    assert model.model is not None

    custom_path = Path("src/dive_color_corrector/models/deep_sesr_2x_1d.keras")
    model = DeepSESR(custom_path)
    assert model.input_size == (240, 320)
    assert model.model is not None

    with pytest.raises(FileNotFoundError):
        DeepSESR("non_existent_model.keras")


def test_preprocess_image() -> None:
    model = DeepSESR()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = model.preprocess_image(img)

    assert processed.shape == (1, 240, 320, 3)
    assert processed.dtype == np.float32
    assert np.all(processed >= -1) and np.all(processed <= 1)


def test_postprocess_image() -> None:
    model = DeepSESR()
    img = np.random.uniform(-1, 1, (1, 240, 320, 3)).astype(np.float32)
    processed = model.postprocess_image(img)

    assert processed.shape == (240, 320, 3)
    assert processed.dtype == np.uint8
    assert np.all(processed >= 0) and np.all(processed <= 255)


@pytest.mark.slow
def test_enhance() -> None:
    model = DeepSESR()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    enhanced = model.enhance(img)

    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8
    assert np.all(enhanced >= 0) and np.all(enhanced <= 255)
