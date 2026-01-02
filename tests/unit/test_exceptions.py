from pathlib import Path

from dive_color_corrector.core.exceptions import (
    DiveColorCorrectorError,
    ImageProcessingError,
    InvalidInputError,
    ModelLoadError,
    VideoProcessingError,
)


def test_dive_color_corrector_error() -> None:
    err = DiveColorCorrectorError("test message")
    assert str(err) == "test message"
    assert isinstance(err, Exception)


def test_video_processing_error() -> None:
    path = "test.mp4"
    err = VideoProcessingError("video error", video_path=path)
    assert str(err) == "video error"
    assert err.video_path == path
    assert isinstance(err, DiveColorCorrectorError)


def test_image_processing_error() -> None:
    path = Path("test.jpg")
    err = ImageProcessingError("image error", image_path=path)
    assert str(err) == "image error"
    assert err.image_path == path
    assert isinstance(err, DiveColorCorrectorError)


def test_model_load_error() -> None:
    path = "model.onnx"
    err = ModelLoadError("model error", model_path=path)
    assert str(err) == "model error"
    assert err.model_path == path
    assert isinstance(err, DiveColorCorrectorError)


def test_invalid_input_error() -> None:
    err = InvalidInputError("invalid input")
    assert str(err) == "invalid input"
    assert isinstance(err, DiveColorCorrectorError)
