"""Custom exceptions for dive color corrector."""

from pathlib import Path


class DiveColorCorrectorError(Exception):
    """Base exception for dive color corrector."""


class VideoProcessingError(DiveColorCorrectorError):
    """Raised when video cannot be opened, read, or written."""

    def __init__(self, message: str, video_path: str | Path | None = None):
        self.video_path = video_path
        super().__init__(message)


class ImageProcessingError(DiveColorCorrectorError):
    """Raised when image cannot be opened, processed, or saved."""

    def __init__(self, message: str, image_path: str | Path | None = None):
        self.image_path = image_path
        super().__init__(message)


class ModelLoadError(DiveColorCorrectorError):
    """Raised when ONNX model file cannot be found or loaded."""

    def __init__(self, message: str, model_path: str | Path | None = None):
        self.model_path = model_path
        super().__init__(message)


class InvalidInputError(DiveColorCorrectorError):
    """Raised when input validation fails."""
