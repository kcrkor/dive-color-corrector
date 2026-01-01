"""Custom exceptions for dive color corrector."""


class DiveColorCorrectorError(Exception):
    """Base exception for dive color corrector."""


class VideoProcessingError(DiveColorCorrectorError):
    """Error during video processing."""


class ImageProcessingError(DiveColorCorrectorError):
    """Error during image processing."""


class ModelLoadError(DiveColorCorrectorError):
    """Error loading ML model."""


class InvalidInputError(DiveColorCorrectorError):
    """Invalid input file or parameters."""
