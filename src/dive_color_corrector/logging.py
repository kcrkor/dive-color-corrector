"""Logging configuration for dive color corrector."""

import logging
import sys


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("dive_color_corrector")
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the application logger."""
    return logging.getLogger("dive_color_corrector")
