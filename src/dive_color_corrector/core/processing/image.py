"""Image processing operations."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from dive_color_corrector.core.correction import correct as correct_simple
from dive_color_corrector.core.models.sesr import SESR_AVAILABLE, DeepSESR, SESRNotAvailableError
from dive_color_corrector.core.utils.constants import PREVIEW_HEIGHT, PREVIEW_WIDTH

__all__ = ["SESR_AVAILABLE", "correct", "correct_image"]

JPEG_EXTENSIONS = {".jpg", ".jpeg"}
PNG_EXTENSIONS = {".png"}
DEFAULT_JPEG_QUALITY = 95


def correct(mat: np.ndarray, use_deep: bool = False) -> np.ndarray:
    if use_deep:
        if not SESR_AVAILABLE:
            raise SESRNotAvailableError()
        model = DeepSESR()
        return model.enhance(mat)
    return correct_simple(mat)


def _get_save_kwargs(image: Image.Image, output_path: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    ext = Path(output_path).suffix.lower()

    if exif := image.info.get("exif"):
        kwargs["exif"] = exif

    if icc := image.info.get("icc_profile"):
        kwargs["icc_profile"] = icc

    if ext in JPEG_EXTENSIONS:
        kwargs["quality"] = DEFAULT_JPEG_QUALITY
        kwargs["subsampling"] = "4:4:4"
    elif ext in PNG_EXTENSIONS:
        kwargs["compress_level"] = 6

    return kwargs


def correct_image(input_path: str, output_path: str | None, use_deep: bool = False) -> bytes:
    with Image.open(input_path) as image:
        original_info = image.info.copy()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.info = original_info
        mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    corrected_mat = correct(rgb_mat, use_deep=use_deep)

    if output_path:
        output_image = Image.fromarray(cv2.cvtColor(corrected_mat, cv2.COLOR_BGR2RGB))
        with Image.open(input_path) as original:
            save_kwargs = _get_save_kwargs(original, output_path)
        output_image.save(output_path, **save_kwargs)

    preview = mat.copy()
    width = preview.shape[1] // 2
    preview[:, width:] = corrected_mat[:, width:]

    preview = cv2.resize(preview, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

    _, encoded = cv2.imencode(".png", preview)
    return bytes(encoded.tobytes())
