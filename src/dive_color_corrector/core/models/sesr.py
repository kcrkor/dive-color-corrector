"""Deep SESR model for underwater image enhancement."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    import onnxruntime

try:
    import onnxruntime as _ort

    SESR_AVAILABLE = True
except ImportError:
    _ort = None
    SESR_AVAILABLE = False

LR_SIZE = (240, 320)
HR_SIZE = (480, 640)


class SESRNotAvailableError(Exception):
    """Raised when SESR is requested but onnxruntime is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Deep SESR requires onnxruntime. Install with: pip install dive_color_corrector[sesr]"
        )


class DeepSESR:
    """Deep SESR model for underwater image enhancement using ONNX Runtime."""

    def __init__(self, model_path: str | Path | None = None):
        if not SESR_AVAILABLE:
            raise SESRNotAvailableError()

        if model_path is None:
            package_root = Path(__file__).parent.parent.parent
            model_path = package_root / "models" / "deep_sesr_2x.onnx"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.session: onnxruntime.InferenceSession = _ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = LR_SIZE
        self.output_size = HR_SIZE

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        img = (img.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(img, axis=0)

    def postprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Postprocess model output to BGR image."""
        img = np.squeeze(img, axis=0)
        img = (img + 1.0) * 0.5
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def enhance(self, img: np.ndarray, use_superres: bool = True) -> np.ndarray:
        """Enhance an underwater image using Deep SESR.

        Args:
            img: Input BGR image
            use_superres: Whether to use 2x super-resolution output

        Returns:
            Enhanced BGR image
        """
        original_size = img.shape[:2]
        x = self.preprocess_image(img)
        outputs = self.session.run(None, {self.input_name: x})

        enhanced = outputs[1] if use_superres and len(outputs) >= 2 else outputs[0]

        enhanced = self.postprocess_image(cast(np.ndarray, enhanced))

        target_size = self.output_size if use_superres else self.input_size
        if original_size != target_size:
            enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))

        return cast(np.ndarray, enhanced)
