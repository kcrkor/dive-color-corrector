"""Deep SESR model for underwater image enhancement.

Based on: https://github.com/IRVLab/Deep-SESR
Paper: https://arxiv.org/pdf/2002.01155.pdf

Model outputs 3 tensors:
- output[0]: Enhanced image at input resolution (320x240)
- output[1]: Super-resolved image at 2x resolution (640x480)
- output[2]: Saliency map
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import tensorflow as tf

LR_SIZE = (240, 320)
HR_SIZE = (480, 640)


class DeepSESR:
    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            package_root = Path(__file__).parent.parent.parent
            model_path = package_root / "models" / "deep_sesr_2x_1d.keras"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.input_size = LR_SIZE
        self.output_size = HR_SIZE

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        img = (img.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(img, axis=0)

    def postprocess_image(self, img: np.ndarray) -> np.ndarray:
        img = np.squeeze(img, axis=0)
        img = (img + 1.0) * 0.5
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def enhance(self, img: np.ndarray, use_superres: bool = True) -> np.ndarray:
        original_size = img.shape[:2]
        x = self.preprocess_image(img)
        y = self.model.predict(x, verbose=0)

        enhanced = (y[1] if use_superres else y[0]) if isinstance(y, list) and len(y) >= 2 else y

        enhanced = self.postprocess_image(cast(np.ndarray, enhanced))

        target_size = self.output_size if use_superres else self.input_size
        if original_size != target_size:
            enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))

        return cast(np.ndarray, enhanced)
