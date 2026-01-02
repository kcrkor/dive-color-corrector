from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def create_test_video(tmp_path: Path) -> Callable[[int, int, int, int], Path]:
    def _create(
        duration_frames: int = 60,
        fps: int = 30,
        width: int = 320,
        height: int = 240,
    ) -> Path:
        path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

        for i in range(duration_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 100 + (i % 50)
            frame[:, :, 1] = 80
            frame[:, :, 2] = 30
            writer.write(frame)

        writer.release()
        return path

    return _create


@pytest.fixture
def underwater_image(fixtures_dir: Path) -> Path:
    return fixtures_dir / "underwater.jpg"


@pytest.fixture
def gopro_image(fixtures_dir: Path) -> Path:
    return fixtures_dir / "gopro_underwater.jpg"


@pytest.fixture
def random_rgb_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_filter_matrix() -> np.ndarray:
    return np.array(
        [
            1.0,
            0.0,
            0.0,
            0,
            0.0,
            0,
            1.0,
            0,
            0,
            0.0,
            0,
            0,
            1.0,
            0,
            0.0,
            0,
            0,
            0,
            1,
            0,
        ],
        dtype=np.float32,
    )
