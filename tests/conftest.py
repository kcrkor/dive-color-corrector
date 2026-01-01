from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def underwater_image(fixtures_dir: Path) -> Path:
    return fixtures_dir / "underwater.jpg"


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
