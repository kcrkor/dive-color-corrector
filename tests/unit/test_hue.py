import numpy as np

from dive_color_corrector.core.color.hue import hue_shift_red


def test_hue_shift_red_identity() -> None:
    mat = np.ones((10, 10, 3), dtype=np.float32) * 100
    shifted = hue_shift_red(mat, 0)

    assert np.allclose(np.sum(shifted, axis=2), mat[..., 0])


def test_hue_shift_red_values() -> None:
    mat = np.array([[[100, 100, 100]]], dtype=np.float32)
    shifted = hue_shift_red(mat, 90)
    assert np.allclose(shifted[0, 0, 0], 46.7, atol=0.1)
    assert np.allclose(shifted[0, 0, 1], 91.7, atol=0.1)
    assert np.allclose(shifted[0, 0, 2], -38.3, atol=0.1)
