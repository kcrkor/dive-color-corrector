import numpy as np

from dive_color_corrector.core.color.filter import (
    apply_filter,
    get_filter_matrix,
    normalizing_interval,
    precompute_filter_matrices,
)


def test_normalizing_interval() -> None:
    array = [10, 50, 100, 200]
    low, high = normalizing_interval(array)
    assert low == 100
    assert high == 200

    array = [0, 255]
    low, high = normalizing_interval(array)
    assert low == 0
    assert high == 255


def test_apply_filter(random_rgb_image: np.ndarray, sample_filter_matrix: np.ndarray) -> None:
    filtered = apply_filter(random_rgb_image, sample_filter_matrix)

    assert np.allclose(random_rgb_image, filtered, atol=1)
    assert filtered.dtype == np.uint8


def test_get_filter_matrix(random_rgb_image: np.ndarray) -> None:
    matrix = get_filter_matrix(random_rgb_image)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (20,)
    assert matrix.dtype == np.float64 or matrix.dtype == np.float32


def test_precompute_filter_matrices() -> None:
    frame_count = 10
    filter_indices = [0, 5, 9]

    m1 = np.eye(4, 5).flatten()
    filter_matrices = np.array([m1, m1, m1])

    interpolated = precompute_filter_matrices(frame_count, filter_indices, filter_matrices)

    assert interpolated.shape == (10, 20)
    for i in range(10):
        assert np.allclose(interpolated[i], m1)

    m2 = m1 * 2
    filter_matrices = np.array([m1, m2])
    filter_indices = [0, 9]

    interpolated = precompute_filter_matrices(10, filter_indices, filter_matrices)
    assert np.allclose(interpolated[5], 1.5555 * m1, atol=0.1)
