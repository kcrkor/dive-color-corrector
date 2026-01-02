"""Color correction filter operations."""

import cv2
import numpy as np

from dive_color_corrector.core.color.constants import (
    BLUE_MAGIC_VALUE,
    MAX_HUE_SHIFT,
    MIN_AVG_RED,
    THRESHOLD_RATIO,
)
from dive_color_corrector.core.color.hue import hue_shift_red


def normalizing_interval(array: np.ndarray | list[float]) -> tuple[float, float]:
    """Find the largest interval between consecutive values.

    Args:
        array: Input array of values

    Returns:
        Tuple of (low, high) values defining the interval
    """
    high: float = 255.0
    low: float = 0.0
    max_dist: float = 0.0

    for i in range(1, len(array)):
        dist = array[i] - array[i - 1]
        if dist > max_dist:
            max_dist = dist
            high = array[i]
            low = array[i - 1]

    return (low, high)


def apply_filter(mat: np.ndarray, filt: np.ndarray | list[float]) -> np.ndarray:
    """Apply color correction filter to matrix.

    Args:
        mat: Input RGB matrix
        filt: Filter matrix (1-D array with at least 15 elements)

    Returns:
        Filtered RGB matrix as uint8
    """
    # Operate in-place on a float32 array to reduce temporaries
    filtered_mat = np.zeros_like(mat, dtype=np.float32)
    filtered_mat[..., 0] = (
        mat[..., 0] * filt[0] + mat[..., 1] * filt[1] + mat[..., 2] * filt[2] + filt[4] * 255
    )
    filtered_mat[..., 1] = mat[..., 1] * filt[6] + filt[9] * 255
    filtered_mat[..., 2] = mat[..., 2] * filt[12] + filt[14] * 255

    return np.clip(filtered_mat, 0, 255).astype(np.uint8)


def get_filter_matrix(mat: np.ndarray) -> np.ndarray:
    """Calculate color correction filter matrix.

    Args:
        mat: Input RGB matrix

    Returns:
        Filter matrix for color correction
    """
    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)

    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while new_avg_r < MIN_AVG_RED:
        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = cv2.calcHist([mat], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([mat], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([mat], [2], None, [256], [0, 256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0] * mat.shape[1]) / THRESHOLD_RATIO
    for x in range(256):
        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x
        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x
        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])

    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    red_offset = (-adjust_r_low / 256) * red_gain
    green_offset = (-adjust_g_low / 256) * green_gain
    blue_offset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array(
        [
            adjust_red,
            adjust_red_green,
            adjust_red_blue,
            0,
            red_offset,
            0,
            green_gain,
            0,
            0,
            green_offset,
            0,
            0,
            blue_gain,
            0,
            blue_offset,
            0,
            0,
            0,
            1,
            0,
        ]
    )


def precompute_filter_matrices(
    frame_count: int, filter_indices: np.ndarray | list[int], filter_matrices: np.ndarray
) -> np.ndarray:
    """Precompute interpolated filter matrices for all frames.

    Args:
        frame_count: Total number of frames in the video
        filter_indices: Array of frame indices where filters were sampled
        filter_matrices: Array of filter matrices at sampled indices,
                        shape (n_samples, filter_size)

    Returns:
        Interpolated filter matrices for all frames as float32,
        shape (frame_count, filter_size)
    """
    # Ensure inputs are numpy arrays with correct dtype
    filter_indices = np.asarray(filter_indices, dtype=np.float32)
    filter_matrices = np.asarray(filter_matrices, dtype=np.float32)

    # Sort by filter indices if not already sorted
    order = np.argsort(filter_indices)
    filter_indices = filter_indices[order]
    filter_matrices = filter_matrices[order]

    filter_matrix_size = filter_matrices.shape[1]
    frame_numbers = np.arange(frame_count, dtype=np.float32)

    # Allocate as float32 to reduce memory usage
    interpolated_matrices = np.zeros((frame_count, filter_matrix_size), dtype=np.float32)

    # Interpolate each coefficient across all frames
    for i in range(filter_matrix_size):
        interpolated_matrices[:, i] = np.interp(
            frame_numbers, filter_indices, filter_matrices[:, i]
        )

    return interpolated_matrices
