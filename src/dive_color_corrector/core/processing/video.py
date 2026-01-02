"""Video processing operations."""

import math
from collections.abc import Generator

import cv2
import numpy as np

from dive_color_corrector.core.color.constants import SAMPLE_SECONDS
from dive_color_corrector.core.color.filter import (
    apply_filter,
    get_filter_matrix,
    precompute_filter_matrices,
)
from dive_color_corrector.core.processing.image import correct
from dive_color_corrector.core.schemas import VideoData
from dive_color_corrector.core.utils.constants import VIDEO_CODEC
from dive_color_corrector.logging import get_logger

logger = get_logger()


def analyze_video(video_path: str, output_path: str) -> Generator[int | VideoData, None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample filter matrices at intervals
    filter_matrix_indices = []
    filter_matrices = []
    count = 0

    logger.info(f"Analyzing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break
            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break
            # Otherwise this is just a faulty frame read, try reading next frame
            continue

        count += 1
        if count % 100 == 0:
            logger.debug(f"Analyzed {count} frames")

        # Sample filter matrix every N seconds
        if count % (fps * SAMPLE_SECONDS) == 0:
            rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filter_matrix_indices.append(count)
            filter_matrices.append(get_filter_matrix(rgb_mat))

        yield count

    cap.release()

    # Convert to numpy arrays
    filter_matrices_arr = np.array(filter_matrices, dtype=np.float32)

    yield VideoData(
        input_video_path=video_path,
        output_video_path=output_path,
        fps=fps,
        frame_count=count,
        width=width,
        height=height,
        filter_indices=filter_matrix_indices,
        filter_matrices=filter_matrices_arr,
    )


def process_video(
    video_data: VideoData, yield_preview: bool = False, use_deep: bool = False
) -> Generator[tuple[float, bytes | None], None, None]:
    cap = cv2.VideoCapture(video_data.input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_data.input_video_path}")

    frame_width = video_data.width
    frame_height = video_data.height
    fps = video_data.fps
    frame_count = video_data.frame_count

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

    out = cv2.VideoWriter(video_data.output_video_path, fourcc, fps, (frame_width, frame_height))

    interpolated_matrices = None
    if not use_deep and len(video_data.filter_matrices) > 0:
        logger.info("Precomputing filter matrices...")
        interpolated_matrices = precompute_filter_matrices(
            frame_count,
            video_data.filter_indices,
            video_data.filter_matrices,
        )

    logger.info("Processing video...")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break
            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break
            # Otherwise this is just a faulty frame read, try reading next
            continue

        # Increment after successful read for correct indexing
        frame_idx = count
        count += 1
        percent = 100.0 * count / frame_count
        if count % 100 == 0:
            logger.debug(f"Processed {percent:.2f}%")

        # Convert to RGB and apply correction
        rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if use_deep:
            # Use deep learning model
            corrected_mat = correct(rgb_mat, use_deep=True)
        elif interpolated_matrices is not None:
            # Use precomputed filter matrix
            corrected_mat = apply_filter(rgb_mat, interpolated_matrices[frame_idx])
            corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
        else:
            # Fallback to per-frame correction
            corrected_mat = correct(rgb_mat, use_deep=False)

        # Write corrected frame
        out.write(corrected_mat)

        if yield_preview:
            preview = frame.copy()
            width = preview.shape[1] // 2
            height = preview.shape[0] // 2
            preview[:, width:] = corrected_mat[:, width:]

            preview = cv2.resize(preview, (width, height))

            yield percent, cv2.imencode(".png", preview)[1].tobytes()
        else:
            yield percent, None

    cap.release()
    out.release()
