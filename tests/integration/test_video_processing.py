from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from dive_color_corrector.core.exceptions import VideoProcessingError
from dive_color_corrector.core.processing.video import analyze_video, process_video
from dive_color_corrector.core.schemas import VideoData


class TestAnalyzeVideo:
    def test_analyze_video_yields_frame_counts(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        video_path = create_test_video(duration_frames=30, fps=30)
        output_path = tmp_path / "output.mp4"

        results = list(analyze_video(str(video_path), str(output_path)))

        frame_counts = [r for r in results if isinstance(r, int)]
        assert len(frame_counts) >= 1

    def test_analyze_video_yields_video_data(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        video_path = create_test_video(duration_frames=30, fps=30)
        output_path = tmp_path / "output.mp4"

        results = list(analyze_video(str(video_path), str(output_path)))

        video_data = results[-1]
        assert isinstance(video_data, VideoData)
        assert video_data.input_video_path == str(video_path)
        assert video_data.output_video_path == str(output_path)
        assert video_data.width == 320
        assert video_data.height == 240
        assert video_data.fps > 0
        assert video_data.frame_count > 0

    def test_analyze_video_invalid_path_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(VideoProcessingError):
            list(analyze_video("nonexistent.mp4", str(tmp_path / "out.mp4")))


class TestProcessVideo:
    @pytest.mark.slow
    def test_process_video_creates_output(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        video_path = create_test_video(duration_frames=30, fps=30)
        output_path = tmp_path / "output.mp4"

        video_data = None
        for item in analyze_video(str(video_path), str(output_path)):
            if isinstance(item, VideoData):
                video_data = item

        assert video_data is not None

        for percent, _ in process_video(video_data):
            assert 0 <= percent <= 100

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    def test_process_video_yields_progress(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        video_path = create_test_video(duration_frames=30, fps=30)
        output_path = tmp_path / "output.mp4"

        video_data = None
        for item in analyze_video(str(video_path), str(output_path)):
            if isinstance(item, VideoData):
                video_data = item

        assert video_data is not None

        progress_values = []
        for percent, preview in process_video(video_data, yield_preview=False):
            progress_values.append(percent)
            assert preview is None

        assert len(progress_values) > 0
        assert progress_values[-1] == pytest.approx(100.0, rel=0.1)

    @pytest.mark.slow
    def test_process_video_with_preview(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        video_path = create_test_video(duration_frames=10, fps=30)
        output_path = tmp_path / "output.mp4"

        video_data = None
        for item in analyze_video(str(video_path), str(output_path)):
            if isinstance(item, VideoData):
                video_data = item

        assert video_data is not None

        previews = []
        for _, preview in process_video(video_data, yield_preview=True):
            if preview is not None:
                previews.append(preview)

        assert len(previews) > 0
        assert all(isinstance(p, bytes) for p in previews)

    def test_process_video_invalid_path_raises_error(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        valid_video = create_test_video(duration_frames=5)
        video_data = VideoData(
            input_video_path=str(valid_video),
            output_video_path=str(tmp_path / "out.mp4"),
            fps=30,
            frame_count=5,
            width=320,
            height=240,
            filter_indices=[],
            filter_matrices=np.array([], dtype=np.float32),
        )

        valid_video.unlink()

        with pytest.raises(VideoProcessingError):
            list(process_video(video_data))


class TestVideoDataSchema:
    def test_video_data_fields(
        self, create_test_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        video_path = create_test_video(duration_frames=10)
        data = VideoData(
            input_video_path=str(video_path),
            output_video_path=str(tmp_path / "output.mp4"),
            fps=30,
            frame_count=100,
            width=1920,
            height=1080,
            filter_indices=[0, 30, 60, 90],
            filter_matrices=np.random.rand(4, 20).astype(np.float32),
        )

        assert data.input_video_path == str(video_path)
        assert data.fps == 30
        assert data.frame_count == 100
        assert data.width == 1920
        assert data.height == 1080
        assert len(data.filter_indices) == 4
        assert data.filter_matrices.shape == (4, 20)

    def test_video_data_validates_nonexistent_path(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoData(
                input_video_path="nonexistent.mp4",
                output_video_path=str(tmp_path / "out.mp4"),
                fps=30,
                frame_count=10,
                width=320,
                height=240,
                filter_indices=[],
                filter_matrices=np.array([], dtype=np.float32),
            )
