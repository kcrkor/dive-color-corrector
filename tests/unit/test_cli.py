from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dive_color_corrector.cli import main
from dive_color_corrector.core.schemas import VideoData


def test_cli_no_args() -> None:
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 1


def test_cli_invalid_mode() -> None:
    with pytest.raises(SystemExit):
        main(["invalid"])


@patch("dive_color_corrector.cli.correct_image")
@patch("dive_color_corrector.cli.Path.exists")
def test_cli_image_mode(mock_exists: Any, mock_correct: Any) -> None:
    mock_exists.return_value = True
    main(["image", "input.jpg", "output.jpg"])
    mock_correct.assert_called_once_with("input.jpg", "output.jpg", use_deep=False)


@patch("dive_color_corrector.cli.analyze_video")
@patch("dive_color_corrector.cli.process_video")
@patch("dive_color_corrector.cli.Path.exists")
def test_cli_video_mode(mock_exists: Any, mock_process: Any, mock_analyze: Any) -> None:
    mock_exists.return_value = True
    mock_video_data = MagicMock(spec=VideoData)
    mock_video_data.input_video_path = "input.mp4"
    mock_video_data.output_video_path = "output.mp4"
    mock_video_data.name = "input.mp4"
    mock_analyze.return_value = [0, 50, mock_video_data]
    mock_process.return_value = [(0.0, None), (100.0, None)]

    with patch("dive_color_corrector.cli.Path") as mock_path:
        mock_input = MagicMock()
        mock_input.exists.return_value = True
        mock_input.name = "input.mp4"
        mock_input.__str__.return_value = "input.mp4"  # type: ignore[attr-defined]

        mock_output = MagicMock()
        mock_output.parent.mkdir.return_value = None
        mock_output.__str__.return_value = "output.mp4"  # type: ignore[attr-defined]

        mock_path.side_effect = lambda x: mock_input if x == "input.mp4" else mock_output

        main(["video", "input.mp4", "output.mp4"])

    mock_analyze.assert_called_once_with("input.mp4", "output.mp4")
    mock_process.assert_called_once()


@patch("dive_color_corrector.cli.process_batch")
@patch("dive_color_corrector.cli.Path.is_dir")
@patch("dive_color_corrector.cli.Path.exists")
def test_cli_batch_mode(mock_exists: Any, mock_is_dir: Any, mock_batch: Any) -> None:
    mock_exists.return_value = True
    mock_is_dir.return_value = True
    main(["batch", "input_dir", "output_dir"])
    mock_batch.assert_called_once()


@patch("dive_color_corrector.cli.SESR_AVAILABLE", False)
def test_cli_sesr_unavailable() -> None:
    with patch("dive_color_corrector.cli.setup_logging"):
        with pytest.raises(SystemExit) as e:
            main(["image", "input.jpg", "output.jpg", "--sesr"])
        assert e.value.code == 1


def test_cli_verbose_logging() -> None:
    with (
        patch("dive_color_corrector.cli.setup_logging") as mock_setup,
        patch("dive_color_corrector.cli.Path.exists") as mock_exists,
        patch("dive_color_corrector.cli.process_single_image") as mock_proc,
    ):
        mock_exists.return_value = True
        mock_proc.return_value = True
        main(["-v", "image", "input.jpg", "output.jpg"])
        mock_setup.assert_called_once_with(level="DEBUG")


@patch("dive_color_corrector.cli.Path.exists")
def test_cli_input_not_exists(mock_exists: Any) -> None:
    mock_exists.return_value = False
    with pytest.raises(SystemExit) as e:
        main(["image", "input.jpg", "output.jpg"])
    assert e.value.code == 1


@patch("dive_color_corrector.cli.process_single_image")
@patch("dive_color_corrector.cli.process_single_video")
@patch("dive_color_corrector.cli.Path")
def test_batch_processing_logic(mock_path: Any, mock_video: Any, mock_image: Any) -> None:
    from dive_color_corrector.cli import process_batch

    mock_input_dir = MagicMock()
    mock_output_dir = MagicMock()

    mock_img = MagicMock()
    mock_img.suffix.lower.return_value = ".jpg"
    mock_img.name = "test.jpg"
    mock_img.stem = "test"
    mock_img.__lt__.side_effect = lambda other: mock_img.name < other.name

    mock_vid = MagicMock()
    mock_vid.suffix.lower.return_value = ".mp4"
    mock_vid.name = "test.mp4"
    mock_vid.stem = "test"
    mock_vid.__lt__.side_effect = lambda other: mock_vid.name < other.name

    mock_input_dir.iterdir.return_value = [mock_img, mock_vid]

    process_batch(mock_input_dir, mock_output_dir, False, False, False)

    mock_image.assert_called_once()
    mock_video.assert_called_once()


@patch("dive_color_corrector.cli.process_single_image")
@patch("dive_color_corrector.cli.process_single_video")
@patch("dive_color_corrector.cli.Path")
def test_batch_processing_images_only(mock_path: Any, mock_video: Any, mock_image: Any) -> None:
    from dive_color_corrector.cli import process_batch

    mock_input_dir = MagicMock()
    mock_output_dir = MagicMock()

    mock_img = MagicMock()
    mock_img.suffix.lower.return_value = ".jpg"
    mock_img.name = "test.jpg"
    mock_img.stem = "test"
    mock_img.__lt__.side_effect = lambda other: mock_img.name < other.name

    mock_vid = MagicMock()
    mock_vid.suffix.lower.return_value = ".mp4"
    mock_vid.name = "test.mp4"
    mock_vid.__lt__.side_effect = lambda other: mock_vid.name < other.name

    mock_input_dir.iterdir.return_value = [mock_img, mock_vid]

    process_batch(mock_input_dir, mock_output_dir, False, True, False)

    mock_image.assert_called_once()
    mock_video.assert_not_called()
