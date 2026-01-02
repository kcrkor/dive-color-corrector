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
    mock_analyze.return_value = [0, 50, mock_video_data]
    mock_process.return_value = [(0.0, None), (100.0, None)]

    main(["video", "input.mp4", "output.mp4"])

    mock_analyze.assert_called_once_with("input.mp4", "output.mp4")
    mock_process.assert_called_once()
