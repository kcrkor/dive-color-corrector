import runpy
from unittest.mock import patch


def test_main_exec() -> None:
    with patch("dive_color_corrector.cli.main") as mock_main:
        runpy.run_module("dive_color_corrector.__main__", run_name="__main__")
        mock_main.assert_called_once()
