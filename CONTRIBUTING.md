# Contributing to Dive Color Corrector

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kcrkor/dive-color-corrector.git
   cd dive-color-corrector
   ```

2. Create virtual environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   uv pip install -e ".[dev,gui]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- Python 3.11+ with type hints
- Google-style docstrings
- Line length: 100 characters
- Formatted with Ruff

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure tests pass and coverage is maintained
4. Update documentation if needed
5. Submit a PR with a clear description

## Reporting Issues

Please use GitHub Issues and include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior

## Project Structure

```
dive-color-corrector/
├── src/dive_color_corrector/    # Main package
│   ├── core/                    # Algorithm implementations
│   │   ├── color/               # Hue shift, filter matrix ops
│   │   ├── models/              # Deep SESR wrapper
│   │   └── processing/          # Image/video orchestration
│   ├── gui/                     # PySimpleGUI interface
│   └── cli.py                   # CLI entry point
├── tests/                       # Test suite
├── packaging/                   # Build configurations
│   ├── pyinstaller/             # PyInstaller specs
│   └── macos/                   # py2app config
└── scripts/                     # Utility scripts
```

## Architecture Notes

### Algorithm Flow
1. Compute average RGB → determine hue shift to reach MIN_AVG_RED=60
2. Apply hue shift using YUV weights
3. Build histogram → find normalizing interval via largest gap
4. Compute gain/offset per channel
5. Apply 20-element filter matrix

### Key Constraints
- TensorFlow 2.16.0-2.19.0 only (Keras 3.x compatibility)
- Never instantiate `DeepSESR()` per video frame
- Always call `.release()` on VideoCapture/VideoWriter
