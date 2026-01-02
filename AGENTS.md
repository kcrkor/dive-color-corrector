# DIVE COLOR CORRECTOR - KNOWLEDGE BASE

**Generated:** 2026-01-01
**Commit:** 38c76e2
**Branch:** main

## OVERVIEW

Underwater image/video color correction tool. Restores red channel absorption via hue-shift algorithm or Deep SESR neural network. Python 3.11+, OpenCV, TensorFlow-CPU.

## STRUCTURE

```
dive-color-corrector/
├── src/dive_color_corrector/
│   ├── core/                    # Algorithm implementations
│   │   ├── color/               # Hue shift, filter matrix ops
│   │   ├── models/              # Deep SESR wrapper (Keras)
│   │   ├── processing/          # Image/video orchestration
│   │   ├── correction.py        # Legacy facade
│   │   └── exceptions.py        # Custom exceptions
│   ├── gui/                     # PySimpleGUI interface
│   ├── cli.py                   # CLI entry point
│   ├── logging.py               # Logging configuration
│   └── __main__.py              # Application entry
├── tests/
│   ├── fixtures/                # Test images
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── utils/                   # Test helpers
├── docs/                        # PRODUCTION_PLAN.md, PROJECT_OVERVIEW.md
├── packaging/                   # Build configurations
├── scripts/                     # Utility scripts
└── examples/                    # Sample before/after images
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Modify color algorithm | `core/color/filter.py`, `core/color/hue.py` | Filter matrix = 20-element array |
| Tune algorithm constants | `core/color/constants.py`, `core/utils/constants.py` | MIN_AVG_RED=60, THRESHOLD_RATIO=2000 |
| Add CLI option | `cli.py` | Uses argparse subparsers |
| Add GUI feature | `gui/app.py` | PySimpleGUI event loop |
| Change deep learning model | `core/models/sesr.py` | Input: 320x240, output: [-1,1] |
| Video processing flow | `core/processing/video.py` | Two-pass: analyze_video → process_video |
| Image processing | `core/processing/image.py` | correct_image() with EXIF preservation |
| Build executable | `.github/workflows/build.yml` | PyInstaller, Windows+Linux only |

## CONVENTIONS

### Import Order
```python
# stdlib → third-party → local
import math
from pathlib import Path

import cv2
import numpy as np

from dive_color_corrector.core.color.filter import apply_filter
```

### Code Style
- **Line length**: 100 (Ruff enforced)
- **Docstrings**: Google-style
- **Type hints**: Required (Python 3.11+)
- **Exports**: Use `__all__` in `__init__.py`

### Error Handling
- Use custom exceptions from `core/exceptions.py`
- Context managers for file/video handles
- Always call `.release()` on VideoCapture/VideoWriter
- Use `logger` from `logging.py` instead of `print()`

## ANTI-PATTERNS (THIS PROJECT)

| Forbidden | Reason |
|-----------|--------|
| Skip `analyze_video` before `process_video` | Filter matrices not computed; will crash |
| Use TensorFlow outside 2.16.0-2.19.0 | Keras 3.x breaking changes |
| Instantiate `DeepSESR()` per frame | Model reloads each time; 100x slower |
| Forget `.release()` on video handles | File corruption, handle leaks |
| Process macOS in CI | TensorFlow-CPU incompatible |
| Use `print()` for status updates | Use `logging` module instead |

## ALGORITHM QUICK REFERENCE

### Simple Correction (Default)
1. Compute average RGB → determine hue shift to reach MIN_AVG_RED=60
2. Apply hue shift: `r = 0.299 + 0.701*cos(h) + 0.168*sin(h)` (YUV weights)
3. Build histogram → find normalizing interval via largest gap
4. Compute gain/offset per channel → BLUE_MAGIC_VALUE=1.2 multiplier
5. Apply 20-element filter matrix

### Filter Matrix Layout
```
[red_gain, red←green, red←blue, 0, red_offset,   # indices 0-4
 0, green_gain, 0, 0, green_offset,               # indices 5-9
 0, 0, blue_gain, 0, blue_offset,                 # indices 10-14
 0, 0, 0, 1, 0]                                   # indices 15-19
```

### Video Processing
- **Pass 1** (`analyze_video`): Sample every 2 seconds → store filter matrices
- **Pass 2** (`process_video`): Interpolate matrices for all frames → apply per-frame
- Memory: `frame_count × 20 × 4 bytes` for precomputed matrices

## COMMANDS

```bash
# Setup development environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,gui]"
uv run pre-commit install

# Run GUI
uv run python -m dive_color_corrector

# Run CLI
uv run python -m dive_color_corrector image input.jpg output.jpg
uv run python -m dive_color_corrector video input.mp4 output.mp4 --use-deep

# Test
uv run pytest tests/
uv run pytest tests/unit --cov=src/dive_color_corrector

# Linting
uv run ruff check src tests
uv run mypy src
```

## NOTES

- **Packaging**: PyInstaller configurations in `packaging/pyinstaller/`
- **Rust migration**: Planned rewrite documented in `docs/RUST_MIGRATION_PLAN.md`
- **Model location**: Binary `.keras` file in `src/dive_color_corrector/models/`
- **Logging**: Configured via `src/dive_color_corrector/logging.py`

## SEE ALSO

- `claude.md` - Extended context for AI assistants
- `docs/PROJECT_OVERVIEW.md` - Detailed algorithm explanation
- `docs/PRODUCTION_PLAN.md` - Current status and roadmap
