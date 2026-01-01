# Claude Code Context

This file provides context for Claude Code and other AI assistants working with the Dive Color Corrector project.

## Project Summary

Dive Color Corrector restores natural colors in underwater images and videos. Water absorbs red light at depth, causing a blue-green cast. This tool applies algorithmic correction to restore true colors.

## Key Files

| File | Purpose |
|------|---------|
| `src/dive_color_corrector/core/correction.py` | Simple color correction algorithm (hue shift method) |
| `src/dive_color_corrector/core/color/filter.py` | Optimized filter operations with precomputation |
| `src/dive_color_corrector/core/processing/image.py` | Image processing with EXIF preservation |
| `src/dive_color_corrector/core/processing/video.py` | Two-pass video processing (analyze + process) |
| `src/dive_color_corrector/core/models/sesr.py` | Deep learning model (TensorFlow/Keras) |
| `src/dive_color_corrector/cli.py` | Command-line interface |
| `src/dive_color_corrector/gui/app.py` | PySimpleGUI desktop application |

## Architecture Patterns

### Video Processing Pipeline

The video processor uses a two-pass architecture:

1. **Analysis pass** (`analyze_video`): Samples filter matrices at intervals, yields frame count for progress
2. **Processing pass** (`process_video`): Uses precomputed interpolated matrices, yields progress/preview

```python
# Usage pattern
for item in analyze_video(input_path, output_path):
    if isinstance(item, dict):
        video_data = item  # Final result

for percent, preview in process_video(video_data, yield_preview=True):
    # Handle progress
```

### Filter Matrix

The color correction uses a 20-element filter matrix:
```
[red_gain, red_from_green, red_from_blue, 0, red_offset,
 0, green_gain, 0, 0, green_offset,
 0, 0, blue_gain, 0, blue_offset,
 0, 0, 0, 1, 0]
```

Indices used: 0, 1, 2, 4 (red), 6, 9 (green), 12, 14 (blue)

## Common Tasks

### Adding a New Processing Option

1. Add parameter to `correct_image`/`process_video` in processing modules
2. Update CLI parser in `cli.py`
3. Update GUI in `gui/app.py`
4. Update tests if applicable

### Modifying the Color Algorithm

1. Constants are in `core/color/constants.py` and `core/utils/constants.py`
2. Filter computation is in `core/color/filter.py::get_filter_matrix`
3. Filter application is in `core/color/filter.py::apply_filter`
4. Simple correction wrapper is in `core/correction.py::correct`

### Working with Video

Key considerations:
- `analyze_video` must complete before `process_video`
- `video_data` dict contains: `input_video_path`, `output_video_path`, `fps`, `frame_count`, `width`, `height`, `filter_indices`, `filter_matrices`
- `precompute_filter_matrices` uses float32 for memory efficiency
- Filter indices are sorted before interpolation

## Conventions

### Code Style
- Python 3.11+ type hints
- Google-style docstrings
- Ruff for linting (line length 100)
- Module-level `__all__` exports

### Imports
```python
# Standard library
import math
from pathlib import Path

# Third-party
import cv2
import numpy as np

# Local
from dive_color_corrector.core.color.filter import apply_filter
```

### Error Handling
- Raise `ValueError` for invalid inputs
- Use context managers for file/video handles
- Always release `VideoCapture` and `VideoWriter`

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_sesr.py -v
```

Test images are in `tests/samples/`.

## Build and Run

```bash
# Install dependencies
uv pip install -e ".[gui]"

# Run CLI
python -m dive_color_corrector image input.jpg output.jpg

# Run GUI
python -m dive_color_corrector
```

## Known Issues

1. **TensorFlow compatibility**: Only tensorflow-cpu 2.16.0-2.19.0 supported
2. **macOS builds**: TensorFlow has compatibility issues, excluded from CI
3. **Large videos**: Memory scales with frame count for precomputed matrices

## Future Work

See `docs/RUST_MIGRATION_PLAN.md` for planned Rust + Tauri migration.

## Questions to Clarify

When working on this project, consider asking about:
- Whether changes should support both simple and deep learning modes
- Performance requirements for video processing
- Target platforms for any new features
- Whether EXIF preservation is required for new image operations
