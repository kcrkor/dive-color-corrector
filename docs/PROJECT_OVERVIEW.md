# Dive Color Corrector - Project Overview

## Purpose

Dive Color Corrector is a Python application designed to restore natural colors in underwater photographs and videos. Water absorbs red light progressively with depth, causing underwater images to appear blue-green. This tool applies algorithmic color correction to restore vibrant, true-to-life colors.

## Architecture

```
dive-color-corrector/
├── src/dive_color_corrector/
│   ├── core/                    # Core processing logic
│   │   ├── color/               # Color manipulation utilities
│   │   │   ├── constants.py     # Algorithm constants
│   │   │   ├── filter.py        # Filter matrix operations
│   │   │   └── hue.py           # Hue shifting operations
│   │   ├── models/              # Deep learning models
│   │   │   └── sesr.py          # Deep SESR underwater enhancement
│   │   ├── processing/          # Image/video processing
│   │   │   ├── image.py         # Single image correction
│   │   │   └── video.py         # Video processing pipeline
│   │   ├── utils/               # Shared utilities
│   │   │   └── constants.py     # Global constants
│   │   └── correction.py        # Simple correction algorithm
│   ├── gui/                     # GUI application
│   │   └── app.py               # PySimpleGUI interface
│   ├── cli.py                   # Command-line interface
│   └── __main__.py              # Entry point
├── tests/                       # Test suite
├── examples/                    # Sample images
└── docs/                        # Documentation
```

## Core Algorithms

### 1. Simple Color Correction (Hue Shift Method)

The primary algorithm works by:

1. **Analyze Average Color**: Calculate mean RGB values of the image
2. **Compute Hue Shift**: Determine rotation angle to restore red channel to target level (MIN_AVG_RED = 60)
3. **Build Filter Matrix**: Generate a 20-element color transformation matrix containing:
   - Red channel gains and offsets (with cross-channel contributions)
   - Green channel gain and offset
   - Blue channel gain and offset (with BLUE_MAGIC_VALUE = 1.2 multiplier)
4. **Histogram Normalization**: Find optimal value ranges using histogram thresholding
5. **Apply Filter**: Transform each pixel using the computed matrix

Key constants:
- `THRESHOLD_RATIO = 2000` - Histogram threshold divisor
- `MIN_AVG_RED = 60` - Target minimum average red value
- `MAX_HUE_SHIFT = 120` - Maximum allowed hue rotation
- `BLUE_MAGIC_VALUE = 1.2` - Blue channel enhancement factor

### 2. Deep Learning Model (Deep SESR)

An optional TensorFlow/Keras-based enhancement using the Deep SESR architecture:
- Input: 320x240 RGB image normalized to [0, 1]
- Output: Enhanced image with values in [-1, 1]
- Post-processing: Scale to [0, 255], resize to original dimensions

## Video Processing Pipeline

For videos, the system uses an optimized two-pass approach:

### Pass 1: Analysis (`analyze_video`)
- Sample filter matrices every N seconds (SAMPLE_SECONDS = 2)
- Store frame indices and corresponding filter matrices
- Yield progress updates for UI feedback

### Pass 2: Processing (`process_video`)
- Precompute interpolated filter matrices for all frames
- Apply correction frame-by-frame using precomputed values
- Write corrected frames to output video

Optimizations:
- `precompute_filter_matrices`: Interpolates filter coefficients for every frame upfront
- `apply_filter`: Uses in-place float32 operations to minimize memory allocations
- Filter matrices stored as float32 to reduce memory footprint

## Dependencies

### Core
- `opencv-python >= 4.8.0` - Image/video I/O and processing
- `numpy >= 1.24.0` - Numerical operations
- `pillow` - EXIF data preservation

### Deep Learning (Optional)
- `tensorflow-cpu >= 2.16.0, <= 2.19.0` - Deep SESR model inference
- `keras >= 3.5.0` - Model loading

### GUI
- `PySimpleGUI >= 4.60.0` - Desktop interface

## Interfaces

### Command Line
```bash
# Image correction
dive-color-corrector image input.jpg output.jpg [--use-deep]

# Video correction
dive-color-corrector video input.mp4 output.mp4 [--use-deep]
```

### GUI Application
- File browser for batch selection
- Real-time preview with before/after comparison
- Progress tracking for video processing
- Toggle for deep learning mode

### Programmatic API
```python
from dive_color_corrector.core.processing.image import correct_image
from dive_color_corrector.core.processing.video import analyze_video, process_video

# Image
preview_bytes = correct_image("input.jpg", "output.jpg", use_deep=False)

# Video
for item in analyze_video("input.mp4", "output.mp4"):
    if isinstance(item, dict):
        video_data = item

for percent, preview in process_video(video_data, yield_preview=True):
    print(f"Progress: {percent:.1f}%")
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Image correction | O(pixels) | Single-pass filter application |
| Video analysis | O(frames/sample_rate) | Samples every 2 seconds |
| Filter precomputation | O(frames * 20) | Linear interpolation |
| Video processing | O(frames * pixels) | Per-frame filter application |

Memory usage for video:
- Filter matrices: `frame_count * 20 * 4 bytes` (float32)
- Frame buffer: `width * height * 3 bytes` per frame

## Future Considerations

- GPU acceleration for filter operations
- Parallel frame processing
- Additional color correction algorithms
- RAW image format support
