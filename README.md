# Dive Color Corrector

[![PyPI version](https://badge.fury.io/py/dive-color-corrector.svg)](https://badge.fury.io/py/dive-color-corrector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/kcrkor/dive-color-corrector/actions/workflows/ci.yml/badge.svg)](https://github.com/kcrkor/dive-color-corrector/actions/workflows/ci.yml)

Restore natural colors in underwater images and videos. Corrects the blue/green color cast caused by water absorbing red light.

## Features

- Color correction for underwater images and videos
- Two correction modes: fast hue-shift algorithm or Deep SESR neural network
- GUI application with real-time preview
- CLI for batch processing
- Cross-platform: Windows, Linux

## Installation

### Download Pre-built Executables

Download from [Releases](https://github.com/kcrkor/dive-color-corrector/releases):

| Version | Description |
|---------|-------------|
| **Dive Color Corrector** | Full version with Deep SESR neural network (~100MB) |
| **Dive Color Corrector Lite** | Lightweight version, fast algorithm only (~30MB) |

### From PyPI

```bash
# Lite (GUI only, no neural network)
pip install dive-color-corrector[gui]

# Full (GUI + Deep SESR neural network)
pip install dive-color-corrector[full]
```

### From Source

```bash
git clone https://github.com/kcrkor/dive-color-corrector.git
cd dive-color-corrector

uv venv && source .venv/bin/activate

# Lite
uv pip install -e ".[gui]"

# Full (with SESR)
uv pip install -e ".[gui,sesr]"
```

## Usage

### GUI Application

```bash
python -m dive_color_corrector
```

### Command Line

```bash
# Single image
python -m dive_color_corrector image input.jpg output.jpg

# Single image with Deep SESR (requires full install)
python -m dive_color_corrector image input.jpg output.jpg --sesr

# Video
python -m dive_color_corrector video input.mp4 output.mp4

# Batch process directory
python -m dive_color_corrector batch ./raw ./corrected
```

## Examples

### Before / After
![Example](./examples/example.jpg)

### Sample Video
[![Video](https://img.youtube.com/vi/NEpl41-LMBs/0.jpg)](https://www.youtube.com/watch?v=NEpl41-LMBs)

## Algorithm

Two correction methods available:

1. **Hue-shift (default)**: Fast algorithm that shifts hue to restore red channel. Best for most underwater photos.

2. **Deep SESR** (`--sesr` flag): Neural network trained on underwater images. Provides super-resolution (2x) and enhanced color correction. Slower but can produce better results for challenging images.

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev,gui,sesr]"

# Run tests
pytest tests/

# Run linting
ruff check src tests
mypy src
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- Algorithm inspired by [nikolajbech/underwater-image-color-correction](https://github.com/nikolajbech/underwater-image-color-correction)
- Thanks to [bornfree](https://github.com/bornfree) and all contributors
