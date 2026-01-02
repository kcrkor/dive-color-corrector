# Dive Color Corrector - Production Readiness Plan

**Created:** 2026-01-01
**Status:** Planning
**Author:** AI Assistant

## Executive Summary

Transform the current MVP into a production-ready application with proper structure, comprehensive testing, complete documentation, and developer tooling.

**Current State:** ~1,155 LOC Python, functional but needs polish
**Target State:** Production-ready with CI/CD, full test coverage, professional docs

---

## Phase 1: Directory Cleanup & Structure

### 1.1 Files to Remove/Relocate

| File/Directory | Action | Reason |
|----------------|--------|--------|
| `output/` | Delete | Empty folder, should be gitignored |
| `inspect_model.py` | Move to `scripts/` | Utility script |
| `Dive Color Corrector.spec` | Delete | Generated file, gitignore |
| `Dive Color Corrector.spec.template` | Move to `packaging/pyinstaller/` | Build config |
| `dive_color_corrector_cli.spec` | Delete | Orphaned spec file |
| `setup.py` | Move to `packaging/macos/` | Legacy py2app only |
| `hatch.toml` | Delete | Consolidate into pyproject.toml |
| `requirements.txt` | Delete | Redundant with pyproject.toml |
| `.pytest_cache/` | Gitignore | Cache |
| `.ruff_cache/` | Gitignore | Cache |
| `build/` | Gitignore | Build artifacts |
| `dist/` | Gitignore | Build artifacts |

### 1.2 Target Directory Structure

```
dive-color-corrector/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Test + lint on PR
│   │   ├── build.yml           # Build executables
│   │   └── release.yml         # Publish to PyPI + GitHub Releases
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── docs/
│   ├── index.md                # Main docs entry
│   ├── installation.md
│   ├── usage/
│   │   ├── cli.md
│   │   ├── gui.md
│   │   └── api.md
│   ├── development/
│   │   ├── contributing.md
│   │   ├── architecture.md
│   │   └── testing.md
│   ├── algorithms/
│   │   ├── hue-shift.md
│   │   └── deep-sesr.md
│   ├── PROJECT_OVERVIEW.md     # Existing
│   ├── RUST_MIGRATION_PLAN.md  # Existing
│   └── PRODUCTION_PLAN.md      # This file
├── examples/
│   ├── example.jpg             # Existing
│   ├── gui.jpg                 # Existing
│   └── scripts/
│       └── batch_process.py    # Example batch script
├── packaging/
│   ├── pyinstaller/
│   │   └── spec.template
│   └── macos/
│       └── setup.py            # py2app config
├── scripts/
│   ├── inspect_model.py        # Model inspection utility
│   └── convert_model.py        # Keras to ONNX conversion
├── src/
│   └── dive_color_corrector/
│       ├── core/
│       │   ├── color/
│       │   │   ├── __init__.py
│       │   │   ├── constants.py
│       │   │   ├── filter.py
│       │   │   └── hue.py
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   └── sesr.py
│       │   ├── processing/
│       │   │   ├── __init__.py
│       │   │   ├── image.py
│       │   │   └── video.py
│       │   ├── utils/
│       │   │   └── constants.py
│       │   ├── __init__.py
│       │   ├── correction.py
│       │   └── exceptions.py   # NEW: Custom exceptions
│       ├── gui/
│       │   ├── assets/
│       │   │   └── logo.png
│       │   ├── __init__.py
│       │   └── app.py
│       ├── models/
│       │   └── deep_sesr_2x_1d.keras
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       └── logging.py          # NEW: Logging configuration
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── fixtures/               # Renamed from samples/
│   │   ├── underwater.jpg
│   │   ├── corrected.jpg
│   │   └── comparison.jpg
│   ├── utils/                  # Test helpers
│   │   ├── __init__.py
│   │   ├── compare_images.py
│   │   └── create_test_image.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_filter.py
│   │   ├── test_hue.py
│   │   ├── test_sesr.py
│   │   └── test_cli.py
│   └── integration/
│       ├── __init__.py
│       ├── test_image_processing.py
│       └── test_video_processing.py
├── .gitignore                  # Updated
├── .pre-commit-config.yaml     # NEW
├── AGENTS.md                   # Existing
├── CHANGELOG.md                # NEW
├── CONTRIBUTING.md             # NEW
├── LICENSE                     # Existing
├── README.md                   # Enhanced
├── claude.md                   # Existing
└── pyproject.toml              # Enhanced
```

---

## Phase 2: Dependency & Configuration Fixes

### 2.1 Updated pyproject.toml

```toml
[project]
name = "dive_color_corrector"
dynamic = ["version"]
description = "Restore natural colors in underwater images and videos"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "KCRK"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    "Topic :: Scientific/Engineering :: Image Processing",
]
keywords = ["underwater", "color-correction", "image-processing", "video-processing", "scuba", "diving"]

dependencies = [
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",                    # CRITICAL: Currently missing!
    "tensorflow-cpu>=2.16.0,<=2.19.0",
    "keras>=3.5.0",
]

[project.optional-dependencies]
gui = ["PySimpleGUI>=4.60.0"]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]
build = [
    "pyinstaller>=6.0.0",
    "pillow>=10.0.0",
]

[project.scripts]
dive-color-corrector = "dive_color_corrector.cli:main"

[project.urls]
Homepage = "https://github.com/kcrkor/dive-color-corrector"
Documentation = "https://github.com/kcrkor/dive-color-corrector#readme"
Repository = "https://github.com/kcrkor/dive-color-corrector"
Issues = "https://github.com/kcrkor/dive-color-corrector/issues"
Changelog = "https://github.com/kcrkor/dive-color-corrector/blob/main/CHANGELOG.md"

[build-system]
requires = ["hatchling>=1.18.0"]
build-backend = "hatchling.build"

[tool.hatch.version]
path = "src/dive_color_corrector/__init__.py"

[tool.hatch.build]
packages = ["src/dive_color_corrector"]

[tool.hatch.build.targets.wheel]
packages = ["src/dive_color_corrector"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "B", "UP", "RUF", "PTH", "SIM"]
ignore = []

[tool.ruff.lint.isort]
known-first-party = ["dive_color_corrector"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.coverage.run]
source = ["src/dive_color_corrector"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
]
```

### 2.2 Updated .gitignore

```gitignore
# Build artifacts
build/
dist/
*.spec
!packaging/**/*.spec
!packaging/**/*.template

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
ENV/
.python-version

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Cache
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/
.tox/

# Output
output/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Local config
.env
.env.local
```

---

## Phase 3: Code Quality & Testing

### 3.1 Custom Exceptions

Create `src/dive_color_corrector/core/exceptions.py`:

```python
"""Custom exceptions for dive color corrector."""


class DiveColorCorrectorError(Exception):
    """Base exception for dive color corrector."""


class VideoProcessingError(DiveColorCorrectorError):
    """Error during video processing."""


class ImageProcessingError(DiveColorCorrectorError):
    """Error during image processing."""


class ModelLoadError(DiveColorCorrectorError):
    """Error loading ML model."""


class InvalidInputError(DiveColorCorrectorError):
    """Invalid input file or parameters."""
```

### 3.2 Logging Configuration

Create `src/dive_color_corrector/logging.py`:

```python
"""Logging configuration for dive color corrector."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("dive_color_corrector")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the application logger."""
    return logging.getLogger("dive_color_corrector")
```

### 3.3 Pre-commit Configuration

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=15000']  # Allow model files up to 15MB
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-pillow]
        args: [--ignore-missing-imports]
```

### 3.4 Test Coverage Targets

| Test File | Module | Coverage Target |
|-----------|--------|-----------------|
| `test_filter.py` | `core/color/filter.py` | 90% |
| `test_hue.py` | `core/color/hue.py` | 95% |
| `test_sesr.py` | `core/models/sesr.py` | 85% |
| `test_cli.py` | `cli.py` | 80% |
| `test_image_processing.py` | `core/processing/image.py` | 85% |
| `test_video_processing.py` | `core/processing/video.py` | 75% |

**Overall Target: 80%+ coverage**

### 3.5 Test Fixtures (conftest.py)

```python
"""Pytest fixtures for dive color corrector tests."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def underwater_image(fixtures_dir: Path) -> Path:
    """Return path to underwater test image."""
    return fixtures_dir / "underwater.jpg"


@pytest.fixture
def random_rgb_image() -> np.ndarray:
    """Generate a random RGB image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_filter_matrix() -> np.ndarray:
    """Return a sample filter matrix for testing."""
    return np.array([
        1.0, 0.0, 0.0, 0, 0.0,
        0, 1.0, 0, 0, 0.0,
        0, 0, 1.0, 0, 0.0,
        0, 0, 0, 1, 0,
    ], dtype=np.float32)
```

---

## Phase 4: Documentation

### 4.1 CHANGELOG.md Template

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production readiness plan documentation
- Custom exceptions module
- Logging configuration module
- Pre-commit hooks configuration
- Expanded test suite

### Changed
- Reorganized project directory structure
- Enhanced pyproject.toml with all dependencies
- Moved utility scripts to dedicated directories

### Fixed
- Circular import in correction.py module
- Missing pillow dependency

### Removed
- Redundant configuration files (hatch.toml, requirements.txt)
- Empty output directory
- Orphaned spec files

## [1.2.0] - 2025-XX-XX

### Added
- Deep SESR model integration for ML-based enhancement
- Two-pass video processing with filter interpolation
- GUI checkbox for deep learning mode

### Changed
- Improved filter matrix precomputation for videos
- Enhanced EXIF preservation in image processing

## [1.1.0] - 2025-XX-XX

### Added
- Video processing support
- Progress indicators in GUI
- CLI video subcommand

## [1.0.0] - 2025-XX-XX

### Added
- Initial release
- Underwater image color correction
- PySimpleGUI desktop application
- CLI interface
```

### 4.2 CONTRIBUTING.md Template

```markdown
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
```

---

## Phase 5: CI/CD Improvements

### 5.1 CI Workflow (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: uv pip install ruff mypy

      - name: Run Ruff
        run: ruff check src tests

      - name: Run Ruff format check
        run: ruff format --check src tests

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.11", "3.12"]
        exclude:
          # TensorFlow issues on some combinations
          - os: windows-latest
            python-version: "3.12"

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true

      - name: Install dependencies
        run: uv pip install -e ".[dev,gui]"

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
        with:
          files: ./coverage.xml
```

### 5.2 Release Workflow (.github/workflows/release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    uses: ./.github/workflows/build.yml

  publish-pypi:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.11"

      - name: Build package
        run: uv build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1

  create-release:
    needs: [build, publish-pypi]
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          path: artifacts

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: artifacts/**/*
          generate_release_notes: true
```

---

## Phase 6: Performance & UX Improvements

### 6.1 Performance Issues & Solutions

| Issue | Impact | Solution | Priority |
|-------|--------|----------|----------|
| `DeepSESR()` instantiated per frame | 100x slower video processing | Singleton or pass model instance | High |
| Print statements everywhere | No log levels, clutters output | Replace with logging module | Medium |
| No GPU support | Slow ML inference | Add optional `tensorflow[gpu]` | Low |
| Filter computed on full image | Slow for large images | Already resizes to 256x256 | Done |

### 6.2 GUI Improvements

| Feature | Current State | Improvement |
|---------|---------------|-------------|
| Drag-and-drop | Not supported | Add file drop zone |
| Progress | Basic bar | Add ETA, speed indicator |
| Errors | Generic message | User-friendly error dialogs |
| Theme | DarkBlue only | Add light/dark toggle |
| Preview | Half-width split | Add slider comparison |

### 6.3 CLI Improvements

| Feature | Current State | Improvement |
|---------|---------------|-------------|
| Verbosity | None | Add `--verbose` / `--quiet` |
| Progress | Print to stdout | Add `--progress` for scripts |
| Batch | Single file only | Add glob pattern support |
| Version | Not shown | Add `--version` flag |
| Output naming | `corrected_` prefix | Add `--output-pattern` |

---

## Phase 7: Implementation Checklist

### Week 1: Critical Fixes & Cleanup

- [x] Add `pillow>=10.0.0` to pyproject.toml
- [x] Delete empty `output/` directory
- [x] Delete generated `*.spec` files (keep template)
- [x] Create `packaging/` directory structure
- [x] Move `setup.py` to `packaging/macos/`
- [x] Move `inspect_model.py` to `scripts/`
- [x] Delete `hatch.toml` (consolidate to pyproject.toml)
- [x] Delete `requirements.txt`
- [x] Update `.gitignore`
- [x] Create `CHANGELOG.md`
- [x] Create `CONTRIBUTING.md`

### Week 2: Code Quality

- [x] Create `src/dive_color_corrector/core/exceptions.py`
- [x] Create `src/dive_color_corrector/logging.py`
- [x] Replace print statements with logging
- [x] Create `.pre-commit-config.yaml`
- [x] Run `pre-commit install`
- [x] Fix any linting issues
- [x] Reorganize tests directory structure
- [x] Create `tests/conftest.py`
- [x] Add unit tests for filter module
- [x] Add unit tests for hue module
- [x] Add unit tests for CLI

### Week 3: Documentation & CI

- [ ] Enhance README.md with badges
- [x] Add installation instructions for pip/uv
- [ ] Add usage examples with screenshots
- [x] Create `.github/workflows/ci.yml`
- [x] Create `.github/workflows/release.yml`
- [ ] Set up Codecov integration
- [ ] Add integration tests
- [ ] Achieve 80%+ test coverage

### Week 4: Polish & Release

- [x] Add mypy type checking
- [x] Fix all type errors
- [ ] Performance optimization for DeepSESR
- [ ] GUI improvements (optional)
- [x] CLI improvements (--verbose flag added)
- [ ] Final testing on Windows + Linux
- [ ] Tag v1.3.0 release
- [ ] Publish to PyPI (optional)

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Test coverage | 39% | 80%+ |
| Linting errors | 0 | 0 |
| Type errors | 0 | 0 |
| Documentation | Basic README | Full docs |
| CI/CD | Build only | Lint + Test + Build + Release |
| Root files | 15+ loose | Organized |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TensorFlow version breaks | Medium | High | Pin exact versions, test matrix |
| macOS build fails | High | Medium | Skip macOS in CI, document limitation |
| Test coverage gaps | Medium | Medium | Focus on critical paths first |
| Breaking changes | Low | High | Semantic versioning, changelog |

---

## Notes

- This plan assumes the project will remain Python-based for now
- Rust migration is documented separately in `RUST_MIGRATION_PLAN.md`
- PyPI publication is optional but recommended
- macOS support requires manual testing due to CI limitations
