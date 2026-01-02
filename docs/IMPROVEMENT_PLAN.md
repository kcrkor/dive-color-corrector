# Dive Color Corrector - Complete Improvement Plan

**Created:** 2026-01-02
**Current Version:** 1.2.1
**Author:** Sisyphus

---

## Executive Summary

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Test Coverage | 56% | 80%+ | High |
| LOC | 1,308 | ~1,500 | - |
| CI/CD | Partial | Full | High |
| Documentation | Basic | Complete | Medium |
| Performance | Good | Optimized | Low |
| Distribution | PyPI + Binaries | + macOS | Medium |

---

## Phase 1: Immediate Fixes (This Week)

### 1.1 Critical CI/CD Issues

| Issue | Status | Action Required |
|-------|--------|-----------------|
| `release.yml` calling non-reusable workflows | **FIXED** | Added `workflow_call` to ci.yml and build.yml |
| Duplicate release logic | Open | Consolidate `build.yml` and `release.yml` |
| Missing PyPI environment | Open | Create `pypi` environment in GitHub Settings |

**Action Items:**
```bash
# Consolidate workflows - release.yml already calls build.yml
# Just need to configure PyPI trusted publishing
gh secret set PYPI_API_TOKEN  # Or use trusted publishing
```

### 1.2 Test Coverage Gaps

| Module | Current | Target | Gap Analysis |
|--------|---------|--------|--------------|
| `video.py` | 11% | 75% | No video integration tests |
| `gui/app.py` | 0% | 50% | No GUI tests (acceptable) |
| `cli.py` | 56% | 80% | Missing subcommand tests |
| `exceptions.py` | 0% | 100% | Unused - wire up or remove |
| `__main__.py` | 0% | 100% | Trivial - add basic test |

**Priority Test Files to Create:**
1. `tests/integration/test_video_processing.py` - Video end-to-end
2. `tests/unit/test_cli_commands.py` - All CLI subcommands
3. `tests/unit/test_exceptions.py` - Exception hierarchy

### 1.3 Unused Code Cleanup

| File | Issue | Action |
|------|-------|--------|
| `core/exceptions.py` | Defined but never raised | Wire into processing code |
| `core/schemas.py` | Pydantic models partially used | Complete integration or remove |

---

## Phase 2: Test Infrastructure (Week 1-2)

### 2.1 Video Processing Tests

```python
# tests/integration/test_video_processing.py
import pytest
from pathlib import Path
from dive_color_corrector.core.processing.video import analyze_video, process_video

@pytest.fixture
def sample_video(tmp_path):
    """Create a minimal test video."""
    # Use OpenCV to create a 2-second test video
    ...

@pytest.mark.slow
def test_video_analyze_yields_progress(sample_video, tmp_path):
    """Test that analyze_video yields progress updates."""
    output = tmp_path / "output.mp4"
    results = list(analyze_video(str(sample_video), str(output)))

    # Should yield progress percentages
    progress_items = [r for r in results if isinstance(r, (int, float))]
    assert len(progress_items) > 0

    # Last item should be video data dict
    assert isinstance(results[-1], dict)

@pytest.mark.slow
def test_video_full_pipeline(sample_video, tmp_path):
    """Test complete video processing pipeline."""
    output = tmp_path / "output.mp4"

    # Analyze
    video_data = None
    for item in analyze_video(str(sample_video), str(output)):
        if isinstance(item, dict):
            video_data = item

    # Process
    for percent, _ in process_video(video_data):
        assert 0 <= percent <= 100

    assert output.exists()
```

### 2.2 CLI Integration Tests

```python
# tests/unit/test_cli_commands.py
import pytest
from click.testing import CliRunner
from dive_color_corrector.cli import main

def test_image_command(fixtures_dir, tmp_path):
    """Test image subcommand."""
    runner = CliRunner()
    input_path = fixtures_dir / "underwater.jpg"
    output_path = tmp_path / "output.jpg"

    result = runner.invoke(main, ["image", str(input_path), str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()

def test_batch_command(fixtures_dir, tmp_path):
    """Test batch subcommand."""
    runner = CliRunner()
    result = runner.invoke(main, ["batch", str(fixtures_dir), str(tmp_path)])
    assert result.exit_code == 0

def test_version_flag():
    """Test --version flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.2" in result.output
```

### 2.3 Test Fixtures Enhancement

```python
# tests/conftest.py - additions
import cv2
import numpy as np

@pytest.fixture
def create_test_video(tmp_path):
    """Factory fixture to create test videos."""
    def _create(duration_seconds=2, fps=30, width=320, height=240):
        path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

        for i in range(int(duration_seconds * fps)):
            # Create blue-tinted frame (simulating underwater)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 100 + (i % 50)  # Blue varies
            frame[:, :, 1] = 80               # Green constant
            frame[:, :, 2] = 30               # Red low (underwater effect)
            writer.write(frame)

        writer.release()
        return path

    return _create
```

---

## Phase 3: Code Quality (Week 2-3)

### 3.1 Exception Integration

Wire up custom exceptions throughout the codebase:

```python
# core/processing/video.py
from dive_color_corrector.core.exceptions import (
    VideoProcessingError,
    InvalidInputError,
)

def analyze_video(input_path: str, output_path: str):
    if not Path(input_path).exists():
        raise InvalidInputError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise VideoProcessingError(f"Failed to open video: {input_path}")
    ...
```

### 3.2 Logging Integration

Replace remaining print statements:

```bash
# Find print statements to replace
grep -rn "print(" src/ --include="*.py" | grep -v "__pycache__"
```

```python
# Replace pattern:
# print(f"Processing frame {i}")
# With:
logger.info("Processing frame %d", i)
```

### 3.3 Type Annotations Audit

```bash
# Run mypy with strict mode
uv run mypy src --strict --ignore-missing-imports

# Expected issues to fix:
# - Missing return types
# - Untyped function arguments
# - Optional type handling
```

---

## Phase 4: Feature Improvements (Week 3-4)

### 4.1 CLI Enhancements

| Feature | Current | Enhancement |
|---------|---------|-------------|
| Verbosity | None | Add `--verbose` / `--quiet` flags |
| Progress | Print | Add `--progress` for machine-readable output |
| Batch patterns | Directory only | Add glob pattern support (`*.jpg`) |
| Output naming | `corrected_` prefix | Add `--output-pattern` option |
| Dry run | None | Add `--dry-run` for batch operations |

```python
# cli.py additions
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
@click.option('--dry-run', is_flag=True, help='Show what would be processed')
```

### 4.2 GUI Improvements

| Feature | Priority | Effort |
|---------|----------|--------|
| Drag-and-drop file selection | High | Low |
| Progress ETA indicator | Medium | Low |
| Comparison slider (before/after) | Medium | Medium |
| Theme toggle (dark/light) | Low | Low |
| Batch progress (file N of M) | High | Low |

### 4.3 Algorithm Improvements

| Optimization | Impact | Effort |
|--------------|--------|--------|
| DeepSESR singleton pattern | High (100x for video) | Low |
| NumPy vectorization in filter.py | Medium | Medium |
| Parallel frame processing | High | High |

**DeepSESR Singleton Fix:**
```python
# core/models/sesr.py
class DeepSESR:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            self._model = ort.InferenceSession(...)
        return self._model
```

---

## Phase 5: Documentation (Week 4)

### 5.1 User Documentation

| Document | Status | Action |
|----------|--------|--------|
| README.md | Basic | Add badges, expand examples |
| Installation guide | In README | Expand with troubleshooting |
| CLI reference | None | Create `docs/usage/cli.md` |
| GUI guide | None | Create `docs/usage/gui.md` with screenshots |
| API reference | None | Generate with `mkdocs` |

### 5.2 Developer Documentation

| Document | Status | Action |
|----------|--------|--------|
| CONTRIBUTING.md | Exists | Update with test requirements |
| Architecture docs | PROJECT_OVERVIEW.md | Expand with diagrams |
| Algorithm explanation | Brief | Create `docs/algorithms/hue-shift.md` |

### 5.3 README Badges

```markdown
[![CI](https://github.com/kcrkor/dive-color-corrector/actions/workflows/ci.yml/badge.svg)](...)
[![Coverage](https://codecov.io/gh/kcrkor/dive-color-corrector/branch/main/graph/badge.svg)](...)
[![PyPI](https://img.shields.io/pypi/v/dive-color-corrector)](...)
[![Python](https://img.shields.io/pypi/pyversions/dive-color-corrector)](...)
[![License](https://img.shields.io/github/license/kcrkor/dive-color-corrector)](...)
```

---

## Phase 6: Distribution (Week 5)

### 6.1 PyPI Publication

**Current Status:** Workflow exists but PyPI environment not configured.

**Steps:**
1. Create PyPI account (if not exists)
2. Configure trusted publishing in PyPI
3. Create `pypi` environment in GitHub repo settings
4. Test with TestPyPI first

```yaml
# release.yml - already configured for trusted publishing
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
```

### 6.2 Binary Distribution

| Platform | Status | Action |
|----------|--------|--------|
| Windows | Working | Monitor for issues |
| Linux | Working | Add AppImage format |
| macOS | Not supported | Document limitation (TensorFlow-CPU) |

### 6.3 Version Management

```python
# src/dive_color_corrector/__init__.py
__version__ = "1.2.1"

# Bump version script
# scripts/bump_version.py
import re
import sys

def bump(part: str):
    """Bump version: major, minor, patch"""
    ...
```

---

## Phase 7: Performance (Week 6+)

### 7.1 Profiling

```bash
# Profile image processing
python -m cProfile -o profile.out -m dive_color_corrector image input.jpg output.jpg

# Analyze
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumtime').print_stats(20)"
```

### 7.2 Optimization Targets

| Function | Current | Potential Improvement |
|----------|---------|----------------------|
| `apply_filter()` | NumPy | Numba JIT (2-5x) |
| `hue_shift_red()` | NumPy | Numba JIT (2-5x) |
| Video frame iteration | Sequential | Parallel with ThreadPoolExecutor |
| DeepSESR instantiation | Per-call | Singleton (100x for video) |

### 7.3 Memory Optimization

| Issue | Current | Improvement |
|-------|---------|-------------|
| Filter matrix storage | float64 | float32 (50% reduction) |
| Video frames in memory | All at once | Streaming with generator |
| Preview images | Full resolution | Thumbnail for GUI |

---

## Phase 8: Future Roadmap

### 8.1 Short-term (v1.3.0)

- [ ] 80%+ test coverage
- [ ] PyPI publication working
- [ ] CLI verbose/quiet flags
- [ ] GUI drag-and-drop
- [ ] DeepSESR singleton optimization

### 8.2 Medium-term (v1.4.0)

- [ ] Batch processing with patterns
- [ ] Progress ETA in GUI
- [ ] Performance optimizations (Numba)
- [ ] API documentation (mkdocs)

### 8.3 Long-term (v2.0.0 - Rust)

See `docs/RUST_MIGRATION_PLAN.md` for full details.

- [ ] Rust core library with Python bindings
- [ ] Tauri desktop application
- [ ] Native performance (2-5x faster)
- [ ] Smaller binaries (~50MB vs ~100MB)
- [ ] macOS support

---

## Implementation Checklist

### Immediate (This Sprint)

- [x] Fix CI/CD workflow_call issue
- [ ] Add video processing tests
- [ ] Wire up custom exceptions
- [ ] Increase test coverage to 70%

### Next Sprint

- [ ] CLI enhancements (verbose, quiet, dry-run)
- [ ] DeepSESR singleton pattern
- [ ] GUI drag-and-drop
- [ ] README badges and documentation

### Following Sprint

- [ ] PyPI trusted publishing setup
- [ ] 80% test coverage achieved
- [ ] Performance profiling and optimization
- [ ] v1.3.0 release

---

## Success Metrics

| Metric | Current | v1.3.0 Target | v2.0.0 Target |
|--------|---------|---------------|---------------|
| Test Coverage | 56% | 80% | 90% |
| Build Time | ~5 min | ~5 min | ~3 min (Rust) |
| Binary Size (Full) | ~100MB | ~100MB | ~50MB |
| Image Processing | ~2s/4K | ~2s/4K | ~0.5s/4K |
| Video Processing | ~30 fps | ~30 fps | ~60 fps |
| Startup Time | ~3s | ~2s | <1s |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TensorFlow breaking changes | Medium | High | Pin versions, test matrix |
| macOS CI failures | High | Low | Skip macOS in CI |
| PyPI publication issues | Medium | Medium | Test with TestPyPI first |
| Rust migration complexity | Medium | High | Incremental approach, Python bindings |
| Video codec issues | Low | Medium | Document supported formats |

---

## Notes

- macOS support is blocked by TensorFlow-CPU incompatibility
- GUI tests are intentionally low priority (PySimpleGUI is stable)
- Rust migration is a separate project, not blocking Python improvements
- Focus on test coverage before new features
