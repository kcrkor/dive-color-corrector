# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete improvement plan documentation

### Changed
- Updated production plan with current status

## [1.2.1] - 2026-01-02

### Fixed
- CI/CD workflow_call triggers for reusable workflows (ci.yml, build.yml)
- Release workflow can now properly call build and test workflows

### Added
- Production readiness plan documentation
- Custom exceptions module
- Logging configuration module
- Pre-commit hooks configuration
- Expanded test suite with fixtures
- CI workflow for linting and testing
- Release workflow for PyPI and GitHub Releases
- Deep SESR ONNX model integration
- Batch processing mode for CLI
- Pydantic validation for configuration

### Changed
- Reorganized project directory structure
- Enhanced pyproject.toml with all dependencies and tool configurations
- Moved utility scripts to dedicated directories
- Improved .gitignore coverage
- Migrated from TensorFlow/Keras to ONNX Runtime for model inference

### Fixed
- Circular import in correction.py module
- Missing pillow dependency
- tensorflow-io-gcs-filesystem excluded on Windows (no wheels available)

### Removed
- Redundant configuration files (hatch.toml, requirements.txt)
- Empty output directory
- Orphaned spec files

## [1.2.0] - 2026-01-01

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
