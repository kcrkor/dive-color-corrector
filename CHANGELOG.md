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
- Expanded test suite with fixtures
- CI workflow for linting and testing
- Release workflow for PyPI and GitHub Releases

### Changed
- Reorganized project directory structure
- Enhanced pyproject.toml with all dependencies and tool configurations
- Moved utility scripts to dedicated directories
- Improved .gitignore coverage

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
