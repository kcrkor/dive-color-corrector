"""Command line interface for dive color corrector."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dive_color_corrector.core.models import SESR_AVAILABLE
from dive_color_corrector.core.processing.image import correct_image
from dive_color_corrector.core.processing.video import analyze_video, process_video
from dive_color_corrector.core.schemas import VideoData
from dive_color_corrector.core.utils.constants import IMAGE_FORMATS, VIDEO_FORMATS
from dive_color_corrector.logging import get_logger, setup_logging

IMAGE_EXTENSIONS = set(IMAGE_FORMATS.keys())
VIDEO_EXTENSIONS = set(VIDEO_FORMATS.keys())


def _add_sesr_arg(parser: argparse.ArgumentParser) -> None:
    sesr_help = "Use Deep SESR neural network"
    if not SESR_AVAILABLE:
        sesr_help += " (unavailable - install with: pip install dive_color_corrector[sesr])"
    parser.add_argument("--sesr", action="store_true", help=sesr_help)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct colors in underwater images and videos")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    image_parser = subparsers.add_parser("image", help="Process a single image")
    image_parser.add_argument("input", type=str, help="Input image path")
    image_parser.add_argument("output", type=str, help="Output image path")
    _add_sesr_arg(image_parser)

    video_parser = subparsers.add_parser("video", help="Process a video")
    video_parser.add_argument("input", type=str, help="Input video path")
    video_parser.add_argument("output", type=str, help="Output video path")
    _add_sesr_arg(video_parser)

    batch_parser = subparsers.add_parser("batch", help="Process all files in a directory")
    batch_parser.add_argument("input_dir", type=str, help="Input directory path")
    batch_parser.add_argument("output_dir", type=str, help="Output directory path")
    _add_sesr_arg(batch_parser)
    batch_parser.add_argument(
        "--images-only", action="store_true", help="Process only images (skip videos)"
    )
    batch_parser.add_argument(
        "--videos-only", action="store_true", help="Process only videos (skip images)"
    )
    batch_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed without processing"
    )

    return parser.parse_args(args)


def process_single_image(input_path: Path, output_path: Path, use_deep: bool, logger: Any) -> bool:
    try:
        correct_image(str(input_path), str(output_path), use_deep=use_deep)
        logger.info(f"Processed: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed {input_path.name}: {e}")
        return False


def process_single_video(input_path: Path, output_path: Path, use_deep: bool, logger: Any) -> bool:
    try:
        video_data = None
        for item in analyze_video(str(input_path), str(output_path)):
            if isinstance(item, VideoData):
                video_data = item

        if video_data is None:
            logger.error(f"Failed to analyze: {input_path.name}")
            return False

        for progress, _ in process_video(video_data, yield_preview=False, use_deep=use_deep):
            if int(progress) % 25 == 0:
                logger.debug(f"{input_path.name}: {int(progress)}%")

        logger.info(f"Processed: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed {input_path.name}: {e}")
        return False


def process_batch(
    input_dir: Path,
    output_dir: Path,
    use_deep: bool,
    images_only: bool,
    videos_only: bool,
    dry_run: bool,
) -> None:
    logger = get_logger()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.iterdir())
    image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
    video_files = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]

    if videos_only:
        image_files = []
    if images_only:
        video_files = []

    total = len(image_files) + len(video_files)
    logger.info(f"Found {len(image_files)} images, {len(video_files)} videos")

    success, failed = 0, 0

    for i, img in enumerate(image_files, 1):
        out_path = output_dir / f"{img.stem}_corrected{img.suffix.lower()}"
        logger.info(f"[{i}/{total}] {img.name}")
        if not dry_run:
            if process_single_image(img, out_path, use_deep, logger):
                success += 1
            else:
                failed += 1
        else:
            success += 1

    for i, vid in enumerate(video_files, len(image_files) + 1):
        out_path = output_dir / f"{vid.stem}_corrected.mp4"
        logger.info(f"[{i}/{total}] {vid.name}")
        if not dry_run:
            if process_single_video(vid, out_path, use_deep, logger):
                success += 1
            else:
                failed += 1
        else:
            success += 1

    logger.info(f"Complete: {success} succeeded, {failed} failed")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verbose and args.quiet:
        print("Error: Cannot specify both --verbose and --quiet")
        sys.exit(1)

    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    setup_logging(level=log_level)
    logger = get_logger()

    if not args.mode:
        logger.error("Please specify a mode: image, video, or batch")
        sys.exit(1)

    if getattr(args, "sesr", False) and not SESR_AVAILABLE:
        logger.error("SESR not available. Install with: pip install dive_color_corrector[sesr]")
        sys.exit(1)

    if args.mode == "batch":
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

        process_batch(
            input_dir,
            output_dir,
            args.sesr,
            args.images_only,
            args.videos_only,
            getattr(args, "dry_run", False),
        )
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist")
        sys.exit(1)

    if not getattr(args, "dry_run", False):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "image":
        if not process_single_image(input_path, output_path, args.sesr, logger):
            sys.exit(1)
    else:
        if not process_single_video(input_path, output_path, args.sesr, logger):
            sys.exit(1)


if __name__ == "__main__":
    main()
