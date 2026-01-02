"""Command line interface for dive color corrector."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dive_color_corrector.core.processing.image import correct_image
from dive_color_corrector.core.processing.video import analyze_video, process_video
from dive_color_corrector.logging import get_logger, setup_logging


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Correct colors in underwater images and videos")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    # Image processing
    image_parser = subparsers.add_parser("image", help="Process a single image")
    image_parser.add_argument("input", type=str, help="Input image path")
    image_parser.add_argument("output", type=str, help="Output image path")
    image_parser.add_argument("--use-deep", action="store_true", help="Use deep learning model")

    # Video processing
    video_parser = subparsers.add_parser("video", help="Process a video")
    video_parser.add_argument("input", type=str, help="Input video path")
    video_parser.add_argument("output", type=str, help="Output video path")
    video_parser.add_argument("--use-deep", action="store_true", help="Use deep learning model")

    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for CLI."""
    args = parse_args(argv)

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger()

    if not args.mode:
        logger.error("Please specify a mode (image or video)")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "image":
            correct_image(str(input_path), str(output_path), use_deep=args.use_deep)
            logger.info(f"Successfully processed image: {output_path}")
        else:  # video mode
            video_data = None
            for item in analyze_video(str(input_path), str(output_path)):
                if isinstance(item, dict):
                    video_data = item
                elif isinstance(item, int) and item % 10 == 0:
                    logger.debug(f"Analyzing video: {item}%")

            if video_data is None:
                logger.error("Failed to analyze video")
                sys.exit(1)

            # Process video with precomputed filter matrices
            for progress, _ in process_video(
                video_data, yield_preview=False, use_deep=args.use_deep
            ):
                if int(progress) % 10 == 0:
                    logger.debug(f"Processing video: {int(progress)}%")

            logger.info(f"Successfully processed video: {output_path}")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
