"""Command line interface for dive color corrector."""

import argparse
import sys
from pathlib import Path

from dive_color_corrector.core.processing.image import correct_image
from dive_color_corrector.core.processing.video import analyze_video, process_video


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Correct colors in underwater images and videos")
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

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    if not args.mode:
        print("Error: Please specify a mode (image or video)", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "image":
            correct_image(str(input_path), str(output_path), use_deep=args.use_deep)
            print(f"Successfully processed image: {output_path}")
        else:  # video mode
            # Two-step video processing: analyze then process
            video_data = None
            for item in analyze_video(str(input_path), str(output_path)):
                if isinstance(item, dict):
                    video_data = item

            if video_data is None:
                print("Error: Failed to analyze video", file=sys.stderr)
                sys.exit(1)

            # Process video with precomputed filter matrices
            for _ in process_video(video_data, yield_preview=False, use_deep=args.use_deep):
                pass

            print(f"\nSuccessfully processed video: {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)
