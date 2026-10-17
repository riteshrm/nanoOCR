
import argparse
import json
import sys
from pathlib import Path

import cv2

from pp_engine import OCREngine


def main():
    parser = argparse.ArgumentParser(
        description="OCR transcription and text redaction"
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to input image"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output files (default: current directory)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    # Load image
    image = cv2.imread(str(args.image))

    if image is None:
        parser.error(f"Unable to read image: {args.image}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OCR models
    engine = OCREngine()

    # Run the same engine used by FastAPI
    text_results, redacted_image = engine.process(image)

    # Output paths
    stem = args.image.stem

    json_path = args.output_dir / f"{stem}_text.json"
    image_path = args.output_dir / f"{stem}_redacted.png"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            text_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save redacted image
    if not cv2.imwrite(str(image_path), redacted_image):
        print("Failed to save redacted image", file=sys.stderr)
        sys.exit(1)

    print(f"JSON saved: {json_path}")
    print(f"Redacted image saved: {image_path}")


if __name__ == "__main__":
    main()
