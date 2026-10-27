"""Quick inference test for the pretrained YOLOv9e safety detection model."""

import argparse
from pathlib import Path

from ultralytics import YOLO

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run safety detection inference on an image."
    )
    parser.add_argument(
        "--image", type=str, default="test.jpg", help="Path to the test image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "yolo9e.pt"),
        help="Path to the YOLO model weights",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(model_path))
    print(f"Model classes ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    results = model(str(image_path), conf=args.conf)
    results[0].show()


if __name__ == "__main__":
    main()
