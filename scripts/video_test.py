"""Run the trained six-class PPE detector on a video file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.taxonomy import TARGET_CLASS_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_ckpt = (
        project_root / "runs" / "detect" / "yolov9e_sh17" / "weights" / "best.pt"
    )
    parser = argparse.ArgumentParser(
        description="Visualize SH17 PPE detections on video."
    )
    parser.add_argument("--video", default="test.mp4")
    parser.add_argument("--model", default=str(default_ckpt))
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "video_test.py requires opencv-python and ultralytics. "
            "Install project dependencies from pyproject.toml."
        ) from exc

    args = parse_args()
    model = YOLO(args.model)

    expected_names = set(TARGET_CLASS_NAMES)
    actual_names = set(model.names.values())
    if not expected_names.issubset(actual_names):
        print(
            f"Warning: Model classes {actual_names} "
            f"do not match target classes {expected_names}"
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video {args.video} -> {args.output}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz)[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.imshow("PPE detection", frame)
        out.write(frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished. Output saved as {args.output}")


if __name__ == "__main__":
    main()
