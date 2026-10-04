from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "yolov8n.pt"
    data_path = project_root / "training" / "data.yaml"

    model = YOLO(model_path)

    model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640
    )


if __name__ == "__main__":
    main()