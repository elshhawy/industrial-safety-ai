"""Validate a trained YOLOv9e checkpoint on the filtered SH17 val split."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, "") and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.mlflow_logging import (  # noqa: E402
    log_json_artifact,
    log_metrics,
    log_params,
)
from training.taxonomy import TARGET_CLASS_NAMES  # noqa: E402
from training.train import (  # noqa: E402
    ensure_accelerator,
    require_mlflow_run,
)
from training.train import (
    resolve_device as _resolve_device,
)

DEFAULT_DATA_YAML = _PROJECT_ROOT / "data" / "filtered_sh17" / "data.yaml"
DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT / "runs" / "detect" / "yolov9e_sh17" / "weights" / "best.pt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLOv9e on the filtered SH17 validation split."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow validation on CPU. Required if no CUDA device is available.",
    )
    parser.add_argument(
        "--project", type=Path, default=_PROJECT_ROOT / "runs" / "detect"
    )
    parser.add_argument("--name", default="yolov9e_sh17_val")
    parser.add_argument(
        "--training-run-id", default=None, help="MLflow run id to link."
    )
    parser.add_argument("--mlflow-experiment", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args(argv)


def resolve_device(device: str) -> str:
    return _resolve_device(device)


def verify_model_class_names(names: Any) -> None:
    if isinstance(names, dict):
        resolved = [
            names.get(i, names.get(str(i))) for i in range(len(TARGET_CLASS_NAMES))
        ]
    elif isinstance(names, (list, tuple)):
        resolved = list(names)[: len(TARGET_CLASS_NAMES)]
    else:
        raise ValueError(f"Unsupported model.names type: {type(names)}")
    if resolved != TARGET_CLASS_NAMES:
        raise ValueError(
            "Checkpoint class names do not match the six-class SH17 taxonomy. "
            f"expected={TARGET_CLASS_NAMES} actual={resolved}"
        )


def validation_kwargs(args: argparse.Namespace, device: str) -> dict[str, Any]:
    if args.split != "val":
        raise ValueError(
            "Validation must use split='val' (the filtered validation subset). "
            f"Got {args.split!r}."
        )
    return {
        "data": str(args.data),
        "split": "val",
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": device,
        "plots": True,
        "project": str(args.project),
        "name": args.name,
    }


def _metric_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in list(values)]


def extract_box_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", metrics)
    report: dict[str, Any] = {}
    mapping = {
        "precision": ("mp",),
        "recall": ("mr",),
        "map50": ("map50",),
        "map50_95": ("map",),
    }
    for out_name, attrs in mapping.items():
        for attr in attrs:
            value = getattr(box, attr, None)
            if value is not None:
                report[out_name] = float(value)
                break
    per_class = {
        "precision": _metric_list(getattr(box, "p", None)),
        "recall": _metric_list(getattr(box, "r", None)),
        "map50": _metric_list(getattr(box, "ap50", None)),
    }
    report["per_class"] = {
        TARGET_CLASS_NAMES[index]: {
            key: values[index]
            for key, values in per_class.items()
            if index < len(values)
        }
        for index in range(len(TARGET_CLASS_NAMES))
    }
    n_images = getattr(metrics, "seen", None)
    if n_images is None:
        n_images = getattr(box, "nt_per_image", None)
    if n_images is not None:
        try:
            report["evaluated_images"] = int(n_images)
        except (TypeError, ValueError):
            report["evaluated_images"] = str(n_images)
    return report


def flatten_class_metrics(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for class_name, values in per_class.items():
        for metric_name, value in values.items():
            flat[f"{metric_name}_{class_name}"] = float(value)
    return flat


def print_report(report: dict[str, Any], model: Path, data: Path) -> None:
    print("=" * 50)
    print("VALIDATION METRICS")
    print("=" * 50)
    print(f"checkpoint: {model}")
    print(f"data: {data}")
    print("split: val")
    for key in ("precision", "recall", "map50", "map50_95", "evaluated_images"):
        if key in report:
            print(f"{key}: {report[key]}")
    print("=" * 50)
    print("PER-CLASS METRICS")
    print("=" * 50)
    for name, values in report.get("per_class", {}).items():
        print(f"{name}: {values}")


def main(argv: list[str] | None = None, model_factory: Any = None) -> int:
    args = parse_args(argv)
    try:
        if not args.model.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.model}. Train first or pass --model."
            )
        if not args.data.is_file():
            raise FileNotFoundError(
                f"Dataset YAML not found: {args.data}. "
                "Run training/prepare_dataset.py first."
            )
        device = resolve_device(args.device)
        ensure_accelerator(device, args.allow_cpu)
        kwargs = validation_kwargs(args, device)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mlflow = None
    run = None
    try:
        tags = {"stage": "val", "evaluated_split": "val"}
        if args.training_run_id:
            tags["training_run_id"] = args.training_run_id
        mlflow, run = require_mlflow_run(
            run_name=args.name,
            experiment_name=args.mlflow_experiment,
            tracking_uri=args.mlflow_tracking_uri,
            tags=tags,
        )
        log_params(
            mlflow,
            {
                "checkpoint": str(args.model),
                "data": str(args.data),
                "split": "val",
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "device": device,
                "training_run_id": args.training_run_id,
                "target_classes": TARGET_CLASS_NAMES,
            },
        )

        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO

        model = model_factory(str(args.model))
        verify_model_class_names(getattr(model, "names", {}))
        metrics = model.val(**kwargs)
        report = extract_box_metrics(metrics)
        print_report(report, args.model, args.data)
        flat_metrics = {
            key: report[key]
            for key in ("precision", "recall", "map50", "map50_95")
            if key in report
        }
        flat_metrics.update(flatten_class_metrics(report.get("per_class", {})))
        log_metrics(mlflow, flat_metrics)
        log_json_artifact(mlflow, report, "validation_report.json")
        print(f"MLflow run id: {run.info.run_id}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if mlflow is not None and run is not None:
            mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
