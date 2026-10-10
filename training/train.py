"""Train YOLOv9e on the filtered six-class SH17 dataset."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, "") and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.mlflow_logging import (  # noqa: E402
    log_json_artifact,
    log_metrics,
    log_params,
    start_run,
)
from training.taxonomy import TARGET_CLASS_NAMES, class_mapping_summary  # noqa: E402

DEFAULT_DATA_YAML = _PROJECT_ROOT / "data" / "filtered_sh17" / "data.yaml"
DEFAULT_MODEL = "yolov9e.pt"
DEFAULT_PROJECT = _PROJECT_ROOT / "runs" / "detect"
MLFLOW_INSTALL_HINT = (
    "MLflow is required for this project and cannot be skipped. Install it from "
    'pyproject.toml (`python -m pip install -e ".[dev]"` after a CUDA PyTorch '
    "install) and optionally set MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME."
)


def default_worker_count() -> int:
    return 2 if os.name == "nt" else 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv9e on filtered SH17.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="YOLOv9e checkpoint or yaml."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--lr0", type=float, default=None, help="Initial learning rate."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, 0, or a CUDA device string.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow YOLOv9e to run on CPU. Required if no CUDA device is available.",
    )
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolov9e_sh17")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--mlflow-experiment", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args(argv)


def assert_yolov9e(model_ref: str) -> str:
    stem = Path(model_ref).stem.lower().replace("_", "").replace("-", "")
    if "yolov9e" not in stem and "yolo9e" not in stem:
        raise ValueError(
            f"This workflow trains YOLOv9e only. Refusing model '{model_ref}'. "
            "Pass --model yolov9e.pt (Ultralytics will download it when you train) "
            "or a local YOLOv9e checkpoint."
        )
    return model_ref


def resolve_device(device: str) -> str:
    requested = device.strip().lower()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install a CUDA build from https://pytorch.org "
            "before installing Ultralytics if you have an NVIDIA GPU."
        ) from exc
    cuda_available = torch.cuda.is_available()
    if requested in {"auto", ""}:
        return "0" if cuda_available else "cpu"
    wants_cuda = requested not in {"cpu"}
    if wants_cuda and not cuda_available:
        raise RuntimeError(
            f"CUDA device '{device}' was requested but torch.cuda.is_available() "
            f"is False (torch={torch.__version__}). Install a CUDA build of PyTorch. "
            "Use --device cpu only for debugging, not for YOLOv9e training."
        )
    return device


def ensure_accelerator(device: str, allow_cpu: bool) -> None:
    if str(device).strip().lower() in {"cpu", "cpu:0"} and not allow_cpu:
        raise RuntimeError(
            "YOLOv9e would run on CPU. Install a CUDA build of PyTorch from "
            "https://pytorch.org, confirm torch.cuda.is_available() is True, "
            "and pass --device 0. To override (slow, not recommended) pass "
            "--allow-cpu."
        )


def require_mlflow_run(
    *,
    run_name: str,
    experiment_name: str | None,
    tracking_uri: str | None,
    tags: dict[str, str],
):
    try:
        return start_run(
            run_name=run_name,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=tags,
        )
    except Exception as exc:
        raise RuntimeError(f"{MLFLOW_INSTALL_HINT} Details: {exc}") from exc


def load_preparation_stats(data_yaml: Path) -> dict[str, Any] | None:
    stats_path = data_yaml.parent / "preparation_stats.json"
    if not stats_path.is_file():
        return None
    import json

    return json.loads(stats_path.read_text(encoding="utf-8"))


def build_train_kwargs(args: argparse.Namespace, device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "device": device,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": True,
    }
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0
    return kwargs


def extract_metrics(results: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    maps = getattr(results, "maps", None)
    if maps is None:
        maps = getattr(getattr(results, "box", None), "maps", None)
    if maps is not None:
        values = list(maps.tolist() if hasattr(maps, "tolist") else maps)
        for index, name in enumerate(TARGET_CLASS_NAMES):
            if index < len(values):
                try:
                    metrics[f"map50_95_{name}"] = float(values[index])
                except (TypeError, ValueError):
                    continue
    return metrics


def train_model(model: Any, train_kwargs: dict[str, Any]) -> Any:
    return model.train(**train_kwargs)


def main(argv: list[str] | None = None, model_factory: Any = None) -> int:
    args = parse_args(argv)
    try:
        model_ref = assert_yolov9e(args.model)
        if not args.data.is_file():
            raise FileNotFoundError(
                f"Dataset YAML not found: {args.data}. "
                "Run training/prepare_dataset.py first."
            )
        device = resolve_device(args.device)
        ensure_accelerator(device, args.allow_cpu)
        train_kwargs = build_train_kwargs(args, device)
        prep_stats = load_preparation_stats(args.data)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    mlflow = None
    run = None
    try:
        mlflow, run = require_mlflow_run(
            run_name=args.name,
            experiment_name=args.mlflow_experiment,
            tracking_uri=args.mlflow_tracking_uri,
            tags={
                "stage": "train",
                "architecture": "YOLOv9e",
                "dataset": "filtered_sh17",
            },
        )
        mapping_artifact = {
            "class_mapping_summary": class_mapping_summary(),
            "target_class_names": list(TARGET_CLASS_NAMES),
            "preparation_stats": prep_stats,
        }
        log_params(
            mlflow,
            {
                "model": model_ref,
                "architecture": "YOLOv9e",
                "data": str(args.data),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "lr0": args.lr0,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "device": device,
                "workers": args.workers,
                "train_val_ratio": None
                if prep_stats is None
                else prep_stats.get("train_ratio"),
                "target_classes": TARGET_CLASS_NAMES,
                "class_mapping_summary": class_mapping_summary(),
                "class_mapping_artifact": "class_mapping.json",
            },
        )
        log_json_artifact(mlflow, mapping_artifact, "class_mapping.json")
        if prep_stats is not None:
            log_json_artifact(mlflow, prep_stats, "preparation_stats.json")

        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO

        model = model_factory(model_ref)
        results = train_model(model, train_kwargs)
        duration_s = time.perf_counter() - started
        metrics = extract_metrics(results)
        save_dir = getattr(results, "save_dir", None) or (args.project / args.name)
        print(f"Training complete. Outputs: {save_dir}")
        print("Expected checkpoints: best.pt and last.pt under the run directory.")
        print(
            "Identical results are not guaranteed across different hardware or "
            "software versions even with a fixed seed."
        )
        log_metrics(mlflow, {**metrics, "training_duration_seconds": duration_s})
        log_params(
            mlflow, {"save_dir": str(save_dir), "mlflow_run_id": run.info.run_id}
        )
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