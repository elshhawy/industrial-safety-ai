"""Optional MLflow helpers. Import mlflow only when a run is started."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install project dependencies from "
            "pyproject.toml (see training/README.md)."
        ) from exc
    return mlflow


def configured_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


def configured_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "industrial-safety-yolov9e")


def start_run(
    *,
    run_name: str,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    tags: Mapping[str, str] | None = None,
):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow = load_mlflow()
    mlflow.set_tracking_uri(tracking_uri or configured_tracking_uri())
    mlflow.set_experiment(experiment_name or configured_experiment_name())
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(dict(tags))
    return mlflow, run


def log_params(mlflow: Any, params: Mapping[str, Any]) -> None:
    payload = {}
    for key, value in params.items():
        if value is None:
            payload[key] = "null"
        elif isinstance(value, (str, int, float, bool)):
            payload[key] = value
        else:
            payload[key] = json.dumps(value, default=str)
    if payload:
        mlflow.log_params(payload)


def sanitize_metric_name(name: str) -> str:
    return (
        str(name).replace("(", "_").replace(")", "").replace("[", "_").replace("]", "")
    )


def log_metrics(mlflow: Any, metrics: Mapping[str, Any]) -> None:
    numeric = {}
    for key, value in metrics.items():
        try:
            clean_key = sanitize_metric_name(str(key))
            numeric[clean_key] = float(value)
        except (TypeError, ValueError):
            continue
    if numeric:
        mlflow.log_metrics(numeric)


def log_json_artifact(mlflow: Any, payload: Mapping[str, Any], filename: str) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))
