"""Start MLflow runs the way the team agreed (see docs/CONVENTIONS.md).

mlflow and dagshub are imported only when they are needed, so this module can be
imported and tested without them (CI installs only the dev extra).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from industrial_safety.mlops.mlflow_utils import (
    build_optional_tags,
    build_required_tags,
    build_run_name,
    sanitize_params,
    validate_tags,
)

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
DAGSHUB_REPO_ENV = "DAGSHUB_REPO"

_EXPERIMENT_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_INSTALL_HINT = 'Install it with: pip install -e ".[tracking]"'


class TrackingNotConfiguredError(RuntimeError):
    """Raised when no tracking server was configured."""


def validate_experiment_name(name: str) -> str:
    """Return the name if it follows the convention, else raise ValueError."""
    if not _EXPERIMENT_NAME.match(name):
        raise ValueError(
            f"Invalid experiment name {name!r}. Use lowercase words joined by "
            "hyphens, for example 'ppe-detection'."
        )
    return name


def _load_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(f"mlflow is not installed. {_INSTALL_HINT}") from exc
    return mlflow


def _load_dagshub() -> Any:
    try:
        import dagshub
    except ImportError as exc:
        raise RuntimeError(f"dagshub is not installed. {_INSTALL_HINT}") from exc
    return dagshub


def connect_tracking(
    repo: str | None = None,
    tracking_uri: str | None = None,
    *,
    mlflow_module: Any | None = None,
    dagshub_module: Any | None = None,
) -> str:
    """Point MLflow at a tracking server and return the URI in use.

    Resolution order:
    - tracking_uri argument or env MLFLOW_TRACKING_URI: used as is.
    - repo argument or env DAGSHUB_REPO (`owner/name`): connect through
      `dagshub.init`. The DagsHub DVC remote is never touched.
    - neither: raise TrackingNotConfiguredError. This is on purpose: without a
      tracking URI, MLflow 3.x silently writes a local mlflow.db.
    """
    uri = tracking_uri or os.environ.get(TRACKING_URI_ENV)
    repo_name = repo or os.environ.get(DAGSHUB_REPO_ENV)
    if not uri and not repo_name:
        raise TrackingNotConfiguredError(
            f"No tracking server configured. Set {DAGSHUB_REPO_ENV}=<owner>/<repo> "
            f"or {TRACKING_URI_ENV}. Refusing to fall back to a local mlflow.db."
        )
    if not uri and not _REPO.match(repo_name or ""):
        raise ValueError(f"{DAGSHUB_REPO_ENV} must look like 'owner/name'.")
    mlflow = mlflow_module or _load_mlflow()
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        owner, name = str(repo_name).split("/", 1)
        dagshub = dagshub_module or _load_dagshub()
        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True, dvc=False)
    return str(mlflow.get_tracking_uri())


def start_tracked_run(
    experiment: str,
    model: str,
    params: Mapping[str, Any] | None = None,
    *,
    author: str | None = None,
    git_commit: str | None = None,
    dataset_version: str | None = None,
    dataset_dvc_file: Path | None = None,
    extra_tags: Mapping[str, str] | None = None,
    run_name: str | None = None,
    repo: str | None = None,
    tracking_uri: str | None = None,
    cwd: Path | None = None,
    day: date | None = None,
    mlflow_module: Any | None = None,
    dagshub_module: Any | None = None,
) -> Any:
    """Start an MLflow run with the mandatory tags and safe params.

    Use it as a context manager:

        with start_tracked_run("ppe-detection", "yolov9e", {"epochs": 50}):
            mlflow.log_metric("val_map50", 0.5)

    The checks that need no network run first. The run is refused before any
    connection if the experiment name is invalid or a mandatory tag is missing.
    Mandatory tags cannot be overridden through extra_tags. No artifact (weights,
    plots) is uploaded by this function.
    """
    validate_experiment_name(experiment)
    required = validate_tags(
        build_required_tags(
            author=author,
            git_commit=git_commit,
            dataset_version=dataset_version,
            dataset_dvc_file=dataset_dvc_file,
            cwd=cwd,
        )
    )
    name = run_name or build_run_name(required["author"], model, day)
    mlflow = mlflow_module or _load_mlflow()
    connect_tracking(
        repo, tracking_uri, mlflow_module=mlflow, dagshub_module=dagshub_module
    )
    tags = {
        **build_optional_tags(cwd),
        **{key: str(value) for key, value in (extra_tags or {}).items()},
        **required,
    }
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=name)
    try:
        mlflow.set_tags(tags)
        mlflow.log_params(sanitize_params({**(params or {}), "model": model}))
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
    return run
