"""Helpers for consistent MLflow tracking.

This module only builds and validates the mandatory run tags. It does not
import mlflow and does not talk to any server, so it is safe to test in CI.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REQUIRED_TAGS: tuple[str, ...] = ("author", "git_commit", "dataset_version")

_MD5_LINE = re.compile(r"^\s*(?:-\s*)?md5:\s*([0-9a-fA-F]+(?:\.dir)?)\s*$")
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
REDACTED_PATH = "<redacted-path>"


class MissingTagsError(ValueError):
    """Raised when one or more mandatory tags have no value."""


def _run_git(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a git command. Return stripped stdout ("" if empty), None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def get_git_commit(cwd: Path | None = None) -> str | None:
    """Return the full hash of HEAD, or None outside a git repository."""
    return _run_git(["rev-parse", "HEAD"], cwd) or None


def get_git_author(cwd: Path | None = None) -> str | None:
    """Return the configured git user.name, or None if it is not set."""
    return _run_git(["config", "user.name"], cwd) or None


def read_dvc_md5(dvc_file: Path) -> str | None:
    """Return the first md5 found in a DVC pointer file, or None."""
    try:
        text = Path(dvc_file).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        match = _MD5_LINE.match(line)
        if match:
            return match.group(1)
    return None


def build_required_tags(
    author: str | None = None,
    git_commit: str | None = None,
    dataset_version: str | None = None,
    dataset_dvc_file: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, str | None]:
    """Collect the mandatory tags. Missing values are returned as None.

    Resolution order for each tag (first non-empty wins):
    - author: argument, env MLFLOW_AUTHOR, git user.name
    - git_commit: argument, git HEAD
    - dataset_version: argument, env DATASET_VERSION, md5 of dataset_dvc_file
    """
    resolved_author = author or os.environ.get("MLFLOW_AUTHOR") or get_git_author(cwd)
    resolved_commit = git_commit or get_git_commit(cwd)
    resolved_dataset = dataset_version or os.environ.get("DATASET_VERSION")
    if not resolved_dataset and dataset_dvc_file is not None:
        resolved_dataset = read_dvc_md5(dataset_dvc_file)
    return {
        "author": resolved_author or None,
        "git_commit": resolved_commit or None,
        "dataset_version": resolved_dataset or None,
    }


def missing_tags(tags: Mapping[str, str | None]) -> list[str]:
    """Return the mandatory tag names that are absent or empty."""
    return [name for name in REQUIRED_TAGS if not (tags.get(name) or "").strip()]


def validate_tags(tags: Mapping[str, str | None]) -> dict[str, str]:
    """Return the mandatory tags as strings, or raise MissingTagsError."""
    missing = missing_tags(tags)
    if missing:
        raise MissingTagsError(
            "Refusing to start run, missing mandatory tag(s): " + ", ".join(missing)
        )
    return {name: str(tags[name]).strip() for name in REQUIRED_TAGS}


def is_git_dirty(cwd: Path | None = None) -> bool | None:
    """Return True if tracked files have uncommitted changes.

    Untracked files are ignored on purpose. Returns None if git is unavailable
    or the folder is not a repository.
    """
    output = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd)
    if output is None:
        return None
    return bool(output)


def build_optional_tags(cwd: Path | None = None) -> dict[str, str]:
    """Return helpful but non-mandatory tags (currently only git_dirty)."""
    dirty = is_git_dirty(cwd)
    if dirty is None:
        return {}
    return {"git_dirty": "true" if dirty else "false"}


def _slug(value: str) -> str:
    return _UNSAFE_NAME_CHARS.sub("-", value.strip()).strip("-")


def build_run_name(author: str, model: str, day: date | None = None) -> str:
    """Return the run name `<author>_<model>_<yyyy-mm-dd>`."""
    author_part = _slug(author)
    model_part = _slug(model)
    if not author_part or not model_part:
        raise ValueError("author and model must contain letters or digits")
    return f"{author_part}_{model_part}_{(day or date.today()).isoformat()}"


def _is_absolute_path(text: str) -> bool:
    if "://" in text:
        return False
    if _WINDOWS_ABSOLUTE.match(text):
        return True
    return (text.startswith("/") and len(text) > 1) or text.startswith("~/")


def _file_name_only(text: str) -> str:
    if _WINDOWS_ABSOLUTE.match(text):
        name = PureWindowsPath(text).name
    else:
        name = PurePosixPath(text).name
    return name or REDACTED_PATH


def sanitize_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Return params as strings, keeping only the file name of absolute paths.

    Only whole values that look like an absolute path are shortened. Paths nested
    inside lists or dicts are not detected.
    """
    clean: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            text = "null"
        elif isinstance(value, str):
            text = _file_name_only(value) if _is_absolute_path(value) else value
        elif isinstance(value, (bool, int, float)):
            text = str(value)
        else:
            text = json.dumps(value, default=str)
        clean[str(key)] = text
    return clean
