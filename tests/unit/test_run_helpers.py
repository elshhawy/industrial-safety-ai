"""Network-free tests for run naming, param sanitizing and git_dirty."""

from __future__ import annotations

import shutil
import subprocess
from datetime import date

import pytest

from industrial_safety.mlops.mlflow_utils import (
    REDACTED_PATH,
    build_optional_tags,
    build_run_name,
    is_git_dirty,
    sanitize_params,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def test_run_name_follows_convention():
    name = build_run_name("ahmed", "yolov9e", date(2026, 10, 21))
    assert name == "ahmed_yolov9e_2026-10-21"


def test_run_name_replaces_unsafe_characters():
    name = build_run_name(" A7med 004 ", "yolo v9/e", date(2026, 1, 2))
    assert name == "A7med-004_yolo-v9-e_2026-01-02"


@pytest.mark.parametrize(("author", "model"), [("", "m"), ("a", "  "), ("///", "m")])
def test_run_name_rejects_empty_parts(author, model):
    with pytest.raises(ValueError):
        build_run_name(author, model, date(2026, 1, 1))


def test_run_name_defaults_to_today():
    assert build_run_name("a", "m").endswith(date.today().isoformat())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("F:\\dev\\industrial-safety-ai\\data.yaml", "data.yaml"),
        ("C:/Users/someone/weights/best.pt", "best.pt"),
        ("\\\\server\\share\\clip.mp4", "clip.mp4"),
        ("/home/someone/project/data.yaml", "data.yaml"),
        ("~/runs/exp1", "exp1"),
    ],
)
def test_absolute_paths_are_reduced_to_file_name(value, expected):
    assert sanitize_params({"p": value}) == {"p": expected}


@pytest.mark.parametrize(
    "value",
    ["yolov9e.pt", "configs/train.yaml", "https://dagshub.com/a/b", "640", "/"],
)
def test_other_strings_are_left_alone(value):
    assert sanitize_params({"p": value}) == {"p": value}


def test_root_like_path_without_name_is_redacted():
    assert sanitize_params({"p": "C:\\"}) == {"p": REDACTED_PATH}


def test_non_string_values_become_strings():
    params = {"a": None, "b": True, "c": 50, "d": 0.5, "e": ["x", "y"]}
    assert sanitize_params(params) == {
        "a": "null",
        "b": "True",
        "c": "50",
        "d": "0.5",
        "e": '["x", "y"]',
    }


def _git(tmp_path, *args):
    base = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*base, *args], cwd=tmp_path, check=True, capture_output=True)


@needs_git
def test_git_dirty_states(tmp_path):
    assert is_git_dirty(tmp_path) is None
    assert build_optional_tags(tmp_path) == {}
    _git(tmp_path, "init", "-q")
    tracked = tmp_path / "f.txt"
    tracked.write_text("one", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    assert is_git_dirty(tmp_path) is False
    assert build_optional_tags(tmp_path) == {"git_dirty": "false"}
    (tmp_path / "new.txt").write_text("untracked", encoding="utf-8")
    assert is_git_dirty(tmp_path) is False
    tracked.write_text("two", encoding="utf-8")
    assert is_git_dirty(tmp_path) is True
    assert build_optional_tags(tmp_path) == {"git_dirty": "true"}
