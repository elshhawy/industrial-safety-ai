"""GPU-free, network-free tests for the mandatory MLflow tag helpers."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from industrial_safety.mlops.mlflow_utils import (
    REQUIRED_TAGS,
    MissingTagsError,
    build_required_tags,
    get_git_commit,
    missing_tags,
    read_dvc_md5,
    validate_tags,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

FULL = {"author": "ahmed", "git_commit": "abc123", "dataset_version": "md5:1"}


def test_validate_accepts_complete_tags():
    assert validate_tags(FULL) == FULL


@pytest.mark.parametrize("name", REQUIRED_TAGS)
def test_validate_rejects_each_missing_tag(name):
    tags = {**FULL, name: None}
    assert missing_tags(tags) == [name]
    with pytest.raises(MissingTagsError, match=name):
        validate_tags(tags)


def test_blank_values_count_as_missing():
    assert missing_tags({**FULL, "author": "   "}) == ["author"]


def test_validate_lists_all_missing_tags_at_once():
    with pytest.raises(MissingTagsError) as exc:
        validate_tags({})
    for name in REQUIRED_TAGS:
        assert name in str(exc.value)


def test_explicit_arguments_override_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_AUTHOR", "from-env")
    monkeypatch.setenv("DATASET_VERSION", "env-version")
    tags = build_required_tags(
        author="arg", git_commit="deadbeef", dataset_version="arg-version"
    )
    assert tags == {
        "author": "arg",
        "git_commit": "deadbeef",
        "dataset_version": "arg-version",
    }


def test_environment_is_used_when_arguments_are_missing(monkeypatch):
    monkeypatch.setenv("MLFLOW_AUTHOR", "from-env")
    monkeypatch.setenv("DATASET_VERSION", "env-version")
    tags = build_required_tags(git_commit="deadbeef")
    assert tags["author"] == "from-env"
    assert tags["dataset_version"] == "env-version"


def test_dataset_version_from_dvc_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DATASET_VERSION", raising=False)
    dvc_file = tmp_path / "data.dvc"
    dvc_file.write_text(
        "outs:\n- md5: 5ad47e048b65ca2bfde4e55282bfd3cf.dir\n  path: data\n",
        encoding="utf-8",
    )
    assert read_dvc_md5(dvc_file) == "5ad47e048b65ca2bfde4e55282bfd3cf.dir"
    tags = build_required_tags(author="a", git_commit="c", dataset_dvc_file=dvc_file)
    assert tags["dataset_version"] == "5ad47e048b65ca2bfde4e55282bfd3cf.dir"


def test_missing_dvc_file_gives_none(tmp_path):
    assert read_dvc_md5(tmp_path / "nope.dvc") is None


@needs_git
def test_git_commit_inside_and_outside_a_repo(tmp_path):
    assert get_git_commit(tmp_path) is None
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run([*git, "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    commit = get_git_commit(tmp_path)
    assert commit is not None and len(commit) == 40
