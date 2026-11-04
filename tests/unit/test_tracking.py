"""Tests for connect_tracking and start_tracked_run (no network, fake mlflow)."""

from __future__ import annotations

from datetime import date

import pytest

from industrial_safety.mlops import tracking
from industrial_safety.mlops.mlflow_utils import MissingTagsError
from industrial_safety.mlops.tracking import (
    TrackingNotConfiguredError,
    connect_tracking,
    start_tracked_run,
    validate_experiment_name,
)

DAY = date(2026, 10, 21)
TAGS = {"author": "ahmed", "git_commit": "abc123", "dataset_version": "data-v1"}


class FakeMLflow:
    def __init__(self):
        self.calls = []
        self.uri = "unset"

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))
        self.uri = uri

    def get_tracking_uri(self):
        return self.uri

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name=None):
        self.calls.append(("start_run", run_name))
        return "RUN"

    def set_tags(self, tags):
        self.calls.append(("set_tags", dict(tags)))

    def log_params(self, params):
        self.calls.append(("log_params", dict(params)))

    def end_run(self, status="FINISHED"):
        self.calls.append(("end_run", status))


class FakeDagshub:
    def __init__(self, mlflow):
        self.mlflow = mlflow
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        self.mlflow.uri = "https://dagshub.com/o/r.mlflow"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in (
        "MLFLOW_AUTHOR",
        "DATASET_VERSION",
        "MLFLOW_TRACKING_URI",
        "DAGSHUB_REPO",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "industrial_safety.mlops.mlflow_utils.get_git_author", lambda cwd=None: None
    )
    monkeypatch.setattr(
        "industrial_safety.mlops.mlflow_utils.get_git_commit", lambda cwd=None: None
    )
    monkeypatch.setattr(tracking, "build_optional_tags", lambda cwd=None: {})


def _start(mlflow, **kwargs):
    options = {
        **TAGS,
        "tracking_uri": "http://tracking.example",
        "day": DAY,
        "mlflow_module": mlflow,
    }
    options.update(kwargs)
    return start_tracked_run("ppe-detection", "yolov9e", **options)


def test_experiment_name_convention():
    assert validate_experiment_name("ppe-detection") == "ppe-detection"
    for bad in ("PPE Detection", "ppe_detection", "-ppe", "ppe-", ""):
        with pytest.raises(ValueError):
            validate_experiment_name(bad)


def test_refuses_when_no_server_is_configured():
    fake = FakeMLflow()
    with pytest.raises(TrackingNotConfiguredError, match="local mlflow.db"):
        connect_tracking(mlflow_module=fake)
    assert fake.calls == []


def test_tracking_uri_argument_is_used():
    fake = FakeMLflow()
    assert connect_tracking(tracking_uri="http://x", mlflow_module=fake) == "http://x"
    assert fake.calls == [("set_tracking_uri", "http://x")]


def test_tracking_uri_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env")
    fake = FakeMLflow()
    assert connect_tracking(mlflow_module=fake) == "http://from-env"


def test_dagshub_repo_uses_init_without_dvc():
    fake = FakeMLflow()
    dagshub = FakeDagshub(fake)
    uri = connect_tracking(
        "elshhawy/industrial-safety-ai-tracking",
        None,
        mlflow_module=fake,
        dagshub_module=dagshub,
    )
    assert dagshub.calls == [
        {
            "repo_owner": "elshhawy",
            "repo_name": "industrial-safety-ai-tracking",
            "mlflow": True,
            "dvc": False,
        }
    ]
    assert uri == "https://dagshub.com/o/r.mlflow"
    assert fake.calls == []


def test_dagshub_repo_from_environment(monkeypatch):
    monkeypatch.setenv("DAGSHUB_REPO", "o/r")
    fake = FakeMLflow()
    dagshub = FakeDagshub(fake)
    connect_tracking(mlflow_module=fake, dagshub_module=dagshub)
    assert dagshub.calls[0]["repo_owner"] == "o"


@pytest.mark.parametrize("bad", ["justname", "a/b/c", "a b/c", "/c"])
def test_bad_repo_value_is_rejected_before_dagshub(bad):
    fake = FakeMLflow()
    dagshub = FakeDagshub(fake)
    with pytest.raises(ValueError, match="owner/name"):
        connect_tracking(bad, None, mlflow_module=fake, dagshub_module=dagshub)
    assert dagshub.calls == []


def test_run_is_refused_when_a_tag_is_missing_and_nothing_is_called():
    fake = FakeMLflow()
    with pytest.raises(MissingTagsError, match="author"):
        _start(fake, author=None)
    assert fake.calls == []


def test_invalid_experiment_name_is_refused_first():
    fake = FakeMLflow()
    with pytest.raises(ValueError, match="experiment name"):
        start_tracked_run("PPE Detection", "m", mlflow_module=fake, **TAGS)
    assert fake.calls == []


def test_full_flow_order_and_content():
    fake = FakeMLflow()
    run = _start(fake, params={"epochs": 50, "data": "F:\\work\\data.yaml"})
    assert run == "RUN"
    assert fake.calls == [
        ("set_tracking_uri", "http://tracking.example"),
        ("set_experiment", "ppe-detection"),
        ("start_run", "ahmed_yolov9e_2026-10-21"),
        ("set_tags", TAGS),
        (
            "log_params",
            {"epochs": "50", "data": "data.yaml", "model": "yolov9e"},
        ),
    ]


def test_extra_tags_cannot_override_mandatory_tags():
    fake = FakeMLflow()
    _start(fake, extra_tags={"author": "someone-else", "stage": "train"})
    tags = next(call[1] for call in fake.calls if call[0] == "set_tags")
    assert tags["author"] == "ahmed"
    assert tags["stage"] == "train"


def test_params_cannot_override_the_model_param():
    fake = FakeMLflow()
    _start(fake, params={"model": "other"})
    params = next(call[1] for call in fake.calls if call[0] == "log_params")
    assert params["model"] == "yolov9e"


def test_run_is_ended_as_failed_if_tagging_fails():
    class BrokenTags(FakeMLflow):
        def set_tags(self, tags):
            raise RuntimeError("server error")

    fake = BrokenTags()
    with pytest.raises(RuntimeError, match="server error"):
        _start(fake)
    assert fake.calls[-1] == ("end_run", "FAILED")


def test_explicit_run_name_wins():
    fake = FakeMLflow()
    _start(fake, run_name="custom-name")
    assert ("start_run", "custom-name") in fake.calls


def test_real_mlflow_roundtrip(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    with start_tracked_run(
        "ppe-detection",
        "yolov9e",
        {"epochs": 3, "data": "/home/someone/data.yaml"},
        tracking_uri=uri,
        day=DAY,
        **TAGS,
    ) as run:
        run_id = run.info.run_id
        mlflow.log_metric("val_map50", 0.5)
    stored = mlflow.tracking.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert stored.info.run_name == "ahmed_yolov9e_2026-10-21"
    for key, value in TAGS.items():
        assert stored.data.tags[key] == value
    assert stored.data.params == {
        "epochs": "3",
        "data": "data.yaml",
        "model": "yolov9e",
    }
    assert stored.data.metrics == {"val_map50": 0.5}
