# Quickstart

From a fresh clone to a first tracked run. Windows, Git Bash.

Status: DVC and experiment tracking are set up. **No dataset is tracked yet** and
the training code is not wired to tracking yet, so this guide ends with a small
connection test.

## 1. Set up the project

Follow "Getting started" in the README, then install the extras:

~~~bash
pip install -e ".[dev,dvc,tracking]"
pre-commit install
pytest
~~~

`pytest` should end with `passed` and no failures. Run the install again after
pulling a change to `pyproject.toml`, so the pinned versions are applied.

## 2. Data (DVC)

Follow [dvc-setup.md](dvc-setup.md). There is no dataset to pull yet. The steps are
still worth doing once, so that you can pull Dataset V1 when it exists.

## 3. Track a run

You need a DagsHub account that the repo owner added as a collaborator of
`elshhawy/industrial-safety-ai-tracking`. Reading is public, writing needs this
(not tested with a teammate yet).

Set three variables in your terminal (they last until you close it):

~~~bash
export DAGSHUB_REPO=elshhawy/industrial-safety-ai-tracking
export MLFLOW_AUTHOR=<your handle, not your real name>
export DATASET_VERSION=<short name of the data you used>
~~~

Connection test, with a throw-away experiment name:

~~~bash
python - <<'EOF'
import mlflow
from industrial_safety.mlops.tracking import start_tracked_run

with start_tracked_run("tracking-smoke-test", "smoke-model", {"epochs": 1}):
    mlflow.log_metric("dummy_metric", 1.0)
EOF
~~~

The first run may open a browser to log in to DagsHub. Without a browser (a server
or Colab), set `DAGSHUB_USER_TOKEN` instead. Then open
`https://dagshub.com/elshhawy/industrial-safety-ai-tracking.mlflow`. The page is
public, so the test run is visible to everyone.

In real work use the experiment name of your task (for example `ppe-detection`)
and log only measured values:

~~~python
with start_tracked_run("ppe-detection", "yolov9e", {"epochs": 50, "imgsz": 640}):
    ...  # train and validate, then:
    mlflow.log_metric("val_map50", val_map50)  # your measured value
~~~

`start_tracked_run` refuses to start when:

- no tracking server is configured (`DAGSHUB_REPO` or `MLFLOW_TRACKING_URI`),
- the experiment name is not lowercase words joined by hyphens,
- `author`, `git_commit` or `dataset_version` cannot be found.

It never uploads weights, plots or datasets, and it shortens a local path in the
params to the file name.

## 4. Rules

Read [CONVENTIONS.md](CONVENTIONS.md) once. The short version:

- The tracking repo is public. Log params, metrics and tags only.
- Weights and data go through DVC, never through MLflow or Git.
- If you train with Ultralytics, turn its MLflow callback off
  (`yolo settings mlflow=False`). By default it uploads the weights, the plots and
  all training arguments.
- Never paste tokens or client secrets anywhere.

## Troubleshooting

- DVC or login errors: [dvc-setup.md](dvc-setup.md#troubleshooting).
- `TrackingNotConfiguredError`: set `DAGSHUB_REPO` (or `MLFLOW_TRACKING_URI`).
- `MissingTagsError`: the message lists the missing tags. Set `MLFLOW_AUTHOR` and
  `DATASET_VERSION`, and run inside the Git repository.
