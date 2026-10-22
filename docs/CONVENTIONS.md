# Experiment tracking conventions

Rules for logging training runs with MLflow. Status: proposal, to be reviewed
by the team in the pull request that adds this file.

## Where runs go

- Tracking repo (DagsHub, **public**): `elshhawy/industrial-safety-ai-tracking`
- MLflow URL: `https://dagshub.com/elshhawy/industrial-safety-ai-tracking.mlflow`
- Public means anyone on the internet can read everything logged there.
- Code lives on GitHub, data and weights live in DVC (Google Drive).

## What may be logged

- Allowed: params, metrics, tags.
- Not allowed: datasets, images or videos of people, model weights, local file
  paths, secrets, real names.
- Weights are versioned with DVC, not uploaded to MLflow.
- The Ultralytics MLflow callback uploads the weights folder, plots and all
  training args by default. Turn it off (`yolo settings mlflow=False`) and log
  through our own helper instead.

## Mandatory tags

Every run must have these three tags. A run missing one must not start.

| Tag | Meaning | Resolved from |
|---|---|---|
| `author` | A handle, not a real name | argument, env `MLFLOW_AUTHOR`, `git config user.name` |
| `git_commit` | Code that produced the run | argument, `git rev-parse HEAD` |
| `dataset_version` | Data the run was trained on | argument, env `DATASET_VERSION`, md5 in the `.dvc` file |

Helpers: `industrial_safety.mlops.mlflow_utils` (`build_required_tags`,
`validate_tags`). Set `MLFLOW_AUTHOR` to a handle so a real name is never
published by the `git config user.name` fallback.

## Naming

- Experiment: one per task, lowercase with hyphens, e.g. `ppe-detection`.
- Run name: `<author>_<model>_<yyyy-mm-dd>`, e.g. `ahmed_yolov9e_2026-10-21`.
- Params: snake_case. Always log at least `model`, `epochs`, `imgsz`, `batch`,
  `seed` and the class list.
- Metrics: snake_case with the split first: `val_map50`, `val_map50_95`,
  `val_precision`, `val_recall`.
- Ultralytics metric names differ (for example `metrics/mAP50(B)`). The mapping
  to the names above is defined when the training code is wired to MLflow.

## Tokens and login

- Each member uses their own DagsHub account and token. Never share a token,
  never commit one, never paste one in chat.
- On a dev machine: `dagshub.init(..., mlflow=True)` opens a browser login.
- Without a browser (servers, Colab): set the env var `DAGSHUB_USER_TOKEN`.
- Never call `dagshub.init(..., dvc=True)`. It writes a DagsHub remote into the
  DVC config, and our DVC remote is Google Drive.

## Local fallback

If no tracking URI is set, MLflow 3.x silently creates `mlflow.db` in the
current folder. It is git-ignored and is not a team record. Set
`MLFLOW_TRACKING_URI` or use the helper.

## Open points

- The DagsHub server showed MLflow 3.5.1, the client used 3.17.1. Artifacts and
  the model registry are not tested against it yet.
- Not known whether the 20 GB DagsHub storage counts public repos.
- The class taxonomy is not decided, so every run records its class list.
