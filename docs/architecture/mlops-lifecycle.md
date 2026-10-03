# MLOps lifecycle (Phase 1 design)

Status: **proposal**. Items marked **Pending** need a team decision.
So far only Git, CI and pre-commit are implemented in this project.

## Goal

Reproducibility: for any model we can answer four questions: which code,
which data, which settings, and which result.

## The loop

~~~text
data (DVC) -> train (Colab/Kaggle) -> track runs (MLflow) -> register best model
     ^                                                            |
     |                                                            v
monitor (logs, measured FPS) <- deploy (Docker Compose) <- package (Docker, CI/CD)
~~~

Git sits under every stage: it stores code, config files and small pointer files.

## What lives where

| Thing | Stored in | Notes |
|---|---|---|
| Code, configs, docs | Git (GitHub) | Always through pull requests |
| Datasets | DVC | Git holds a small `.dvc` pointer; the files live in a DVC remote |
| Model weights | DVC remote or MLflow artifacts | **Pending** |
| Experiment records (settings, metrics) | MLflow | **Pending**: where MLflow runs when training happens on Colab/Kaggle |
| Docker images | **Pending** | Registry not chosen |
| Secrets, credentials | Never in Git | The repository is public |

## Conventions (recommendation)

- A dataset version is a Git commit that contains the updated `.dvc` pointer,
  plus a tag such as `data-v1`.
- Every training run logs: the Git commit, the dataset pointer hash, the
  parameters, the metrics, and where it ran (Colab, Kaggle or local).
- Only measured numbers are reported (mAP, FPS), with the hardware and the
  data version stated.
- The real-time pipeline uses a named, registered model version, never
  "whatever file is on someone's laptop".
- Data and model files never go into Git.

## Who does what (proposed)

| Person | MLOps-related work |
|---|---|
| Abdelrahman | DVC and MLflow setup, CI/CD, Docker, monitoring |
| Rizk | Dataset versions (`dvc add`, `dvc push`) and data quality |
| Ahmed Osama | Training scripts and runs, logging to MLflow |
| Ahmed Selim | Loads the registered model in the real-time pipeline |

## Plan for Term 1 (proposed)

| Step | Deliverable | Phase |
|---|---|---|
| 1 | Repo, tooling, CI, branch rules, contributing guide | 1 (done) |
| 2 | This design document | 1 |
| 3 | DVC initialised in the repo (no remote yet) | 1-2 |
| 4 | Dataset v1 tracked with DVC, remote chosen | 2 |
| 5 | Training script that logs to MLflow | 2 |
| 6 | Docker image and Compose file for the runtime | 2, hardened in 6 |
| 7 | CI builds the image; end-to-end test | 2-6 |
| 8 | Monitoring: logs and measured FPS/latency | 6 |

Phase numbers follow the team's 12-week plan.

## Pending decisions

1. DVC remote: where the team stores shared data. The repository is public and
   videos may show people, so the storage must be access-restricted and
   credentials must never be committed.
2. MLflow with Colab/Kaggle: how runs are recorded when training does not
   happen on a local machine.
3. Model and tracker choice (the team decides, not MLOps alone).
4. Docker image registry.
5. Monitoring stack (Term 2).
6. Whether to build an end-to-end path with stubs first (walking skeleton).
7. Repository license.

## Sandbox experiment

DVC 3.67.1 was tried in a throwaway repository outside this project:
`dvc add` produced a pointer file; tags `data-v1` and `data-v2` marked two
dataset versions; `git checkout <tag> -- data.dvc` followed by `dvc checkout`
switched between them; and after deleting the local data and cache,
`dvc pull` restored everything from a folder-based remote. The folder remote
was for learning only.
