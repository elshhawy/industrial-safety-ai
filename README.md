# Industrial Safety AI System

Real-time AI-based industrial safety monitoring. The system detects people and
PPE (personal protective equipment) in video, tracks each person, checks
danger-zone and PPE rules, and logs violation events.

## Status

- Phase 1 (study, requirements and setup): in progress.
- Term 1 goal: a working, end-to-end, deployable core system.
- Term 2 (planned): alerts, dashboard, monitoring stack.

> The architecture and conventions below are proposals until the team confirms them.

## Architecture (Term 1, proposed)

~~~text
Build time (MLOps):
  data + DVC -> training (Colab/Kaggle) -> MLflow (runs, registry) -> model file

Runtime (Docker Compose, one machine):
  camera/video -> ai-worker -> backend API (FastAPI) -> PostgreSQL

Inside ai-worker:
  detection -> tracking -> safety rules -> violation event
~~~

DVC (Google Drive remote) and MLflow tracking (DagsHub) are set up. Each member
still needs access, see the guides below. No dataset is tracked with DVC yet.
Docker is planned.

## Repository structure

~~~text
industrial-safety-ai/
├── .github/
│   ├── workflows/            # CI (GitHub Actions)
│   └── ISSUE_TEMPLATE/       # issue templates (empty for now)
├── backend/                  # FastAPI + PostgreSQL service
├── configs/                  # configuration files
├── data/                     # datasets (DVC later, not tracked by Git)
├── docker/                   # Dockerfiles and Compose files
├── docs/
│   ├── architecture/         # design notes and diagrams
│   └── decisions/            # decision records
├── models/                   # model weights (DVC/MLflow later, not in Git)
├── notebooks/                # Colab/Kaggle experiments
├── scripts/                  # helper scripts
├── src/
│   └── industrial_safety/
│       ├── common/           # shared code (config, logging)
│       ├── data_prep/        # data preparation
│       ├── detection/        # person and PPE detection
│       ├── tracking/         # multi-object tracking
│       ├── safety/           # zones, rules, violations
│       └── pipeline/         # real-time integration
├── tests/
│   ├── unit/
│   └── integration/
├── training/                 # training scripts
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
~~~

## Getting started (Windows, Git Bash)

Requires Python 3.10 or newer (developed on 3.12).

One-time Git settings on each Windows machine:

~~~bash
git config --global core.autocrlf false
git config --global core.longpaths true
~~~

Set up the project:

~~~bash
git clone https://github.com/elshhawy/industrial-safety-ai.git
cd industrial-safety-ai
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
pre-commit install
pytest
~~~

Optional extras (install only what you need):

- `dvc`: data versioning with the shared Google Drive remote.
- `tracking`: MLflow and DagsHub client for experiment tracking.

~~~bash
pip install -e ".[dev,dvc,tracking]"
~~~

Run the install again after pulling a change to `pyproject.toml`, so the pinned
versions are applied. The full path from a clone to a first tracked run is in
[docs/quickstart.md](docs/quickstart.md).

## Quality checks

`pre-commit` runs automatically on every commit. CI (GitHub Actions) runs
the same checks plus the tests on every pull request. To run them by hand:

~~~bash
ruff check src tests
ruff format --check src tests
pytest
~~~

## Guides

- [docs/quickstart.md](docs/quickstart.md): from a fresh clone to a first tracked run.
- [docs/dvc-setup.md](docs/dvc-setup.md): connect to the shared DVC storage.
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md): rules for experiment tracking.

## Team conventions (proposed)

- Work on a branch; do not commit directly to `main`.
- Branch names: `feat/<area>-<desc>`, `fix/...`, `docs/...`, `chore/...`.
- Commit messages start with `feat:`, `fix:`, `docs:`, `test:`, `ci:` or `chore:`.
- Open a pull request; CI must pass and one teammate reviews it.
- Do not commit data, model weights or secrets (`.env`, `.dvc/config.local`, tokens).

## Team

| Name | Role |
|---|---|
| Abdelrahman | MLOps and infrastructure |
| Ahmed Osama | Computer vision (detection) |
| Ahmed El-Shennawy | Tracking and safety logic |
| Rizk | Data |
| Malek | Backend |
| Ahmed Selim | Real-time integration |
