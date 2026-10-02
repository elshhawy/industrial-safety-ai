# Contributing

## One-time setup (Windows, Git Bash)

1. Install Git and Python 3.12 (3.10 or newer works).
2. Accept the repository invitation (check your email or GitHub notifications).
3. Set your Git identity and two Windows settings:

~~~bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global core.autocrlf false
git config --global core.longpaths true
~~~

4. Clone the project into a short path outside OneDrive (for example `D:\dev`) and set it up:

~~~bash
git clone https://github.com/elshhawy/industrial-safety-ai.git
cd industrial-safety-ai
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
pre-commit install
pytest
~~~

`pytest` should end with `1 passed`. Activate the virtual environment
(`source .venv/Scripts/activate`) in every new terminal before working.

## Daily workflow

~~~bash
git switch main
git pull --prune
git switch -c feat/<area>-<short-description>
~~~

Work and commit often. `pre-commit` checks every commit.

~~~bash
git add <files>
git commit -m "feat: short description"
git push -u origin feat/<area>-<short-description>
~~~

Then on GitHub: open a Pull Request, fill in the template, wait for CI to be
green, ask a teammate to review, and use **Squash and merge**. The branch is
deleted automatically. Clean up your machine:

~~~bash
git switch main
git pull --prune
git branch -D <your-branch>
~~~

(`-D` is needed because a squash merge creates a new commit, so Git cannot
tell the branch was merged. Only use it after the PR is merged.)

## Rules

- Never push to `main`. GitHub blocks it. Every change goes through a pull request.
- Branch names: `feat/...`, `fix/...`, `docs/...`, `chore/...`.
- Commit messages start with `feat:`, `fix:`, `docs:`, `test:`, `ci:` or `chore:`.
- Keep pull requests small (one or two days of work). Split big work into steps.
- If a branch lives for more than a couple of days, update it:
  `git fetch origin` then `git merge origin/main`.
- This repository is **public**. Never commit data, videos or images of people,
  model weights, keys or `.env` files.
- Any number you report (mAP, FPS, ...) must be measured, with the source stated.
- Do not edit another person's folder directly. Open a pull request and ask the owner to review.

## When something fails

- **pre-commit says Failed and files were modified:** run `git add` on those files and commit again.
- **CI is red:** open the failed step in the Checks tab of the PR, then run
  `ruff check src tests`, `ruff format src tests` and `pytest` locally.
- **Stuck:** ask in the team chat and paste the full command output.

## Folder owners (proposed)

| Folder | Owner |
|---|---|
| `src/industrial_safety/detection/`, `training/` | Ahmed Osama |
| `src/industrial_safety/tracking/`, `src/industrial_safety/safety/` | Ahmed El-Shennawy |
| `src/industrial_safety/data_prep/`, `data/` | Rizk |
| `backend/` | Malek |
| `src/industrial_safety/pipeline/` | Ahmed Selim |
| `.github/`, `docker/`, `pyproject.toml` | Abdelrahman |
| `src/industrial_safety/common/`, `configs/`, `tests/`, `docs/` | Shared (review by the team) |
