# Industrial Safety AI System

Real-time AI-based industrial safety monitoring: person and PPE detection,
tracking, danger-zone monitoring, violation detection and event logging.

## Status
Phase 1: Study, requirements and setup (in progress).

## Repository structure
- `src/industrial_safety/` : detection, tracking, safety logic, real-time pipeline, data preparation, shared code
- `backend/`   : FastAPI + PostgreSQL service
- `training/`  : training scripts
- `configs/`   : configuration files
- `notebooks/` : experiments (Colab/Kaggle)
- `scripts/`   : helper scripts
- `docker/`    : Docker and Compose files
- `tests/`     : unit and integration tests
- `docs/`      : architecture and decision records
- `data/`, `models/` : managed by DVC later (not tracked by Git)

## Team
Abdelrahman (MLOps) - Ahmed Osama (Computer Vision) - Ahmed El-Shennawy (Tracking and Safety)
- Rizk (Data) - Malek (Backend) - Ahmed Selim (Real-Time Integration)
