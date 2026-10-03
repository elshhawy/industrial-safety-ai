# Backend Handoff Document (MLOps & API Consumers)

**Owner:** Malek — Backend Engineer
**Recipient 1:** Abdelrahman — MLOps
**Recipient 2:** API Consumers & Future Dashboard Developers

---

## 1. System Overview

The **Industrial Safety AI Backend** is a synchronous REST service built with **FastAPI**, **SQLAlchemy 2**, **Pydantic 2**, and **PostgreSQL**.

Its core responsibilities are:
1. Ingesting structured safety events from Edge computer vision pipelines.
2. Storing events, safety violations, and alert records in an atomic PostgreSQL transaction.
3. Guaranteeing event deduplication via database uniqueness constraints.
4. Exposing paginated, filterable read APIs for monitoring dashboards, audits, and downstream analytics.

This term-one service is intended for trusted integration environments. It has no
authentication, media upload, external notification delivery, or alert status update API.
An alert is a database record returned through the read API.

---

## 2. Runtime Prerequisites

| Component | Minimum Version | Verified Version | Notes |
|---|---|---|---|
| **Python** | 3.12+ | 3.12.14 | Managed via venv or uv |
| **PostgreSQL** | 16+ | 17.11 | No optional extension required |
| **Operating System** | Linux / Windows / macOS | Windows 11 | PostgreSQL or Docker Compose for the local database |

---

## 3. Environment Variables Configuration

Copy `.env.example` to `.env` in the `backend/` directory:

| Variable | Required | Default | Description | Example |
|---|---|---|---|---|
| `APP_ENV` | No | `development` | Environment label; currently does not change API behavior | `development` |
| `DATABASE_URL` | Yes | — | PostgreSQL connection URI using the `postgresql+psycopg` driver | `postgresql+psycopg://user:pass@localhost:5432/safety_db` |
| `TEST_DATABASE_URL` | For Tests | — | Dedicated disposable test DB ending in `_test` | `postgresql+psycopg://user:pass@localhost:5432/safety_db_test` |
| `LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

---

## 4. Deployment & Startup Sequence

Follow this exact startup sequence:

```text
1. PostgreSQL Database Service starts
              ↓
2. Alembic Migrations applied (alembic upgrade head)
              ↓
3. FastAPI Server starts (uvicorn app.main:app)
              ↓
4. Health check verifies liveness & DB connection (GET /health)
```

### Exact Commands

```bash
# 1. Navigate to backend directory
cd backend

# 2. Install dependencies (frozen lockfile)
pip install -r requirements-lock.txt

# 3. Apply database migrations
alembic upgrade head

# 4. Optional: verify database readiness
python -m scripts.seed

# 5. Start the integration server on loopback
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 5. Health Monitoring Endpoint

```http
GET /health
```

### Healthy Response (`200 OK`)
```json
{
  "status": "ok",
  "database": "ok"
}
```

### Unavailable Response (`503 Service Unavailable`)
```json
{
  "status": "degraded",
  "database": "unavailable"
}
```

Use `GET /health` as a readiness probe. It checks PostgreSQL and does not expose connection details.

---

## 6. Automated Testing & Verification

### Running Automated Tests

```bash
# Run unit & schema contract tests only (no database required)
pytest -m "not integration" -v

# Create a dedicated disposable PostgreSQL database ending in _test, then run:
# Set DATABASE_URL and TEST_DATABASE_URL in backend/.env before this command.
pytest -v

# Verify schema parity between models and database
alembic check
```

### Running Live Integration Smoke Test

When the server is running on `http://127.0.0.1:8000`:

```bash
python -m scripts.integration_smoke_test --url http://127.0.0.1:8000
```

---

## 7. Downstream REST Read APIs

Consumers (Dashboard / Monitoring) must query these REST endpoints. **Direct PostgreSQL queries are prohibited.**

### 7.1 Events API

- **List Events**: `GET /api/v1/events`
  - Query parameters:
    - `camera_id` (string, optional)
    - `event_type` (string, optional)
    - `is_violation` (boolean, optional)
    - `from` (ISO 8601 datetime, optional)
    - `to` (ISO 8601 datetime, optional)
    - `limit` (integer, default 50, max 100)
    - `offset` (integer, default 0)
  - Response shape:
    ```json
    {
      "total": 120,
      "limit": 50,
      "offset": 0,
      "items": [
        {
          "id": "e6f6f2b6-1f7e-406b-b7d4-1ae201344aaa",
          "event_id": "evt_CAM01_001",
          "event_type": "ppe_violation",
          "occurred_at": "2026-10-02T14:32:05Z",
          "camera_id": "CAM-01",
          "track_id": "42",
          "person_id": null,
          "zone_id": "ZONE-A",
          "confidence": 0.93,
          "is_violation": true,
          "violation_type": "missing_helmet",
          "severity": "high",
          "description": "Worker missing helmet",
          "ppe_details": {"helmet": false, "vest": true},
          "frame_reference": "frames/CAM-01/evt_001.jpg",
          "created_at": "2026-10-02T14:32:06Z"
        }
      ]
    }
    ```
- **Get Event by ID**: `GET /api/v1/events/{event_id}` (backend UUID)

### 7.2 Violations API

- **List Violations**: `GET /api/v1/violations`
  - Query parameters:
    - `violation_type` (string, optional)
    - `severity` (`low`, `medium`, `high`, `critical`, optional)
    - `camera_id` (string, optional)
    - `status` (`open`, `resolved`, optional)
    - `from` / `to` (ISO 8601 datetimes, optional)
    - `limit` / `offset` (pagination)
- **Get Violation by ID**: `GET /api/v1/violations/{violation_id}` (backend UUID)

### 7.3 Alerts API

- **List Alerts**: `GET /api/v1/alerts`
  - Query parameters:
    - `status` (`active`, `acknowledged`, `resolved`, optional)
    - `limit` / `offset` (pagination)
- **Get Alert by ID**: `GET /api/v1/alerts/{alert_id}` (backend UUID)

---

## 8. Interactive API Documentation

Interactive Swagger documentation is exposed at:
- Swagger UI: `http://<host>:<port>/docs`
- ReDoc: `http://<host>:<port>/redoc`
- OpenAPI JSON Spec: `http://<host>:<port>/openapi.json`
