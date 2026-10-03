# Industrial Safety AI System — Backend

Backend platform for real-time safety event ingestion, violation persistence, alert generation, and monitoring data retrieval.

---

## 1. System Context & Responsibility Chain

```text
Ahmed Osama (Detection / PPE Detection)
       ↓
Ahmed El-Shennawy (Tracking + Danger Zones + Safety Rules)
       ↓
Ahmed Selim (Real-Time Integration Pipeline)
       ↓
[ MALEK — FASTAPI BACKEND + POSTGRESQL ]
  - Event Ingestion API (/api/v1/events)
  - Strict Pydantic Validation & Normalization
  - Atomic PostgreSQL Persistence (Events + Violations + Alerts)
  - Upstream Retry Deduplication (HTTP 409)
  - Read APIs (/api/v1/events, /api/v1/violations, /api/v1/alerts)
       ↓
 ┌─────────────────────────┬─────────────────────────┐
 ↓                         ↓                         ↓
Abdelrahman (MLOps)        API Consumers             Future Dashboard
Deployment & Run Contract   Reporting & Integrations  Stored Alerts & Auditing
```

See the detailed integration contracts:
- [AI-to-Backend Integration Contract](../docs/integration/AI_TO_BACKEND_CONTRACT.md) — For Ahmed Selim
- [Backend MLOps & Consumer Handoff](../docs/integration/BACKEND_HANDOFF.md) — For Abdelrahman & Dashboard consumers

---

## 2. Prerequisites and scope

- **Python**: 3.12 or newer.
- **PostgreSQL**: 16 or newer (tested with PostgreSQL 17).
- PostgreSQL is the only runtime dependency. `docker-compose.yml` starts a local database, not the API.
- This is a term-one integration backend for trusted development environments. It has no authentication or external alert delivery. An alert is a stored record available through the read API.

---

## 3. Quickstart & Setup

### 3.1 Virtual environment and database

```powershell
cd industrial-safety-ai\backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
# Replace the example password in .env with one local password in both places.
docker compose up -d db
```

If PostgreSQL is already installed, create the `industrial_safety` database and set
`DATABASE_URL` in `.env` instead of running Compose. Keep `.env` private. For a fresh
Compose database, the configured user and database are created automatically.

### 3.2 Database migration and verification

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic check
.\.venv\Scripts\python.exe -m scripts.seed  # checks connectivity; inserts no data
```

### 3.3 Start the backend server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

- Swagger UI: `http://127.0.0.1:8010/docs`
- Health check: `http://127.0.0.1:8010/health`
- `GET /health` returns `200` when PostgreSQL is ready and `503` otherwise.

---

## 4. API Surface

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | PostgreSQL readiness check |
| `POST` | `/api/v1/events` | Ingest structured AI safety event; creates violation & alert if applicable |
| `GET` | `/api/v1/events` | List events with filters (`camera_id`, `event_type`, `is_violation`, `from`, `to`, `limit`, `offset`) |
| `GET` | `/api/v1/events/{id}` | Retrieve single event by backend UUID |
| `GET` | `/api/v1/violations` | List violations with filters (`violation_type`, `severity`, `camera_id`, `status`, `limit`, `offset`) |
| `GET` | `/api/v1/violations/{id}` | Retrieve single violation by backend UUID |
| `GET` | `/api/v1/alerts` | List alerts with status filter and pagination |
| `GET` | `/api/v1/alerts/{id}` | Retrieve single alert by backend UUID |

---

## 5. Event Ingestion Flow (`POST /api/v1/events`)

An upstream structured AI event is ingested as follows:

1. **Schema Validation**: Pydantic strictly validates all fields, bounds, and cross-field logic. Violation event types require `is_violation: true`, `violation_type`, and `severity`.
2. **Duplicate Check**: The database enforces uniqueness on `event_id`. Duplicate retries return `HTTP 409 Conflict` and do not create another alert. The sender must retain the same ID across retries.
3. **Atomic Transaction**:
   - `events` table record is inserted.
   - If `is_violation == true`, a linked `violations` record is created.
   - An active `alerts` record is created and linked to the violation.
   - The entire chain is committed in a single atomic transaction.

### Example Ingestion Request

```json
{
  "event_id": "evt_CAM02_20261002_112030_002",
  "event_type": "ppe_violation",
  "occurred_at": "2026-10-02T11:20:30Z",
  "camera_id": "CAM-02",
  "track_id": "TRK-205",
  "person_id": null,
  "zone_id": "ZONE-CONSTRUCTION",
  "confidence": 0.92,
  "is_violation": true,
  "violation_type": "missing_helmet",
  "severity": "high",
  "description": "Worker operating in construction zone without mandated safety helmet",
  "ppe": {
    "helmet": false,
    "vest": true,
    "gloves": true,
    "goggles": false
  },
  "frame_reference": "frames/CAM-02/evt_002.jpg",
  "metadata": {
    "weather": "clear"
  }
}
```

### Ingestion Response (`201 Created`)

```json
{
  "status": "accepted",
  "event_id": "evt_CAM02_20261002_112030_002",
  "backend_id": "98f9e36f-fe8d-4cde-a73b-a7a20804d136",
  "violation_created": true,
  "alert_created": true
}
```

---

## 6. Testing & Quality Assurance

### 6.1 Run automated pytest suite

```powershell
# Unit and contract tests only (no database needed)
.\.venv\Scripts\python.exe -m pytest -m "not integration" -v

# Create a dedicated disposable database once (never point tests at development data):
docker compose exec db createdb -U safety_admin industrial_safety_test
# Full suite including PostgreSQL integration; reads TEST_DATABASE_URL from .env
.\.venv\Scripts\python.exe -m pytest -v
```

### 6.2 Run Live End-to-End Integration Smoke Test

```powershell
.\.venv\Scripts\python.exe -m scripts.integration_smoke_test --url http://127.0.0.1:8010
```

The smoke test writes five sample events to the configured development database and
uses new event IDs on each run. Do not aim it at a shared or production database.

### 6.3 Send Sample Events CLI

```powershell
# Send sample PPE violation with automatic duplicate retry verification
.\.venv\Scripts\python.exe -m scripts.send_sample_event --payload ppe_violation_event.json --duplicate
```

---

## 7. Ready-to-Use Example Payloads

Located in `backend/examples/`:
- `normal_person_event.json`
- `ppe_violation_event.json`
- `danger_zone_violation_event.json`
- `zone_entry_event.json`
- `zone_exit_event.json`

## 8. Handoff

The sender contract is [AI-to-Backend Integration Contract](../docs/integration/AI_TO_BACKEND_CONTRACT.md).
Deployment and consumer details are in [Backend Handoff](../docs/integration/BACKEND_HANDOFF.md).
The backend stores event metadata and optional `frame_reference` text only; it does not
upload image evidence, evaluate computer vision rules, deliver notifications, or manage
alert status transitions.
