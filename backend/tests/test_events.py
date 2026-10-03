import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.models.entities import Alert, SafetyEvent, Violation

pytestmark = pytest.mark.integration

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str) -> dict:
    with open(EXAMPLES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def test_case_a_normal_person_event(client, db_session):
    payload = {**load_example("normal_person_event.json"), "event_id": str(uuid4())}
    response = client.post("/api/v1/events", json=payload)
    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "accepted"
    assert result["event_id"] == payload["event_id"]
    assert result["violation_created"] is False
    assert result["alert_created"] is False

    # Verify database state
    event_count = db_session.scalar(select(func.count()).select_from(SafetyEvent))
    violation_count = db_session.scalar(select(func.count()).select_from(Violation))
    alert_count = db_session.scalar(select(func.count()).select_from(Alert))
    assert event_count == 1
    assert violation_count == 0
    assert alert_count == 0

    # Verify retrieval
    backend_id = result["backend_id"]
    get_res = client.get(f"/api/v1/events/{backend_id}")
    assert get_res.status_code == 200
    stored = get_res.json()
    assert stored["event_id"] == payload["event_id"]
    assert stored["is_violation"] is False


def test_case_b_ppe_violation_full_chain(client, db_session):
    payload = {**load_example("ppe_violation_event.json"), "event_id": str(uuid4())}
    response = client.post("/api/v1/events", json=payload)
    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "accepted"
    assert result["violation_created"] is True
    assert result["alert_created"] is True

    # Verify event stored
    event = db_session.scalar(select(SafetyEvent).where(SafetyEvent.event_id == payload["event_id"]))
    assert event is not None
    assert event.is_violation is True

    # Verify violation created and linked
    violation = db_session.scalar(select(Violation).where(Violation.event_id == event.id))
    assert violation is not None
    assert violation.violation_type == "missing_helmet"
    assert violation.severity == "high"
    assert violation.camera_id == payload["camera_id"]

    # Verify alert created and linked
    alert = db_session.scalar(select(Alert).where(Alert.violation_id == violation.id))
    assert alert is not None
    assert "missing_helmet" in alert.message
    assert alert.status == "active"


def test_case_c_danger_zone_violation(client, db_session):
    payload = {**load_example("danger_zone_violation_event.json"), "event_id": str(uuid4())}
    response = client.post("/api/v1/events", json=payload)
    assert response.status_code == 201
    result = response.json()
    assert result["violation_created"] is True
    assert result["alert_created"] is True

    violations = client.get("/api/v1/violations").json()
    assert violations["total"] == 1
    assert violations["items"][0]["violation_type"] == "danger_zone_unauthorized"
    assert violations["items"][0]["severity"] == "critical"

    alerts = client.get("/api/v1/alerts").json()
    assert alerts["total"] == 1
    assert "danger_zone_unauthorized" in alerts["items"][0]["message"]


def test_case_d_zone_entry(client, db_session):
    payload = {**load_example("zone_entry_event.json"), "event_id": str(uuid4())}
    response = client.post("/api/v1/events", json=payload)
    assert response.status_code == 201
    assert response.json()["violation_created"] is False
    assert db_session.scalar(select(func.count()).select_from(Violation)) == 0


def test_case_e_zone_exit(client, db_session):
    payload = {**load_example("zone_exit_event.json"), "event_id": str(uuid4())}
    response = client.post("/api/v1/events", json=payload)
    assert response.status_code == 201
    assert response.json()["violation_created"] is False
    assert db_session.scalar(select(func.count()).select_from(Violation)) == 0


def test_case_f_duplicate_retry_behavior(client, db_session):
    fixed_id = f"evt_dup_{uuid4()}"
    payload = {**load_example("ppe_violation_event.json"), "event_id": fixed_id}

    # First attempt: succeeds
    res1 = client.post("/api/v1/events", json=payload)
    assert res1.status_code == 201

    # Second attempt with same event_id: rejected with 409
    res2 = client.post("/api/v1/events", json=payload)
    assert res2.status_code == 409
    assert res2.json()["detail"] == "event_id already received"

    # Confirm only 1 event, 1 violation, 1 alert in DB
    assert db_session.scalar(select(func.count()).select_from(SafetyEvent)) == 1
    assert db_session.scalar(select(func.count()).select_from(Violation)) == 1
    assert db_session.scalar(select(func.count()).select_from(Alert)) == 1


def test_case_g_invalid_payload_rejected(client, db_session):
    # Missing required camera_id
    invalid = {
        "event_id": f"evt_inv_{uuid4()}",
        "event_type": "person_detected",
        "occurred_at": "2026-10-02T14:32:05Z",
        # camera_id omitted
        "is_violation": False,
    }
    response = client.post("/api/v1/events", json=invalid)
    assert response.status_code == 422
    assert db_session.scalar(select(func.count()).select_from(SafetyEvent)) == 0


def test_case_g2_violation_missing_type_rejected(client, db_session):
    # is_violation=True but missing violation_type and severity
    invalid = {
        "event_id": f"evt_inv_{uuid4()}",
        "event_type": "ppe_violation",
        "occurred_at": "2026-10-02T14:32:05Z",
        "camera_id": "CAM-01",
        "is_violation": True,
        # violation_type and severity missing
    }
    response = client.post("/api/v1/events", json=invalid)
    assert response.status_code == 422
    assert db_session.scalar(select(func.count()).select_from(SafetyEvent)) == 0


def test_filtering_and_pagination(client):
    e1 = {
        **load_example("normal_person_event.json"),
        "event_id": str(uuid4()),
        "camera_id": "CAM-01",
        "occurred_at": "2026-10-02T10:00:00Z",
    }
    e2 = {
        **load_example("ppe_violation_event.json"),
        "event_id": str(uuid4()),
        "camera_id": "CAM-02",
        "occurred_at": "2026-10-02T11:00:00Z",
    }
    client.post("/api/v1/events", json=e1)
    client.post("/api/v1/events", json=e2)

    # Filter by camera_id
    res_cam = client.get("/api/v1/events?camera_id=CAM-01").json()
    assert res_cam["total"] == 1
    assert res_cam["items"][0]["camera_id"] == "CAM-01"

    # Filter by is_violation
    res_viol = client.get("/api/v1/events?is_violation=true").json()
    assert res_viol["total"] == 1
    assert res_viol["items"][0]["event_id"] == e2["event_id"]

    # Filter by event_type
    res_type = client.get("/api/v1/events?event_type=person_detected").json()
    assert res_type["total"] == 1

    # Filter by time range
    res_time = client.get("/api/v1/events?from=2026-10-02T10:30:00Z").json()
    assert res_time["total"] == 1
    assert res_time["items"][0]["camera_id"] == "CAM-02"

    # Pagination
    res_page = client.get("/api/v1/events?limit=1&offset=0").json()
    assert res_page["total"] == 2
    assert len(res_page["items"]) == 1


def test_unknown_event_id(client):
    assert client.get(f"/api/v1/events/{uuid4()}").status_code == 404
    assert client.get("/api/v1/events/not-a-valid-uuid").status_code == 422
