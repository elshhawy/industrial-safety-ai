import json
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str) -> dict:
    with open(EXAMPLES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def test_violations_read_and_filters(client):
    v1 = {
        **load_example("ppe_violation_event.json"),
        "event_id": str(uuid4()),
        "camera_id": "CAM-01",
        "violation_type": "missing_helmet",
        "severity": "high",
    }
    v2 = {
        **load_example("danger_zone_violation_event.json"),
        "event_id": str(uuid4()),
        "camera_id": "CAM-02",
        "violation_type": "danger_zone_unauthorized",
        "severity": "critical",
    }
    normal = {
        **load_example("normal_person_event.json"),
        "event_id": str(uuid4()),
    }

    client.post("/api/v1/events", json=v1)
    client.post("/api/v1/events", json=v2)
    client.post("/api/v1/events", json=normal)

    # Total violations should be 2, ignoring normal event
    all_v = client.get("/api/v1/violations").json()
    assert all_v["total"] == 2

    # Filter by violation_type
    filtered_type = client.get("/api/v1/violations?violation_type=missing_helmet").json()
    assert filtered_type["total"] == 1
    assert filtered_type["items"][0]["camera_id"] == "CAM-01"

    # Filter by severity
    filtered_sev = client.get("/api/v1/violations?severity=critical").json()
    assert filtered_sev["total"] == 1
    assert filtered_sev["items"][0]["violation_type"] == "danger_zone_unauthorized"

    # Filter by camera_id
    filtered_cam = client.get("/api/v1/violations?camera_id=CAM-01").json()
    assert filtered_cam["total"] == 1

    # Get single violation by backend ID
    v_id = all_v["items"][0]["id"]
    single = client.get(f"/api/v1/violations/{v_id}")
    assert single.status_code == 200
    assert single.json()["id"] == v_id

    # 404 for unknown ID
    assert client.get(f"/api/v1/violations/{uuid4()}").status_code == 404
    assert client.get("/api/v1/violations/bad-id").status_code == 422
