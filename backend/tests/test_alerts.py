import json
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str) -> dict:
    with open(EXAMPLES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def test_alerts_read_and_filters(client):
    v1 = {
        **load_example("ppe_violation_event.json"),
        "event_id": str(uuid4()),
        "camera_id": "CAM-01",
    }
    client.post("/api/v1/events", json=v1)

    alerts = client.get("/api/v1/alerts").json()
    assert alerts["total"] == 1
    assert alerts["items"][0]["status"] == "active"
    assert "missing_helmet" in alerts["items"][0]["message"]

    # Filter by status
    active_alerts = client.get("/api/v1/alerts?status=active").json()
    assert active_alerts["total"] == 1

    ack_alerts = client.get("/api/v1/alerts?status=acknowledged").json()
    assert ack_alerts["total"] == 0

    # Get single alert by ID
    a_id = alerts["items"][0]["id"]
    single = client.get(f"/api/v1/alerts/{a_id}")
    assert single.status_code == 200
    assert single.json()["id"] == a_id

    # 404 for unknown ID
    assert client.get(f"/api/v1/alerts/{uuid4()}").status_code == 404
    assert client.get("/api/v1/alerts/invalid-uuid").status_code == 422
