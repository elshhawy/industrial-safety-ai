"""Integration smoke test for Industrial Safety AI System.

Tests the live running API end-to-end against all acceptance criteria:
  - System health check (/health)
  - Case A: Normal event ingestion (stored, no violation/alert)
  - Case B: PPE violation ingestion (stored, violation + alert generated)
  - Case C: Danger zone violation ingestion (stored, critical violation + alert generated)
  - Case D: Zone entry event
  - Case E: Zone exit event
  - Case F: Duplicate event retry (409 Conflict)
  - Case G: Invalid contract payload (422 Rejected)
  - Data retrieval via REST Read APIs (Events, Violations, Alerts)
"""
import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def request_json(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict]:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = Request(url, data=payload, headers=headers, method=method)
    try:
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except HTTPError as exc:
        err_body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(err_body)
        except Exception:
            parsed = {"raw": err_body}
        return exc.code, parsed


def load_example(name: str) -> dict:
    with open(EXAMPLES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial Safety AI integration smoke test")
    parser.add_argument("--url", default="http://127.0.0.1:8010", help="Backend base URL")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    passed = 0
    total = 0

    def step(title: str, condition: bool, details: str = ""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {title}")
        else:
            print(f"  [FAIL] {title} - {details}")

    print("\n==================================================================")
    print(" Industrial Safety AI Backend — End-to-End Integration Smoke Test")
    print(f" Target: {base_url}")
    print("==================================================================\n")

    # 1. Health check
    print("--- 1. Service Health & Connectivity ---")
    status, body = request_json(f"{base_url}/health")
    step("GET /health returns 200", status == 200, f"Got {status}")
    step("Service status is 'ok'", body.get("status") == "ok", f"Got {body.get('status')}")
    step("PostgreSQL database is connected ('ok')", body.get("database") == "ok", f"Got {body.get('database')}")

    # 2. Case A: Normal event
    print("\n--- 2. Case A: Normal Event (person_detected) ---")
    norm_id = f"smoke_norm_{uuid4()}"
    norm_payload = {**load_example("normal_person_event.json"), "event_id": norm_id}
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=norm_payload)
    step("POST /api/v1/events returns 201", status == 201, f"Got {status}")
    step("Event accepted", body.get("status") == "accepted")
    step("Violation is NOT created", body.get("violation_created") is False)
    step("Alert is NOT created", body.get("alert_created") is False)
    norm_backend_id = body.get("backend_id")

    # 3. Case B: PPE violation
    print("\n--- 3. Case B: PPE Violation (missing_helmet) ---")
    ppe_id = f"smoke_ppe_{uuid4()}"
    ppe_payload = {**load_example("ppe_violation_event.json"), "event_id": ppe_id}
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=ppe_payload)
    step("POST /api/v1/events returns 201", status == 201, f"Got {status}")
    step("Violation record created", body.get("violation_created") is True)
    step("Alert record created", body.get("alert_created") is True)
    ppe_backend_id = body.get("backend_id")

    # 4. Case C: Danger Zone violation
    print("\n--- 4. Case C: Danger Zone Violation (danger_zone_unauthorized) ---")
    dz_id = f"smoke_dz_{uuid4()}"
    dz_payload = {**load_example("danger_zone_violation_event.json"), "event_id": dz_id}
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=dz_payload)
    step("POST /api/v1/events returns 201", status == 201, f"Got {status}")
    step("Critical violation created", body.get("violation_created") is True)
    step("Alert created", body.get("alert_created") is True)

    # 5. Case D & E: Zone Entry and Exit
    print("\n--- 5. Cases D & E: Zone Entry & Exit Events ---")
    entry_id = f"smoke_entry_{uuid4()}"
    entry_payload = {**load_example("zone_entry_event.json"), "event_id": entry_id}
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=entry_payload)
    step("Zone entry returns 201", status == 201, f"Got {status}")
    step("Zone entry creates no violation", body.get("violation_created") is False)

    exit_id = f"smoke_exit_{uuid4()}"
    exit_payload = {**load_example("zone_exit_event.json"), "event_id": exit_id}
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=exit_payload)
    step("Zone exit returns 201", status == 201, f"Got {status}")
    step("Zone exit creates no violation", body.get("violation_created") is False)

    # 6. Case F: Duplicate retry behavior
    print("\n--- 6. Case F: Upstream Duplicate Retry Handling ---")
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=ppe_payload)
    step("Resending same event_id returns 409 Conflict", status == 409, f"Got {status}")
    step("Error detail explains duplicate", "already received" in str(body.get("detail", "")).lower())

    # 7. Case G: Invalid payload rejection
    print("\n--- 7. Case G: Invalid Contract Schema Rejection ---")
    invalid_payload = {
        "event_id": f"smoke_inv_{uuid4()}",
        "event_type": "invalid_type",
        "occurred_at": "not-a-datetime",
        "camera_id": "",
    }
    status, body = request_json(f"{base_url}/api/v1/events", method="POST", data=invalid_payload)
    step("Invalid payload returns 422 Unprocessable Entity", status == 422, f"Got {status}")

    # 8. REST Read APIs & Data Integrity Verification
    print("\n--- 8. Downstream Read APIs Verification ---")
    # Events list
    status, events_data = request_json(f"{base_url}/api/v1/events")
    step("GET /api/v1/events returns 200", status == 200, f"Got {status}")
    step("Events collection includes submitted events", events_data.get("total", 0) >= 5)

    # Event detail
    status, single_event = request_json(f"{base_url}/api/v1/events/{norm_backend_id}")
    step("GET /api/v1/events/{id} retrieves correct event", status == 200 and single_event.get("event_id") == norm_id)

    # Violations list
    status, violations_data = request_json(f"{base_url}/api/v1/violations")
    step("GET /api/v1/violations returns 200", status == 200, f"Got {status}")
    step("Violations list includes safety violations", violations_data.get("total", 0) >= 2)
    v_items = violations_data.get("items", [])
    if v_items:
        v_id = v_items[0]["id"]
        status, single_v = request_json(f"{base_url}/api/v1/violations/{v_id}")
        step("GET /api/v1/violations/{id} retrieves violation record", status == 200 and single_v.get("id") == v_id)

    # Alerts list
    status, alerts_data = request_json(f"{base_url}/api/v1/alerts")
    step("GET /api/v1/alerts returns 200", status == 200, f"Got {status}")
    step("Alerts list includes generated alerts", alerts_data.get("total", 0) >= 2)
    a_items = alerts_data.get("items", [])
    if a_items:
        a_id = a_items[0]["id"]
        status, single_a = request_json(f"{base_url}/api/v1/alerts/{a_id}")
        step("GET /api/v1/alerts/{id} retrieves alert record", status == 200 and single_a.get("id") == a_id)

    print("\n==================================================================")
    print(f" Summary: {passed}/{total} checks passed ({(passed/total)*100:.1f}%)")
    print("==================================================================")

    if passed != total:
        print("\nSmoke test FAILED!")
        sys.exit(1)
    else:
        print("\nAll integration smoke tests PASSED successfully!")


if __name__ == "__main__":
    main()
