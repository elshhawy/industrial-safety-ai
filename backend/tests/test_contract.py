from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.main import app
from app.schemas.event import SAMPLE_EVENT, SafetyEventCreate


def test_health_and_openapi():
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        expected_paths = {
            "/health",
            "/api/v1/events",
            "/api/v1/events/{event_id}",
            "/api/v1/violations",
            "/api/v1/violations/{violation_id}",
            "/api/v1/alerts",
            "/api/v1/alerts/{alert_id}",
        }
        assert expected_paths.issubset(set(schema["paths"]))


def test_valid_contract_and_timezone_normalization():
    event = SafetyEventCreate.model_validate(SAMPLE_EVENT)
    assert event.confidence == 0.93
    assert event.is_violation is True
    assert event.violation_type == "missing_helmet"

    converted = SafetyEventCreate.model_validate({**SAMPLE_EVENT, "occurred_at": "2026-10-02T17:32:05+03:00"})
    assert converted.occurred_at == event.occurred_at


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", 1.01),
        ("confidence", -0.01),
        ("event_type", "UNKNOWN_TYPE"),
        ("severity", "EXTREME"),
        ("violation_type", "NOT_A_VIOLATION"),
        ("occurred_at", "2026-10-02 14:32:05"),  # missing timezone
        ("camera_id", ""),  # empty min_length=1
        ("event_id", ""),
    ],
)
def test_invalid_contract_fields(field, value):
    with pytest.raises(ValidationError):
        SafetyEventCreate.model_validate({**SAMPLE_EVENT, field: value})


def test_cross_field_validation_violation_requires_details():
    # is_violation=True but missing violation_type
    with pytest.raises(ValidationError) as exc:
        SafetyEventCreate.model_validate({**SAMPLE_EVENT, "violation_type": None})
    assert "violation_type is required" in str(exc.value)

    # is_violation=True but missing severity
    with pytest.raises(ValidationError) as exc:
        SafetyEventCreate.model_validate({**SAMPLE_EVENT, "severity": None})
    assert "severity is required" in str(exc.value)

    # is_violation=False but violation_type is provided
    normal = {
        "event_id": "evt_norm_01",
        "event_type": "person_detected",
        "occurred_at": "2026-10-02T14:32:05Z",
        "camera_id": "CAM-01",
        "is_violation": False,
        "violation_type": "missing_helmet",
        "severity": None,
    }
    with pytest.raises(ValidationError) as exc:
        SafetyEventCreate.model_validate(normal)
    assert "violation_type must be null" in str(exc.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"is_violation": False},
        {"event_type": "person_detected"},
        {"event_type": "zone_entry"},
    ],
)
def test_event_type_and_violation_flag_must_agree(changes):
    with pytest.raises(ValidationError, match="is_violation must match event_type"):
        SafetyEventCreate.model_validate({**SAMPLE_EVENT, **changes})


def test_violation_flag_is_required():
    payload = {key: value for key, value in SAMPLE_EVENT.items() if key != "is_violation"}
    with pytest.raises(ValidationError):
        SafetyEventCreate.model_validate(payload)


def test_inclusive_boundaries():
    valid_zero = SafetyEventCreate.model_validate({
        **SAMPLE_EVENT,
        "confidence": 0.0,
    })
    assert valid_zero.confidence == 0.0

    valid_one = SafetyEventCreate.model_validate({
        **SAMPLE_EVENT,
        "confidence": 1.0,
    })
    assert valid_one.confidence == 1.0

    valid_null_confidence = SafetyEventCreate.model_validate({
        **SAMPLE_EVENT,
        "confidence": None,
    })
    assert valid_null_confidence.confidence is None
