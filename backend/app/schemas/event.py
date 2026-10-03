from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.entities import EventType, Severity, ViolationType

SAMPLE_EVENT = {
    "event_id": "evt_CAM01_20261002_143205_001",
    "event_type": "ppe_violation",
    "occurred_at": "2026-10-02T14:32:05Z",
    "camera_id": "CAM-01",
    "track_id": "42",
    "person_id": None,
    "zone_id": "ZONE-A",
    "confidence": 0.93,
    "is_violation": True,
    "violation_type": "missing_helmet",
    "severity": "high",
    "description": "Worker entered restricted zone without helmet",
    "ppe": {
        "helmet": False,
        "vest": True,
        "gloves": True,
        "goggles": False,
    },
    "frame_reference": "frames/CAM-01/evt_001.jpg",
    "metadata": {},
}


class SafetyEventCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={"examples": [SAMPLE_EVENT]},
    )

    event_id: str = Field(min_length=1, max_length=128, description="Stable unique identifier retained across retries.")
    event_type: EventType
    occurred_at: AwareDatetime
    camera_id: str = Field(min_length=1, max_length=64, description="Camera identifier.")
    track_id: str | None = Field(default=None, max_length=64)
    person_id: str | None = Field(default=None, max_length=64)
    zone_id: str | None = Field(default=None, max_length=64)
    confidence: float | None = Field(default=None, ge=0, le=1)
    is_violation: bool
    violation_type: ViolationType | None = None
    severity: Severity | None = None
    description: str | None = Field(default=None, max_length=512)
    ppe: dict[str, Any] | None = None
    frame_reference: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] | None = None

    @field_validator("occurred_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_violation_fields(self) -> "SafetyEventCreate":
        violation_events = {EventType.PPE_VIOLATION, EventType.DANGER_ZONE_VIOLATION}
        if (self.event_type in violation_events) != self.is_violation:
            raise ValueError("is_violation must match event_type")
        if self.is_violation:
            if self.violation_type is None:
                raise ValueError("violation_type is required when is_violation is true")
            if self.severity is None:
                raise ValueError("severity is required when is_violation is true")
        else:
            if self.violation_type is not None:
                raise ValueError("violation_type must be null when is_violation is false")
            if self.severity is not None:
                raise ValueError("severity must be null when is_violation is false")
        return self


class SafetyEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_id: str
    event_type: EventType
    occurred_at: AwareDatetime
    camera_id: str
    track_id: str | None
    person_id: str | None
    zone_id: str | None
    confidence: float | None
    is_violation: bool
    violation_type: ViolationType | None
    severity: Severity | None
    description: str | None
    ppe_details: dict[str, Any] | None = None
    frame_reference: str | None
    created_at: AwareDatetime


class SafetyEventList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[SafetyEventRead]


class SafetyEventIngestResponse(BaseModel):
    status: str = "accepted"
    event_id: str
    backend_id: UUID
    violation_created: bool
    alert_created: bool
