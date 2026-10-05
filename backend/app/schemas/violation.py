from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict

from app.models.entities import Severity, ViolationStatus, ViolationType


class ViolationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_id: UUID
    violation_type: ViolationType
    severity: Severity
    camera_id: str
    description: str | None
    status: ViolationStatus
    created_at: AwareDatetime


class ViolationList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ViolationRead]
