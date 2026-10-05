from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict

from app.models.entities import AlertStatus


class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    violation_id: UUID
    message: str
    status: AlertStatus
    created_at: AwareDatetime


class AlertList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AlertRead]
