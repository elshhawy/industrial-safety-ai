from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.models.entities import Severity, ViolationStatus, ViolationType
from app.schemas.violation import ViolationList, ViolationRead
from app.services import violations as violation_service

router = APIRouter(prefix="/api/v1/violations", tags=["Violations"])
DatabaseSession = Annotated[Session, Depends(get_session)]


@router.get("", response_model=ViolationList)
def list_violations(
    session: DatabaseSession,
    violation_type: ViolationType | None = None,
    severity: Severity | None = None,
    camera_id: str | None = None,
    status: ViolationStatus | None = None,
    from_time: datetime | None = Query(default=None, alias="from"),
    to_time: datetime | None = Query(default=None, alias="to"),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """List safety violations with optional filtering and pagination."""
    return violation_service.list_violations(
        session,
        violation_type=violation_type,
        severity=severity,
        camera_id=camera_id,
        status=status,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{violation_id}",
    response_model=ViolationRead,
    responses={404: {"description": "Violation not found"}},
)
def get_violation(violation_id: UUID, session: DatabaseSession):
    """Retrieve one safety violation by its backend UUID."""
    return violation_service.get_violation(session, violation_id)
