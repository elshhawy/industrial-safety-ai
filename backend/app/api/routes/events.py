from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.models.entities import EventType
from app.schemas.event import (
    SafetyEventCreate,
    SafetyEventIngestResponse,
    SafetyEventList,
    SafetyEventRead,
)
from app.services import events as event_service

router = APIRouter(prefix="/api/v1/events", tags=["Events"])
DatabaseSession = Annotated[Session, Depends(get_session)]


@router.post(
    "",
    response_model=SafetyEventIngestResponse,
    status_code=201,
    responses={
        409: {"description": "Duplicate event_id"},
        422: {"description": "Validation error"},
    },
)
def create_event(payload: SafetyEventCreate, session: DatabaseSession):
    """Ingest a structured AI safety event.

    Automatically creates a violation and alert record if is_violation is true.
    """
    event, violation, alert = event_service.ingest_event(session, payload)
    return SafetyEventIngestResponse(
        status="accepted",
        event_id=event.event_id,
        backend_id=event.id,
        violation_created=violation is not None,
        alert_created=alert is not None,
    )


@router.get("", response_model=SafetyEventList)
def list_events(
    session: DatabaseSession,
    event_type: EventType | None = None,
    camera_id: str | None = None,
    is_violation: bool | None = None,
    from_time: datetime | None = Query(default=None, alias="from"),
    to_time: datetime | None = Query(default=None, alias="to"),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """List safety events with optional filtering and pagination."""
    return event_service.list_events(
        session,
        event_type=event_type,
        camera_id=camera_id,
        is_violation=is_violation,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{event_id}",
    response_model=SafetyEventRead,
    responses={404: {"description": "Event not found"}},
)
def get_event(event_id: UUID, session: DatabaseSession):
    """Retrieve one safety event by its backend UUID."""
    return event_service.get_event(session, event_id)
