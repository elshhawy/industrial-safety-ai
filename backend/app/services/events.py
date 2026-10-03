from datetime import datetime
import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.entities import (
    Alert,
    AlertStatus,
    EventType,
    SafetyEvent,
    Violation,
    ViolationStatus,
)
from app.schemas.event import SafetyEventCreate

logger = logging.getLogger(__name__)


def ingest_event(
    session: Session, payload: SafetyEventCreate
) -> tuple[SafetyEvent, Violation | None, Alert | None]:
    event = SafetyEvent(
        event_id=payload.event_id,
        event_type=payload.event_type,
        occurred_at=payload.occurred_at,
        camera_id=payload.camera_id,
        track_id=payload.track_id,
        person_id=payload.person_id,
        zone_id=payload.zone_id,
        confidence=payload.confidence,
        is_violation=payload.is_violation,
        violation_type=payload.violation_type,
        severity=payload.severity,
        description=payload.description,
        ppe_details=payload.ppe,
        frame_reference=payload.frame_reference,
        metadata_=payload.metadata,
    )
    session.add(event)

    violation: Violation | None = None
    alert: Alert | None = None

    try:
        session.flush()

        if payload.is_violation:
            desc = payload.description or f"{payload.violation_type} on camera {payload.camera_id}"
            violation = Violation(
                event_id=event.id,
                violation_type=payload.violation_type,  # type: ignore[arg-type]
                severity=payload.severity,  # type: ignore[arg-type]
                camera_id=payload.camera_id,
                description=desc,
                status=ViolationStatus.OPEN,
            )
            session.add(violation)
            session.flush()

            msg = f"Safety alert: {payload.violation_type} ({payload.severity}) on camera {payload.camera_id}"
            alert = Alert(
                violation_id=violation.id,
                message=msg,
                status=AlertStatus.ACTIVE,
            )
            session.add(alert)
            session.flush()

        session.commit()
    except IntegrityError as exc:
        session.rollback()
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint == "events_event_id_key":
            logger.warning("Duplicate event_id received: %s", payload.event_id)
            raise HTTPException(status_code=409, detail="event_id already received") from exc
        logger.error("Database integrity error during event ingestion: %s", exc)
        raise HTTPException(status_code=500, detail="Database integrity error") from exc
    except Exception as exc:
        session.rollback()
        logger.error("Unexpected error during event ingestion: %s", exc)
        raise

    session.refresh(event)
    logger.info(
        "Event ingested: %s (id=%s, violation=%s)",
        event.event_id,
        event.id,
        event.is_violation,
    )
    return event, violation, alert


def get_event(session: Session, event_id: UUID) -> SafetyEvent:
    event = session.get(SafetyEvent, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def list_events(
    session: Session,
    event_type: EventType | None = None,
    camera_id: str | None = None,
    is_violation: bool | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    filters = []
    if event_type is not None:
        filters.append(SafetyEvent.event_type == event_type)
    if camera_id is not None:
        filters.append(SafetyEvent.camera_id == camera_id)
    if is_violation is not None:
        filters.append(SafetyEvent.is_violation == is_violation)
    if from_time is not None:
        filters.append(SafetyEvent.occurred_at >= from_time)
    if to_time is not None:
        filters.append(SafetyEvent.occurred_at <= to_time)

    total = session.scalar(select(func.count()).select_from(SafetyEvent).where(*filters)) or 0
    query = (
        select(SafetyEvent)
        .where(*filters)
        .order_by(SafetyEvent.occurred_at.desc(), SafetyEvent.id.desc())
        .limit(limit)
        .offset(offset)
    )
    items = session.scalars(query).all()
    return {"total": total, "limit": limit, "offset": offset, "items": items}
