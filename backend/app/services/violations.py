from datetime import datetime
import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.entities import Severity, Violation, ViolationStatus, ViolationType

logger = logging.getLogger(__name__)


def get_violation(session: Session, violation_id: UUID) -> Violation:
    violation = session.get(Violation, violation_id)
    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation


def list_violations(
    session: Session,
    violation_type: ViolationType | None = None,
    severity: Severity | None = None,
    camera_id: str | None = None,
    status: ViolationStatus | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    filters = []
    if violation_type is not None:
        filters.append(Violation.violation_type == violation_type)
    if severity is not None:
        filters.append(Violation.severity == severity)
    if camera_id is not None:
        filters.append(Violation.camera_id == camera_id)
    if status is not None:
        filters.append(Violation.status == status)
    if from_time is not None:
        filters.append(Violation.created_at >= from_time)
    if to_time is not None:
        filters.append(Violation.created_at <= to_time)

    total = session.scalar(select(func.count()).select_from(Violation).where(*filters)) or 0
    query = (
        select(Violation)
        .where(*filters)
        .order_by(Violation.created_at.desc(), Violation.id.desc())
        .limit(limit)
        .offset(offset)
    )
    items = session.scalars(query).all()
    return {"total": total, "limit": limit, "offset": offset, "items": items}
