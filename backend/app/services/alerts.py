import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.entities import Alert, AlertStatus

logger = logging.getLogger(__name__)


def get_alert(session: Session, alert_id: UUID) -> Alert:
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


def list_alerts(
    session: Session,
    status: AlertStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    filters = []
    if status is not None:
        filters.append(Alert.status == status)

    total = session.scalar(select(func.count()).select_from(Alert).where(*filters)) or 0
    query = (
        select(Alert)
        .where(*filters)
        .order_by(Alert.created_at.desc(), Alert.id.desc())
        .limit(limit)
        .offset(offset)
    )
    items = session.scalars(query).all()
    return {"total": total, "limit": limit, "offset": offset, "items": items}
