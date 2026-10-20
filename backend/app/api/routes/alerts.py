from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.models.entities import AlertStatus
from app.schemas.alert import AlertList, AlertRead
from app.services import alerts as alert_service

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])
DatabaseSession = Annotated[Session, Depends(get_session)]


@router.get("", response_model=AlertList)
def list_alerts(
    session: DatabaseSession,
    status: AlertStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """List safety alerts with optional status filter and pagination."""
    return alert_service.list_alerts(session, status=status, limit=limit, offset=offset)


@router.get(
    "/{alert_id}",
    response_model=AlertRead,
    responses={404: {"description": "Alert not found"}},
)
def get_alert(alert_id: UUID, session: DatabaseSession):
    """Retrieve one safety alert by its backend UUID."""
    return alert_service.get_alert(session, alert_id)
