"""Seed and database readiness script for Industrial Safety AI backend."""
import logging
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.entities import Alert, SafetyEvent, Violation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed")


def main() -> None:
    engine = get_engine()
    with Session(engine) as session:
        events_count = session.scalar(select(func.count()).select_from(SafetyEvent)) or 0
        violations_count = session.scalar(select(func.count()).select_from(Violation)) or 0
        alerts_count = session.scalar(select(func.count()).select_from(Alert)) or 0

        logger.info("Database is connected and ready.")
        logger.info("Current records: events=%d, violations=%d, alerts=%d", events_count, violations_count, alerts_count)
        logger.info("Industrial Safety AI backend requires no manual entity pre-registration.")
        logger.info("Upstream pipeline can immediately POST events to /api/v1/events.")


if __name__ == "__main__":
    main()
