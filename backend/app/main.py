import logging
import os
from pathlib import Path

from dotenv import dotenv_values

from fastapi import FastAPI

from app.api.routes.alerts import router as alerts_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.violations import router as violations_router
log_level = os.environ.get("LOG_LEVEL") or dotenv_values(
    Path(__file__).resolve().parents[1] / ".env"
).get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Industrial Safety AI System — Backend API",
    version="1.0.0",
    description=(
        "Backend platform for real-time industrial safety event ingestion, "
        "violation persistence, and alert generation. Consumed by AI Edge pipelines, "
        "MLOps, and downstream monitoring dashboards."
    ),
)

app.include_router(health_router)
app.include_router(events_router)
app.include_router(violations_router)
app.include_router(alerts_router)
