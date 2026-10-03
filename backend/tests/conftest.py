import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_session
from app.main import app


@pytest.fixture(scope="session")
def postgres_engine():
    url = os.environ.get("TEST_DATABASE_URL") or dotenv_values(
        Path(__file__).resolve().parents[1] / ".env"
    ).get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated database ending in _test")
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql" or not (parsed.database or "").endswith("_test"):
        pytest.fail("TEST_DATABASE_URL must be PostgreSQL with a database name ending in _test")
    development_url = get_settings().database_url
    development = make_url(development_url)
    if (parsed.host, parsed.port, parsed.database) == (development.host, development.port, development.database):
        pytest.fail("Test and development databases must be different")
    previous = os.environ.get("DATABASE_URL")
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    try:
        os.environ["DATABASE_URL"] = url
        get_settings.cache_clear()
        command.upgrade(config, "head")
        command.check(config)
        engine = create_engine(url)
        yield engine
        engine.dispose()
        command.downgrade(config, "base")
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        get_settings.cache_clear()


@pytest.fixture
def db_session(postgres_engine):
    with postgres_engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_session] = lambda: db_session
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
