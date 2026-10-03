from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app


class FakeSession:
    def __init__(self, error=None):
        self.error = error

    def execute(self, _query):
        if self.error:
            raise self.error
        return 1


def test_health_endpoint_ready():
    app.dependency_overrides[get_session] = lambda: FakeSession()
    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": "ok"}
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_unavailable_does_not_expose_error():
    app.dependency_overrides[get_session] = lambda: FakeSession(
        RuntimeError("secret database connection detail")
    )
    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "degraded", "database": "unavailable"}
    finally:
        app.dependency_overrides.clear()
