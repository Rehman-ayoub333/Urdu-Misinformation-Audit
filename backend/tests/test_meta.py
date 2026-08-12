"""Tests for the meta endpoints (TESTING_STRATEGY.md Section 1).

Milestone 0 covers only the placeholder health endpoint. The full coverage this
file is specified to have — `/health` returning 503 when the model failed to load,
`/examples` returning the static fixture, `/model-info` returning all required
fields — is added at Milestone 6 with `app/routers/meta.py`.
"""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """The placeholder health endpoint is reachable and returns 200."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_health_reports_no_model_loaded_at_scaffold_stage(client: TestClient) -> None:
    """No checkpoint exists yet, so readiness must not claim a model is loaded.

    This asserts the scaffold is honest rather than convenient: it fails the moment
    a model load is wired up without updating the health contract, which is exactly
    when BACKEND_SPECIFICATION.md Section 3's real 200/503 semantics must land.
    """
    assert client.get("/api/v1/health").json()["model_loaded"] is False
