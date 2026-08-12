"""Shared pytest fixtures for the backend suite (TESTING_STRATEGY.md Section 1).

Milestone 0 provides only the TestClient fixture. The mocked small-model fixture
described in TESTING_STRATEGY.md is added at Milestone 6, when there is a model
loader for it to stand in for.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient bound to the FastAPI app (DECISION_REGISTER.md E13)."""
    with TestClient(app) as test_client:
        yield test_client
