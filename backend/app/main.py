"""FastAPI application entry point.

Milestone 0 scaffold. This module currently holds nothing but the app object and
a single placeholder health endpoint, so that `uvicorn app.main:app` starts and
serves something verifiable (ROADMAP.md Milestone 0 acceptance criteria).

At Milestone 6 this file becomes the app factory described in ARCHITECTURE.md
Section 4: CORS configuration from `app/config.py`, startup model load via
`app/ml/model_loader.py`, rate-limit middleware (`app/security/rate_limit.py`),
and router registration for `app/routers/{analyze,explain,meta}.py`. The health
endpoint below moves to `routers/meta.py` at that point, per the endpoint-to-module
map in BACKEND_SPECIFICATION.md Section 6.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

API_V1_PREFIX = "/api/v1"

app = FastAPI(
    title="Urdu Misinformation Audit API",
    description=(
        "Inference and explainability API for the Urdu misinformation audit platform. "
        "Scaffold only — no analysis endpoints are implemented yet."
    ),
    version="0.0.0",
)


class HealthResponse(BaseModel):
    """Response body for the health endpoint (BACKEND_SPECIFICATION.md Section 3)."""

    status: Literal["ok"]
    model_loaded: bool


@app.get(f"{API_V1_PREFIX}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe.

    Placeholder: always returns 200 with ``model_loaded=False``, because no model
    is loaded at Milestone 0 — there is no checkpoint yet. The real readiness
    semantics from BACKEND_SPECIFICATION.md Section 3 (200 with ``model_loaded=True``
    when the startup load succeeded, 503 when it failed) are implemented at
    Milestone 6 alongside `app/ml/model_loader.py`.
    """
    return HealthResponse(status="ok", model_loaded=False)
