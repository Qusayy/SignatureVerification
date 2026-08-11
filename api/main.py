"""FastAPI application.

Run locally::

    python -m api.seed          # create a demo employee and enrol customers
    uvicorn api.main:app --reload

The service is advisory. It scores signatures and explains its reasoning; it
never accepts or rejects one. That constraint is asserted in ``/api/health``
and in every verification response so no client can be built against a
different assumption.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import init_db
from api.routers import audit, auth, customers, images, verify
from api.schemas import HealthOut
from api.services.inference import get_service
from api.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sigver.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    init_db()

    warnings = settings.production_warnings()
    if warnings and settings.is_production:
        for warning in warnings:
            logger.error("PRODUCTION MISCONFIGURATION: %s", warning)
    elif warnings:
        for warning in warnings:
            logger.warning("Development default in use: %s", warning)

    get_service()  # load the model at startup so the first operator does not wait
    yield


app = FastAPI(
    title="Signature Verification",
    version="0.1.0",
    description=(
        "Advisory signature verification for counter operations. The system scores a captured "
        "signature against a customer's stored specimens and explains the result. It never "
        "accepts or rejects — the employee always decides, and every decision is logged."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(customers.router)
app.include_router(verify.router)
app.include_router(audit.router)
app.include_router(images.router)


@app.get("/api/health", response_model=HealthOut, tags=["ops"])
def health() -> HealthOut:
    settings = get_settings()
    status_info = get_service().status()
    return HealthOut(
        status="ok" if status_info["model_loaded"] else "degraded",
        model_loaded=status_info["model_loaded"],
        model_version=status_info["model_version"],
        cohort_normalisation=status_info["cohort_normalisation"],
        writer_normalisation=status_info.get("writer_normalisation", False),
        calibrated=status_info["calibrated"],
        advisory_only=settings.advisory_only,
        warnings=settings.production_warnings(),
        error=status_info["error"],
    )
