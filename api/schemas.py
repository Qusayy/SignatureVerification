"""Request and response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    employee: EmployeeOut


class EmployeeOut(BaseModel):
    id: str
    username: str
    full_name: str
    location: str

    model_config = {"from_attributes": True}


# --------------------------------------------------------------------------
# Customers
# --------------------------------------------------------------------------


class CustomerCreate(BaseModel):
    customer_number: str = Field(min_length=1, max_length=64)
    full_name: str = ""
    script: Literal["latin", "arabic", "mixed", "unknown"] = "unknown"


class ReferenceOut(BaseModel):
    id: str
    image_url: str
    canvas_url: str
    captured_at: datetime | None
    source: str
    created_at: datetime


class CustomerOut(BaseModel):
    id: str
    customer_number: str
    full_name: str
    script: str
    n_references: int
    enrolled: bool
    references: list[ReferenceOut] = []


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectionOut(BaseModel):
    bbox: BoundingBox
    confidence: float
    method: str


class ComparisonOut(BaseModel):
    # NOTE: `raw` is not a similarity. When writer normalisation applies it is
    # the combined similarity minus `intra_reference_mean`, so it sits near
    # zero for a genuine signature and goes negative for one less consistent
    # than the customer's own specimens. The similarities are reported
    # alongside it precisely because the two are easy to confuse.
    raw: float
    max_similarity: float
    mean_similarity: float
    min_similarity: float
    per_reference: list[float]
    n_references: int
    single_reference: bool
    # The baseline `raw` is measured against. Without it a client cannot
    # explain why an 89% match scored 50, and the panel reads as broken.
    intra_reference_mean: float = 0.0
    writer_normalised: bool = False
    # "own" | "population" | "none" — which baseline produced `raw`. A client
    # must not present a population baseline as though it measured this
    # customer.
    baseline_source: str = "none"


class PipelineStageOut(BaseModel):
    """One step of the pipeline, for the visual replay.

    Purely explanatory. Nothing here feeds back into the score — it is the same
    computation, recorded as it happens.
    """

    key: str
    title: str
    caption: str
    kind: Literal["image", "vector", "compare", "score"]
    image_url: str | None = None
    metrics: dict[str, Any] = {}


class VerificationOut(BaseModel):
    """The advisory result shown to the employee.

    ``advisory_only`` is always true and is included in every response so no
    client can be built on the assumption that the system decides.
    """

    event_id: str
    score: float = Field(ge=0, le=100, description="Calibrated confidence, 0-100")
    band: Literal["green", "amber", "red"]
    guidance: str
    reason: str
    comparison: ComparisonOut
    detection: DetectionOut | None
    warnings: list[str]
    suspected_copy: bool
    calibrated: bool
    model_version: str
    latency_ms: int
    crop_url: str
    overlay_url: str
    page_url: str | None
    reference_urls: list[str]
    stages: list[PipelineStageOut] = []
    advisory_only: Literal[True] = True


class DecisionIn(BaseModel):
    decision: Literal["accept", "reject"]
    note: str = ""
    seconds_to_decide: int | None = None


class DecisionOut(BaseModel):
    id: str
    event_id: str
    decision: str
    note: str
    agreed_with_model: bool | None
    created_at: datetime


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


class AuditEventOut(BaseModel):
    id: str
    customer_number: str
    employee_username: str
    score: float
    band: str
    suspected_copy: bool
    n_references: int
    warnings: list[str]
    model_version: str
    created_at: datetime
    decision: str | None
    decision_note: str | None
    agreed_with_model: bool | None


class AgreementSummary(BaseModel):
    """The pilot's headline measurement: how often employee and model agree."""

    total_decisions: int
    agreements: int
    disagreements: int
    agreement_rate: float | None
    by_band: dict[str, dict[str, int]]
    note: str


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    cohort_normalisation: bool
    writer_normalisation: bool = False
    calibrated: bool
    advisory_only: bool
    warnings: list[str]
    error: str | None = None
