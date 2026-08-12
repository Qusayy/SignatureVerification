"""Verification and the employee's decision.

The two are separate endpoints on purpose. ``POST /api/verify`` produces
advice; ``POST /api/verify/{id}/decision`` records what the human did with it.
Nothing here ever accepts or rejects a signature on its own.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import commit, get_session
from api.models.tables import (
    Customer,
    Employee,
    EmployeeDecision,
    VerificationEvent,
)
from api.schemas import (
    BoundingBox,
    ComparisonOut,
    DecisionIn,
    DecisionOut,
    DetectionOut,
    PipelineStageOut,
    ScoreDiagnosticsOut,
    VerificationOut,
)
from api.security.auth import current_employee
from api.services.images import read_upload
from api.services.inference import ModelNotLoaded, get_service
from api.services.storage import get_store
from ml.preprocess.pipeline import BlankSignatureError
from ml.scoring.verifier import EnrolmentBundle
from ml.scoring.znorm import CohortStats

router = APIRouter(prefix="/api/verify", tags=["verification"])


def _stage_url(store, customer_id: str, key: str, image, invert: bool) -> str:
    """Persist one trace image and return the URL that serves it.

    Trace images go through the same encrypted store as everything else. They
    are derived from a biometric image and are just as identifying as the
    original, so they get the same protection rather than being written
    somewhere convenient.
    """
    stored = store.put_image(image, prefix=f"stages/{customer_id}")
    return f"/api/images/{stored}" + ("?invert=1" if invert else "")


def _load_enrolment(session: Session, customer: Customer, store) -> EnrolmentBundle:
    if not customer.references:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Customer {customer.customer_number} has no specimen signature on file. "
            "Enrol at least one before verifying.",
        )

    embeddings = np.asarray([r.embedding for r in customer.references], dtype=np.float64)
    stats = None
    reference_mean = None
    if customer.enrolment:
        if customer.enrolment.cohort_std:
            stats = CohortStats(customer.enrolment.cohort_mean, customer.enrolment.cohort_std)
        # A stored 0.0 means "never computed" — an enrolment written before this
        # was cached. Falling back to None recomputes it rather than silently
        # scoring as though the customer's specimens agreed perfectly.
        reference_mean = customer.enrolment.intra_reference_mean or None

    canvases = []
    for reference in customer.references:
        try:
            canvases.append(store.get_image(reference.canvas_key))
        except (FileNotFoundError, ValueError):
            # A missing canvas costs the overlay and the copy check, but the
            # score itself only needs the embedding, so degrade rather than
            # fail the whole verification.
            continue

    return EnrolmentBundle(
        signer_id=customer.customer_number,
        embeddings=embeddings,
        cohort_stats=stats,
        canvases=canvases,
        reference_mean=reference_mean,
    )


@router.post("", response_model=VerificationOut)
async def verify_signature(
    customer_number: str = Form(...),
    file: UploadFile = File(..., description="Scanned form or cropped signature"),
    is_full_page: bool = Form(True),
    bbox_x: int | None = Form(None),
    bbox_y: int | None = Form(None),
    bbox_width: int | None = Form(None),
    bbox_height: int | None = Form(None),
    explain: bool = Form(True),
    session: Session = Depends(get_session),
    employee: Employee = Depends(current_employee),
) -> VerificationOut:
    """Score a captured signature against a customer's stored specimens.

    Supplying a bounding box overrides automatic detection — an employee
    correcting the crop always wins over the detector.

    ``explain`` captures every intermediate stage of the pipeline for the visual
    replay. It costs a dozen extra image writes per verification, so a caller
    that only wants the number should turn it off.
    """
    started = time.perf_counter()

    customer = session.execute(
        select(Customer).where(Customer.customer_number == customer_number)
    ).scalar_one_or_none()
    if customer is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No customer {customer_number}")

    store = get_store()
    service = get_service()
    image = await read_upload(file)
    enrolment = _load_enrolment(session, customer, store)

    bbox = None
    if None not in (bbox_x, bbox_y, bbox_width, bbox_height):
        bbox = (int(bbox_x), int(bbox_y), int(bbox_width), int(bbox_height))  # type: ignore[arg-type]

    try:
        output = service.verify(
            image, enrolment, bbox=bbox, is_full_page=is_full_page, explain=explain
        )
    except ModelNotLoaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except BlankSignatureError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"No signature detected in the captured area — rescan rather than score this. ({exc})",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    result = output.result
    latency_ms = int((time.perf_counter() - started) * 1000)

    page_key = store.put_image(image, prefix=f"pages/{customer.id}") if is_full_page else None
    crop_key = store.put_image(output.crop, prefix=f"crops/{customer.id}")
    overlay_key = store.put_image(output.overlay, prefix=f"overlays/{customer.id}")

    event = VerificationEvent(
        customer_id=customer.id,
        employee_id=employee.id,
        score=result.score,
        band=result.band.value,
        normalized_score=result.normalized_score,
        calibrated=result.calibrated,
        suspected_copy=result.suspected_copy,
        n_references=result.comparison.n_references,
        page_key=page_key,
        crop_key=crop_key,
        overlay_key=overlay_key,
        detection=output.detection.to_dict() if output.detection else None,
        comparison=result.comparison.to_dict(),
        warnings=result.warnings,
        reason=output.reason,
        model_version=result.model_version,
        latency_ms=latency_ms,
    )
    session.add(event)
    # Commit before responding, not in dependency teardown. The client gets
    # event_id in this response and may POST a decision against it immediately;
    # if the row is not durable yet, that decision 404s.
    commit(session)

    stages = [
        PipelineStageOut(
            key=stage.key,
            title=stage.title,
            caption=stage.caption,
            kind=stage.kind,
            image_url=(
                _stage_url(store, customer.id, stage.key, stage.image, stage.invert_for_display)
                if stage.image is not None
                else None
            ),
            metrics=stage.metrics,
        )
        for stage in output.stages
    ]

    detection_out = None
    if output.detection:
        x, y, w, h = output.detection.bbox
        detection_out = DetectionOut(
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            confidence=output.detection.confidence,
            method=output.detection.method,
        )

    return VerificationOut(
        event_id=event.id,
        score=result.score,
        band=result.band.value,  # type: ignore[arg-type]
        guidance=result.guidance,
        reason=output.reason,
        comparison=ComparisonOut(**result.comparison.to_dict()),
        detection=detection_out,
        warnings=result.warnings,
        suspected_copy=result.suspected_copy,
        calibrated=result.calibrated,
        model_version=result.model_version,
        latency_ms=latency_ms,
        crop_url=f"/api/images/{crop_key}",
        overlay_url=f"/api/images/{overlay_key}",
        page_url=f"/api/images/{page_key}" if page_key else None,
        reference_urls=[f"/api/images/{r.canvas_key}?invert=1" for r in customer.references],
        stages=stages,
        diagnostics=ScoreDiagnosticsOut(**output.diagnostics)
        if output.diagnostics
        else None,
    )


@router.post("/{event_id}/decision", response_model=DecisionOut, status_code=status.HTTP_201_CREATED)
def record_decision(
    event_id: str,
    payload: DecisionIn,
    session: Session = Depends(get_session),
    employee: Employee = Depends(current_employee),
) -> DecisionOut:
    """Record the employee's judgement against a verification.

    Append-only: a decision cannot be changed once recorded. If an employee
    needs to revise one, re-run the verification and decide again, leaving both
    records in the audit trail.
    """
    event = session.get(VerificationEvent, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such verification event")
    if event.decision is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A decision is already recorded for this verification. Re-verify to decide again; "
            "the audit trail is append-only.",
        )

    # Agreement is only meaningful where the advice was not "inconclusive".
    agreed: bool | None = None
    if event.band == "green":
        agreed = payload.decision == "accept"
    elif event.band == "red":
        agreed = payload.decision == "reject"

    decision = EmployeeDecision(
        event_id=event.id,
        employee_id=employee.id,
        decision=payload.decision,
        note=payload.note,
        agreed_with_model=agreed,
        seconds_to_decide=payload.seconds_to_decide,
    )
    session.add(decision)
    commit(session)

    return DecisionOut(
        id=decision.id,
        event_id=event.id,
        decision=decision.decision,
        note=decision.note,
        agreed_with_model=decision.agreed_with_model,
        created_at=decision.created_at,
    )
