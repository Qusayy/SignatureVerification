"""Customer records and specimen signature enrolment."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import commit, get_session
from api.models.tables import Customer, CustomerEnrolment, Employee, ReferenceSignature
from api.schemas import CustomerCreate, CustomerOut, ReferenceOut
from api.security.auth import current_employee
from api.services.images import read_upload
from api.services.inference import ModelNotLoaded, get_service
from api.services.storage import get_store
from ml.preprocess.pipeline import BlankSignatureError

router = APIRouter(prefix="/api/customers", tags=["customers"])


def _to_out(customer: Customer, *, with_references: bool = False) -> CustomerOut:
    references = []
    if with_references:
        references = [
            ReferenceOut(
                id=r.id,
                image_url=f"/api/images/{r.image_key}",
                canvas_url=f"/api/images/{r.canvas_key}?invert=1",
                captured_at=r.captured_at,
                source=r.source,
                created_at=r.created_at,
            )
            for r in customer.references
        ]
    return CustomerOut(
        id=customer.id,
        customer_number=customer.customer_number,
        full_name=customer.full_name,
        script=customer.script,
        n_references=len(customer.references),
        enrolled=customer.enrolment is not None,
        references=references,
    )


@router.post("", response_model=CustomerOut, status_code=status.HTTP_201_CREATED)
def create_customer(
    payload: CustomerCreate,
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> CustomerOut:
    existing = session.execute(
        select(Customer).where(Customer.customer_number == payload.customer_number)
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Customer number already exists")

    customer = Customer(**payload.model_dump())
    session.add(customer)
    # Durable before responding: the client uses customer_number immediately.
    commit(session)
    return _to_out(customer)


@router.get("", response_model=list[CustomerOut])
def list_customers(
    q: str = "",
    limit: int = 50,
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> list[CustomerOut]:
    statement = select(Customer).order_by(Customer.customer_number).limit(min(limit, 200))
    if q:
        pattern = f"%{q}%"
        statement = statement.where(
            Customer.customer_number.ilike(pattern) | Customer.full_name.ilike(pattern)
        )
    return [_to_out(c) for c in session.execute(statement).scalars()]


def _get_customer(session: Session, customer_number: str) -> Customer:
    customer = session.execute(
        select(Customer).where(Customer.customer_number == customer_number)
    ).scalar_one_or_none()
    if customer is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No customer {customer_number}")
    return customer


@router.get("/{customer_number}", response_model=CustomerOut)
def get_customer(
    customer_number: str,
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> CustomerOut:
    return _to_out(_get_customer(session, customer_number), with_references=True)


@router.post(
    "/{customer_number}/references",
    response_model=CustomerOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_references(
    customer_number: str,
    files: list[UploadFile] = File(..., description="Specimen signature images"),
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> CustomerOut:
    """Enrol one or more specimen signatures for a customer.

    Each image is preprocessed and embedded once here, so verification at the
    desk never repeats that work. Per-customer cohort statistics are refreshed
    afterwards, because they depend on the full specimen set.
    """
    customer = _get_customer(session, customer_number)
    service = get_service()
    store = get_store()

    try:
        for upload in files:
            image = await read_upload(upload)
            try:
                canvas, embedding = service.embed_reference(image)
            except BlankSignatureError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"{upload.filename}: {exc}",
                ) from exc

            session.add(
                ReferenceSignature(
                    customer_id=customer.id,
                    image_key=store.put_image(image, prefix=f"references/{customer.id}"),
                    canvas_key=store.put_image(canvas, prefix=f"canvases/{customer.id}"),
                    embedding=[float(v) for v in embedding],
                )
            )
    except ModelNotLoaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    session.flush()
    session.refresh(customer)
    _refresh_enrolment(session, customer, service)
    session.flush()
    session.refresh(customer)
    return _to_out(customer, with_references=True)


@router.delete("/{customer_number}/references/{reference_id}", response_model=CustomerOut)
def delete_reference(
    customer_number: str,
    reference_id: str,
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> CustomerOut:
    customer = _get_customer(session, customer_number)
    reference = session.get(ReferenceSignature, reference_id)
    if reference is None or reference.customer_id != customer.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such reference signature")

    session.delete(reference)
    session.flush()
    session.refresh(customer)
    _refresh_enrolment(session, customer, get_service())
    commit(session)
    session.refresh(customer)
    return _to_out(customer, with_references=True)


def _refresh_enrolment(session: Session, customer: Customer, service) -> None:
    """Recompute cached per-customer scoring state after a specimen change."""
    import numpy as np

    if not customer.references:
        if customer.enrolment:
            session.delete(customer.enrolment)
        return

    embeddings = np.asarray([r.embedding for r in customer.references], dtype=np.float64)
    try:
        stats = service.enrolment_stats(embeddings)
    except ModelNotLoaded:
        return
    if stats is None:
        # No cohort available; nothing to cache. Verification falls back to raw
        # similarity and flags the response accordingly.
        return

    enrolment = customer.enrolment or CustomerEnrolment(customer_id=customer.id)
    enrolment.cohort_mean = stats.mean
    enrolment.cohort_std = stats.std
    enrolment.n_references = len(customer.references)
    enrolment.model_version = service.verifier.model_version if service.verifier else ""
    session.add(enrolment)
