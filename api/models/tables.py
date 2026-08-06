"""Database schema.

Design notes that matter for an organisation:

* :class:`VerificationEvent` and :class:`EmployeeDecision` are **append-only**.
  Nothing in the API updates or deletes them. An audit trail that can be edited
  is not an audit trail, and the whole point of automating this process is that
  every judgement becomes reviewable afterwards.
* The employee's decision is stored separately from the model's score, so the
  two can never be conflated. This also makes the pilot's central question —
  how often do the employee and the model disagree, and who was right —
  answerable with a single query.
* Signature images are never stored in the database. Only encrypted object
  keys are, so the images live in one place with one access policy.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Employee(Base):
    """An operator who uses the system."""

    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160), default="")
    password_hash: Mapped[str] = mapped_column(String(256))
    location: Mapped[str] = mapped_column(String(64), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Customer(Base):
    """An organisation customer whose specimen signatures are on file."""

    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    customer_number: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160), default="")
    script: Mapped[str] = mapped_column(String(16), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    references: Mapped[list[ReferenceSignature]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    enrolment: Mapped[CustomerEnrolment | None] = relationship(
        back_populates="customer", uselist=False, cascade="all, delete-orphan"
    )


class ReferenceSignature(Base):
    """One stored specimen signature for a customer."""

    __tablename__ = "reference_signatures"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    # Object-storage keys. The raw image and its preprocessed canvas.
    image_key: Mapped[str] = mapped_column(String(256))
    canvas_key: Mapped[str] = mapped_column(String(256))
    # Embedding stored as JSON so the schema stays portable across SQLite and
    # PostgreSQL. A pgvector column is the obvious upgrade at production scale.
    embedding: Mapped[list] = mapped_column(JSON)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="enrolment")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    customer: Mapped[Customer] = relationship(back_populates="references")


class CustomerEnrolment(Base):
    """Cached per-customer scoring state, refreshed when specimens change.

    Holds the Z-norm statistics, which depend only on the customer's specimens
    and the background cohort. Recomputing them on every verification would
    mean scoring the whole cohort at the operator's desk for no benefit.
    """

    __tablename__ = "customer_enrolments"

    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), primary_key=True)
    cohort_mean: Mapped[float] = mapped_column(Float)
    cohort_std: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), default="")
    n_references: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    customer: Mapped[Customer] = relationship(back_populates="enrolment")


class VerificationEvent(Base):
    """One verification performed at a desk. Append-only."""

    __tablename__ = "verification_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)

    score: Mapped[float] = mapped_column(Float)
    band: Mapped[str] = mapped_column(String(8))
    normalized_score: Mapped[float] = mapped_column(Float)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=True)
    suspected_copy: Mapped[bool] = mapped_column(Boolean, default=False)
    n_references: Mapped[int] = mapped_column(Integer, default=0)

    # Uploaded page, the crop that was scored, and the generated overlay.
    page_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    crop_key: Mapped[str] = mapped_column(String(256))
    overlay_key: Mapped[str | None] = mapped_column(String(256), nullable=True)

    detection: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    comparison: Mapped[dict] = mapped_column(JSON)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    reason: Mapped[str] = mapped_column(Text, default="")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    decision: Mapped[EmployeeDecision | None] = relationship(
        back_populates="event", uselist=False
    )


class EmployeeDecision(Base):
    """The employee's judgement. Append-only, and never made by the system."""

    __tablename__ = "employee_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("verification_events.id"), unique=True, index=True
    )
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)
    decision: Mapped[str] = mapped_column(String(16))  # "accept" | "reject"
    note: Mapped[str] = mapped_column(Text, default="")
    # Whether the employee's decision matched the model's advisory band. The
    # pilot's key measurement, denormalised so the disagreement query is cheap.
    agreed_with_model: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    seconds_to_decide: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    event: Mapped[VerificationEvent] = relationship(back_populates="decision")


Index("ix_events_customer_created", VerificationEvent.customer_id, VerificationEvent.created_at)
Index("ix_decisions_agreement", EmployeeDecision.agreed_with_model, EmployeeDecision.created_at)
