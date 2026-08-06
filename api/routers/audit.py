"""Audit trail and pilot measurement.

The agreement summary is the number the pilot exists to produce: how
often the employee and the model reach the same conclusion, and where they
diverge. Disagreements are the interesting cases — each one is either a model
error worth fixing or a human error worth catching, and adjudicating them
produces the most valuable labelled data the project will generate.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import get_session
from api.models.tables import Customer, Employee, EmployeeDecision, VerificationEvent
from api.schemas import AgreementSummary, AuditEventOut
from api.security.auth import current_employee

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/events", response_model=list[AuditEventOut])
def list_events(
    customer_number: str | None = None,
    band: str | None = None,
    disagreements_only: bool = False,
    since: datetime | None = None,
    limit: int = Query(100, le=500),
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> list[AuditEventOut]:
    statement = (
        select(VerificationEvent, Customer, Employee)
        .join(Customer, Customer.id == VerificationEvent.customer_id)
        .join(Employee, Employee.id == VerificationEvent.employee_id)
        .order_by(VerificationEvent.created_at.desc())
        .limit(limit)
    )
    if customer_number:
        statement = statement.where(Customer.customer_number == customer_number)
    if band:
        statement = statement.where(VerificationEvent.band == band)
    if since:
        statement = statement.where(VerificationEvent.created_at >= since)
    if disagreements_only:
        statement = statement.join(
            EmployeeDecision, EmployeeDecision.event_id == VerificationEvent.id
        ).where(EmployeeDecision.agreed_with_model.is_(False))

    out: list[AuditEventOut] = []
    for event, customer, employee in session.execute(statement):
        decision = event.decision
        out.append(
            AuditEventOut(
                id=event.id,
                customer_number=customer.customer_number,
                employee_username=employee.username,
                score=event.score,
                band=event.band,
                suspected_copy=event.suspected_copy,
                n_references=event.n_references,
                warnings=event.warnings or [],
                model_version=event.model_version,
                created_at=event.created_at,
                decision=decision.decision if decision else None,
                decision_note=decision.note if decision else None,
                agreed_with_model=decision.agreed_with_model if decision else None,
            )
        )
    return out


@router.get("/agreement", response_model=AgreementSummary)
def agreement_summary(
    since: datetime | None = None,
    session: Session = Depends(get_session),
    _employee: Employee = Depends(current_employee),
) -> AgreementSummary:
    """How often the employee's decision matched the model's advisory band."""
    statement = select(EmployeeDecision, VerificationEvent).join(
        VerificationEvent, VerificationEvent.id == EmployeeDecision.event_id
    )
    if since:
        statement = statement.where(EmployeeDecision.created_at >= since)

    total = 0
    agreements = 0
    disagreements = 0
    by_band: dict[str, dict[str, int]] = {}

    for decision, event in session.execute(statement):
        total += 1
        bucket = by_band.setdefault(event.band, {"accept": 0, "reject": 0, "disagreements": 0})
        bucket[decision.decision] = bucket.get(decision.decision, 0) + 1
        if decision.agreed_with_model is True:
            agreements += 1
        elif decision.agreed_with_model is False:
            disagreements += 1
            bucket["disagreements"] += 1

    conclusive = agreements + disagreements
    return AgreementSummary(
        total_decisions=total,
        agreements=agreements,
        disagreements=disagreements,
        agreement_rate=(agreements / conclusive) if conclusive else None,
        by_band=by_band,
        note=(
            "Agreement is computed only for green and red advice; amber is inconclusive by "
            "design and is excluded. A high agreement rate alone does not justify automation — "
            "the disagreements must be adjudicated to establish who was right."
        ),
    )
