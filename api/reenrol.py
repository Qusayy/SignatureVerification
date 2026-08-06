"""Re-embed every stored specimen after a model or preprocessing change.

**Why this exists.** Stored embeddings are produced by a specific model reading
canvases produced by a specific preprocessing pipeline. Change either and every
``ReferenceSignature.embedding`` in the database becomes meaningless — but it
does not become *obviously* meaningless. Scores keep coming back in the normal
0-100 range, bands keep looking plausible, and nothing errors. That is the
dangerous part: silent, confident, wrong.

This is the most consequential maintenance operation in the system, so it is a
tool rather than a paragraph in a runbook.

Usage::

    # Check whether anything is stale, change nothing
    python -m api.reenrol --check

    # Re-embed everything with the currently configured model
    python -m api.reenrol --apply

The raw specimen images are re-read from object storage, so this reproduces the
full chain — preprocess, embed, refresh cohort statistics — exactly as
enrolment originally did.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from sqlalchemy import select

from api.db import get_sessionmaker, init_db
from api.models.tables import Customer, CustomerEnrolment
from api.services.inference import get_service
from api.services.storage import get_store


def _current_version(service) -> str:
    return service.verifier.model_version if service.verifier else ""


def check(session, service) -> dict:
    """Report which customers hold embeddings from a different model version."""
    current = _current_version(service)
    rows = session.execute(select(CustomerEnrolment)).scalars().all()
    total_customers = session.execute(select(Customer)).scalars().all()
    stale = [e for e in rows if e.model_version != current]

    return {
        "current_model_version": current,
        "customers": len(total_customers),
        "customers_with_cached_enrolment": len(rows),
        "stale_enrolments": len(stale),
        "customers_without_enrolment": len(total_customers) - len(rows),
        "stale_versions": sorted({e.model_version for e in stale}),
    }


def reenrol(session, service, store, *, dry_run: bool = False) -> dict:
    """Recompute every stored specimen embedding and every enrolment statistic."""
    customers = session.execute(select(Customer)).scalars().all()
    current = _current_version(service)

    updated_references = 0
    updated_customers = 0
    failures: list[str] = []

    for customer in customers:
        if not customer.references:
            continue

        embeddings: list[np.ndarray] = []
        for reference in customer.references:
            try:
                image = store.get_image(reference.image_key)
                canvas, embedding = service.embed_reference(image)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append(f"{customer.customer_number}/{reference.id}: {exc}")
                continue

            embeddings.append(embedding)
            if not dry_run:
                # The canvas changes too when preprocessing changes, and it
                # drives the overlay and the copy-detection check.
                reference.canvas_key = store.put_image(canvas, prefix=f"canvases/{customer.id}")
                reference.embedding = [float(v) for v in embedding]
            updated_references += 1

        if not embeddings:
            continue

        stats = service.enrolment_stats(np.vstack(embeddings))
        if not dry_run:
            if stats is not None:
                enrolment = customer.enrolment or CustomerEnrolment(customer_id=customer.id)
                enrolment.cohort_mean = stats.mean
                enrolment.cohort_std = stats.std
                enrolment.n_references = len(embeddings)
                enrolment.model_version = current
                session.add(enrolment)
            elif customer.enrolment is not None:
                # No cohort available any more; drop the cached statistics
                # rather than leave stale ones that look valid.
                session.delete(customer.enrolment)
        updated_customers += 1

    return {
        "model_version": current,
        "customers_updated": updated_customers,
        "references_re_embedded": updated_references,
        "failures": failures,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Report staleness, change nothing")
    group.add_argument("--apply", action="store_true", help="Re-embed and commit")
    group.add_argument("--dry-run", action="store_true", help="Do the work, discard the result")
    args = parser.parse_args()

    init_db()
    service = get_service()
    if not service.is_ready:
        raise SystemExit(service.load_error or "Model is not loaded")

    session = get_sessionmaker()()
    try:
        if args.check:
            import json

            report = check(session, service)
            print(json.dumps(report, indent=2))
            if report["stale_enrolments"]:
                print(
                    f"\n{report['stale_enrolments']} customer(s) hold embeddings from another "
                    "model version. Their scores are not meaningful. Run --apply.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            print("\nAll enrolments match the loaded model.")
            return

        import json

        result = reenrol(session, service, get_store(), dry_run=args.dry_run)
        if args.apply:
            session.commit()
        print(json.dumps(result, indent=2))
        if result["failures"]:
            raise SystemExit(f"{len(result['failures'])} specimen(s) could not be re-embedded")
    finally:
        session.close()


if __name__ == "__main__":
    main()
