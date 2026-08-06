"""Seed the database for a demo.

Creates a operator and enrols customers using signers from the **test**
split of the manifest — writers the model has never seen, which is the honest
way to demo a writer-independent system.

It also writes a folder of query images the demonstrator can drag into the UI,
each labelled genuine or forgery, so the demo has both outcomes ready to show::

    python -m api.seed --customers 12
    # → data/demo_queries/<customer>/genuine_*.png and forgery_*.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from api.db import get_sessionmaker, init_db
from api.models.tables import Customer, CustomerEnrolment, Employee, ReferenceSignature
from api.security.auth import hash_password
from api.services.inference import ModelNotLoaded, get_service
from api.services.storage import get_store
from ml.config import DATA_ROOT
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest

DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo1234"


def seed_employee(session, username: str, password: str) -> Employee:
    employee = session.execute(
        select(Employee).where(Employee.username == username)
    ).scalar_one_or_none()
    if employee:
        return employee
    employee = Employee(
        username=username,
        full_name="Demo Operator",
        password_hash=hash_password(password),
        location="Head Office",
    )
    session.add(employee)
    session.flush()
    return employee


def seed_customers(
    session,
    manifest: Manifest,
    *,
    n_customers: int,
    references_per_customer: int,
    split: str,
    query_dir: Path,
) -> dict:
    service = get_service()
    store = get_store()

    by_signer: dict[str, dict[str, list]] = defaultdict(lambda: {"genuine": [], "forgery": []})
    for record in manifest.by_split(split):  # type: ignore[arg-type]
        key = "genuine" if record.label == "genuine" else "forgery"
        by_signer[record.signer_id][key].append(record)

    eligible = [
        s
        for s, buckets in by_signer.items()
        if len(buckets["genuine"]) > references_per_customer and buckets["forgery"]
    ]
    if not eligible:
        raise SystemExit(
            f"No signer in the {split!r} split has more than {references_per_customer} genuine "
            "samples plus a forgery. Generate a larger corpus first."
        )

    if query_dir.exists():
        shutil.rmtree(query_dir)
    query_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    summary: list[dict] = []

    for signer in sorted(eligible)[:n_customers]:
        number = f"C{created + 1001}"
        if session.execute(
            select(Customer).where(Customer.customer_number == number)
        ).scalar_one_or_none():
            created += 1
            continue

        buckets = by_signer[signer]
        script = buckets["genuine"][0].script
        customer = Customer(
            customer_number=number,
            full_name=f"Demo Customer {created + 1}",
            script=script,
        )
        session.add(customer)
        session.flush()

        # Enrol the first N genuine samples as stored specimens.
        embeddings = []
        for record in buckets["genuine"][:references_per_customer]:
            image = cv2.imread(str(manifest.resolve(record)), cv2.IMREAD_GRAYSCALE)
            canvas, embedding = service.embed_reference(image)
            embeddings.append(embedding)
            session.add(
                ReferenceSignature(
                    customer_id=customer.id,
                    image_key=store.put_image(image, prefix=f"references/{customer.id}"),
                    canvas_key=store.put_image(canvas, prefix=f"canvases/{customer.id}"),
                    embedding=[float(v) for v in embedding],
                    source="seed",
                )
            )

        stats = service.enrolment_stats(np.vstack(embeddings))
        if stats is not None:
            session.add(
                CustomerEnrolment(
                    customer_id=customer.id,
                    cohort_mean=stats.mean,
                    cohort_std=stats.std,
                    n_references=len(embeddings),
                    model_version=service.verifier.model_version if service.verifier else "",
                )
            )

        # Query images for the demonstrator: held-back genuine samples and
        # skilled forgeries of the same signer.
        customer_dir = query_dir / number
        customer_dir.mkdir(parents=True, exist_ok=True)
        for i, record in enumerate(buckets["genuine"][references_per_customer:][:3]):
            shutil.copy(manifest.resolve(record), customer_dir / f"genuine_{i}.png")
        for i, record in enumerate(buckets["forgery"][:3]):
            shutil.copy(manifest.resolve(record), customer_dir / f"forgery_{i}.png")

        summary.append(
            {
                "customer_number": number,
                "script": script,
                "references": len(embeddings),
                "query_images": len(list(customer_dir.iterdir())),
            }
        )
        created += 1

    return {"customers": summary, "query_dir": str(query_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the demo database")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--customers", type=int, default=12)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to draw demo customers from. Test = writers the model never saw.",
    )
    parser.add_argument("--username", default=DEMO_USERNAME)
    parser.add_argument("--password", default=DEMO_PASSWORD)
    parser.add_argument("--query-dir", type=Path, default=DATA_ROOT / "demo_queries")
    args = parser.parse_args()

    init_db()
    manifest = Manifest.load(args.manifest)

    try:
        get_service()
    except ModelNotLoaded as exc:
        raise SystemExit(str(exc)) from exc
    if not get_service().is_ready:
        raise SystemExit(get_service().load_error or "Model is not loaded; train one first.")

    session = get_sessionmaker()()
    try:
        seed_employee(session, args.username, args.password)
        summary = seed_customers(
            session,
            manifest,
            n_customers=args.customers,
            references_per_customer=args.references,
            split=args.split,
            query_dir=args.query_dir,
        )
        session.commit()
    finally:
        session.close()

    print(json.dumps(summary, indent=2))
    print(f"\nSign in as {args.username} / {args.password}")
    print(f"Demo query images: {args.query_dir.resolve()}")
    print(
        "\nEach customer folder holds genuine_*.png (should score high) and forgery_*.png "
        "(should score low). Drag them into the verify screen."
    )


if __name__ == "__main__":
    main()
