"""Explain one score, end to end, in terms you can check by hand.

    python -m api.explain --customer C1001 --image path/to/query.png

Prints every quantity between the pixels and the number on screen. Use it when
a score looks wrong: it turns "why did a 9% match score 90?" into a line-by-line
answer naming the stage that produced the surprise.

It reads the same artifacts the service does, so what it prints is what the
service would do — not a reconstruction. The same figures are in the interface
under "Score breakdown", for anyone not at a terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from api.db import get_sessionmaker, init_db
from api.models.tables import Customer
from api.services.inference import get_service
from api.settings import get_settings
from ml.scoring.compare import compare_to_references


def _bar(label: str, value: float, width: int = 40) -> str:
    """A crude visual for a value in [-1, 1], so the scale is obvious."""
    filled = int(round(abs(value) * width))
    glyph = "#" if value >= 0 else "-"
    return f"  {label:<34}{value:+8.4f}  {glyph * min(filled, width)}"


def explain(customer_number: str, image_path: Path) -> int:
    settings = get_settings()
    service = get_service()
    if not service.is_ready:
        print(f"Model not loaded: {service.load_error}")
        return 1

    verifier = service.verifier
    assert verifier is not None
    calibrator = verifier.calibrator

    init_db()
    session = get_sessionmaker()()
    try:
        customer = session.execute(
            select(Customer).where(Customer.customer_number == customer_number)
        ).scalar_one_or_none()
        if customer is None:
            print(f"No customer {customer_number}")
            return 1
        if not customer.references:
            print(f"Customer {customer_number} has no specimens on file")
            return 1
        embeddings = np.asarray([r.embedding for r in customer.references], dtype=np.float64)
        enrolled_under = customer.enrolment.model_version if customer.enrolment else ""
    finally:
        session.close()

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read {image_path}")
        return 1

    print("=" * 74)
    print(f"  {customer_number}   {image_path.name}")
    print("=" * 74)

    print("\nARTIFACTS")
    print(f"  checkpoint                        {settings.checkpoint_path}")
    print(f"  model version                     {verifier.model_version}")
    print(f"  enrolled under                    {enrolled_under or '(unknown)'}")
    if enrolled_under and enrolled_under != verifier.model_version:
        print("  !! stale enrolment: the service refuses this customer with a 409.")
        print("     python -m api.reenrol --apply")
    print(
        f"  calibrator fitted on              {calibrator.fitted_on or '(unknown)'} "
        f"({calibrator.n_fit_genuine} genuine / {calibrator.n_fit_impostor} impostor)"
    )
    print(
        f"  fitted for                        {calibrator.protocol_references} "
        "specimen(s) per customer"
    )
    print(f"  distinct scores it can emit       {calibrator.distinct_scores}")
    print(
        f"  input domain (similarity)         {float(calibrator.x.min()):.4f} "
        f"to {float(calibrator.x.max()):.4f}"
    )
    print(f"  highest score it can emit         {float(calibrator.y.max()) * 100:.1f}")
    if calibrator.thin_fit:
        print("  !! thin fit: fewer than 200 comparisons per class, so the curve is coarse.")

    green_min, red_max = calibrator.effective_edges()
    print("\nBANDS")
    print(f"  green from                        {green_min}")
    print(f"  red at or below                   {red_max}")
    print(
        "  derived to hold                   FAR <= "
        f"{calibrator.operating_points.get('green_max_far', 0):.0%}, FRR <= "
        f"{calibrator.operating_points.get('red_max_frr', 0):.0%} on validation"
    )

    query = verifier.embed_images([image])[0]
    comparison = compare_to_references(query, embeddings)

    print("\nSIMILARITY")
    for i, value in enumerate(comparison.per_reference):
        print(_bar(f"vs specimen {i + 1}", value))
    print(_bar("used (nearest specimen)", comparison.similarity))

    score = calibrator.score_0_100(comparison.similarity)
    band = calibrator.band(comparison.similarity)
    print("\nSCORE")
    if comparison.similarity < float(calibrator.x.min()):
        print("  below the curve's domain -> clamped to its lowest output")
    elif comparison.similarity > float(calibrator.x.max()):
        print("  above the curve's domain -> clamped to its highest output")
    print(f"  calibrated score                  {score:.1f} / 100   ({band.value})")

    genuine = calibrator.share_reaching(comparison.similarity, "genuine")
    impostor = calibrator.share_reaching(comparison.similarity, "skilled")
    if genuine is not None:
        print("\nIN CONTEXT")
        print(f"  genuine signatures reaching this  {genuine:.0%}")
        if impostor is not None:
            print(f"  skilled forgeries reaching this   {impostor:.0%}")

    print("\nSANITY")
    if calibrator.score_0_100(comparison.similarity - 0.05) > score:
        print("  !! the curve is not monotone here. Report this output.")
    else:
        print("  monotone: a closer match cannot score lower than this one.")
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Explain one verification score")
    parser.add_argument("--customer", required=True, help="Customer number, e.g. C1001")
    parser.add_argument("--image", required=True, type=Path, help="Query signature image")
    args = parser.parse_args()

    raise SystemExit(explain(args.customer, args.image))


if __name__ == "__main__":
    main()
