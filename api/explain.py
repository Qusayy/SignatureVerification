"""Explain one score, end to end, in terms you can check by hand.

    python -m api.explain --customer C1001 --image path/to/query.png

Prints every quantity between the pixels and the number on screen, plus the
arithmetic connecting them. Use it whenever a score looks wrong: it turns
"why did a 9% match score 90?" into a line-by-line answer naming which stage
produced the surprise.

It reads the same artifacts the service does, so what it prints is what the
service would do — not a reconstruction.
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
from api.services.storage import get_store
from api.settings import get_settings
from ml.config import SCORING
from ml.scoring.compare import compare_to_references, intra_reference_mean


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
        stored_mean = customer.enrolment.intra_reference_mean if customer.enrolment else None
        store = get_store()
        canvases = []
        for reference in customer.references:
            try:
                canvases.append(store.get_image(reference.canvas_key))
            except Exception:  # noqa: BLE001
                pass
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
    print(f"  calibrator fitted on              {calibrator.fitted_on or '(unknown)'} "
          f"({calibrator.n_fit_genuine} genuine / {calibrator.n_fit_impostor} impostor)")
    print(f"  calibrator weights stamp          {calibrator.weights_id or '(none)'}")
    print(f"  cohort applied                    {verifier.cohort is not None and SCORING.cohort_normalise}")

    distinct = sorted({round(float(v) * 100, 1) for v in calibrator.y})
    print(f"  distinct scores it can emit       {len(distinct)}: {distinct}")
    print(f"  its input domain                  {float(calibrator.x.min()):+.4f} "
          f"to {float(calibrator.x.max()):+.4f}")
    print("    anything outside that range is clamped to the nearest end, which is")
    print("    how a score saturates at the top or bottom regardless of the input.")

    print("\nSPECIMENS ON FILE")
    print(f"  count                             {len(embeddings)}")
    recomputed = intra_reference_mean(embeddings)
    print(f"  agreement among them (recomputed) {recomputed:+.4f}")
    print(f"  agreement stored at enrolment     "
          f"{stored_mean if stored_mean is not None else '(none)'}")
    print(f"  population baseline in calibrator {calibrator.population_reference_mean:+.4f}")
    floor = calibrator.genuine_similarity_floor or SCORING.absolute_similarity_floor
    origin = "measured on this corpus" if calibrator.genuine_similarity_floor else "config default"
    print(f"  genuine similarity floor          {floor:+.4f}  ({origin})")
    if len(embeddings) >= 2 and recomputed < SCORING.min_specimen_agreement:
        print(f"  !! below min_specimen_agreement ({SCORING.min_specimen_agreement}): these")
        print("     specimens do not look like the same hand. Broken enrolment.")

    query = service.verifier.embed_images([image])[0]
    plain = compare_to_references(query, embeddings, writer_normalise=False)
    scored = compare_to_references(
        query,
        embeddings,
        writer_normalise=SCORING.writer_normalise,
        reference_mean=stored_mean if stored_mean else None,
        population_reference_mean=calibrator.population_reference_mean,
    )

    print("\nSIMILARITY (before any normalisation)")
    for i, s in enumerate(plain.per_reference):
        print(_bar(f"vs specimen {i + 1}", s))
    print(_bar("closest", plain.max_similarity))
    print(_bar("average", plain.mean_similarity))
    combined = 0.5 * plain.max_similarity + 0.5 * plain.mean_similarity
    print(_bar("combined (0.5*closest + 0.5*avg)", combined))

    print("\nARITHMETIC")
    print(f"  baseline source                   {scored.baseline_source}")
    print(f"  baseline subtracted               {scored.intra_reference_mean:+.4f}")
    relative = combined - scored.intra_reference_mean
    print(f"  combined - baseline               {relative:+.4f}")
    absolute = combined - floor
    print(f"  combined - similarity floor       {absolute:+.4f}")
    print(f"  raw = min of the two              {scored.raw:+.4f}"
          f"   <- {'absolute floor' if absolute < relative else 'relative margin'} binds")

    score = calibrator.score_0_100(scored.raw)
    band = calibrator.band(scored.raw, SCORING)
    print("\nSCORE")
    if scored.raw < float(calibrator.x.min()):
        print("  !! raw is BELOW the calibrator's domain -> clamped to its lowest output")
    elif scored.raw > float(calibrator.x.max()):
        print("  !! raw is ABOVE the calibrator's domain -> clamped to its highest output.")
        print("     This is what makes everything score ~100. The usual cause is that")
        print("     `raw` is a bare similarity while the curve was fitted on margins.")
    print(f"  calibrated score                  {score:.1f} / 100   ({band.value})")

    print("\nSANITY")
    if combined < floor and score >= 50:
        print(f"  !! combined similarity {combined:.3f} is below the absolute floor yet the")
        print(f"     score is {score:.1f}. Something is bypassing the guard - report this output.")
    elif combined < floor:
        print(f"  combined similarity {combined:.3f} is below the absolute floor and the")
        print(f"     score is correctly low ({score:.1f}).")
    else:
        print("  nothing anomalous: the similarity is in the plausible range.")
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
