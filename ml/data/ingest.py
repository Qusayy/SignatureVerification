"""Ingest signature corpora into the canonical manifest.

Supported sources:

``synthetic``
    Output of :mod:`ml.data.synth`. Track B.

``cedar``
    The CEDAR signature database, whose flat layout is
    ``full_org/original_<signer>_<n>.png`` and
    ``full_forg/forgeries_<signer>_<n>.png``. Research licence, so Track A.

``internal``
    The organisation's own export. Expected layout, which the Phase 0 discovery step
    should confirm against the real database::

        <root>/
          <signer_id>/
            reference/   # stored specimen signatures from the customer record
            query/       # freshly captured signatures, if any
            forgery/     # skilled forgeries from the internal collection
          metadata.csv   # optional: signer_id,script,captured_at

    Track B — this is the only data that may back a production model.

Signer IDs are namespaced by source (``cedar:0042``) so two corpora can never
collide and silently merge two different people into one identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from ml.config import DATA_ROOT, SCRIPTS, TRACK_B
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, Record

__all__ = ["ingest_synthetic", "ingest_cedar", "ingest_internal"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _relative(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path.resolve()
    return str(rel).replace("\\", "/")


# --------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------


def ingest_synthetic(root: Path, manifest_root: Path = DATA_ROOT) -> list[Record]:
    """Read the ``manifest_raw.json`` written by :mod:`ml.data.synth`."""
    root = Path(root)
    raw_path = root / "manifest_raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Generate the corpus first: python -m ml.data.synth"
        )

    records = []
    for entry in json.loads(raw_path.read_text(encoding="utf-8")):
        image = root / entry["path"]
        records.append(
            Record(
                image_path=_relative(image, manifest_root),
                signer_id=f"synthetic:{entry['signer_id']}",
                label=entry["label"],
                script=entry["script"],
                source="synthetic",
                licence_track=TRACK_B.name,
                # First few genuine samples stand in for stored specimens.
                is_reference=entry["label"] == "genuine" and entry["path"].endswith(("g00.png", "g01.png", "g02.png")),
            )
        )
    return records


# --------------------------------------------------------------------------
# CEDAR
# --------------------------------------------------------------------------

_CEDAR_PATTERN = re.compile(r"^(original|forgeries)_(\d+)_(\d+)\.", re.IGNORECASE)


def ingest_cedar(root: Path, manifest_root: Path = DATA_ROOT) -> list[Record]:
    """Ingest the CEDAR database. Tagged Track A automatically."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = _CEDAR_PATTERN.match(path.name)
        if not match:
            continue
        kind, signer, _index = match.groups()
        records.append(
            Record(
                image_path=_relative(path, manifest_root),
                signer_id=f"cedar:{int(signer):04d}",
                label="genuine" if kind.lower() == "original" else "skilled_forgery",
                script="latin",
                source="cedar",  # Record.__post_init__ forces Track A
            )
        )
    if not records:
        raise FileNotFoundError(
            f"No CEDAR-style images under {root}. Expected names like original_12_5.png"
        )
    return records


# --------------------------------------------------------------------------
# Internal export
# --------------------------------------------------------------------------


def ingest_internal(root: Path, manifest_root: Path = DATA_ROOT) -> list[Record]:
    """Ingest the organisation's own export. Track B."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Internal export root {root} does not exist")

    metadata: dict[str, dict[str, str]] = {}
    meta_path = root / "metadata.csv"
    if meta_path.exists():
        with meta_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                signer = (row.get("signer_id") or "").strip()
                if signer:
                    metadata[signer] = row

    folder_labels = {
        "reference": ("genuine", True),
        "genuine": ("genuine", False),
        "query": ("genuine", False),
        "forgery": ("skilled_forgery", False),
        "forgeries": ("skilled_forgery", False),
    }

    records: list[Record] = []
    skipped: list[str] = []
    for signer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        signer = signer_dir.name
        meta = metadata.get(signer, {})
        script = (meta.get("script") or SCRIPTS.default).strip().lower()
        if script not in SCRIPTS.values:
            script = SCRIPTS.default
        captured_at = (meta.get("captured_at") or "").strip() or None

        for sub in sorted(p for p in signer_dir.iterdir() if p.is_dir()):
            mapping = folder_labels.get(sub.name.lower())
            if mapping is None:
                skipped.append(str(sub))
                continue
            label, is_reference = mapping
            for image in sorted(sub.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                records.append(
                    Record(
                        image_path=_relative(image, manifest_root),
                        signer_id=f"internal:{signer}",
                        label=label,  # type: ignore[arg-type]
                        script=script,
                        source="internal",
                        licence_track=TRACK_B.name,
                        is_reference=is_reference,
                        captured_at=captured_at,
                    )
                )
    if skipped:
        print(f"Skipped {len(skipped)} unrecognised folder(s), e.g. {skipped[:3]}")
    if not records:
        raise FileNotFoundError(f"No images found under {root} in the expected layout")
    return records


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def reference_profile(records: list[Record]) -> dict:
    """Profile how many stored specimens each signer has.

    The discovery question that most affects model design: a deployment where
    every signer has a single specimen cannot use per-signer threshold
    calibration, and its accuracy ceiling is meaningfully lower.
    """
    per_signer: dict[str, int] = defaultdict(int)
    for r in records:
        if r.is_reference:
            per_signer[r.signer_id] += 1
    signers_with_refs = len(per_signer)
    all_signers = {r.signer_id for r in records}
    counts = sorted(per_signer.values())
    histogram: dict[str, int] = defaultdict(int)
    for signer in all_signers:
        histogram[str(per_signer.get(signer, 0))] += 1
    return {
        "signers": len(all_signers),
        "signers_with_references": signers_with_refs,
        "signers_without_references": len(all_signers) - signers_with_refs,
        "references_per_signer_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        "median_references": counts[len(counts) // 2] if counts else 0,
    }


INGESTORS = {"synthetic": ingest_synthetic, "cedar": ingest_cedar, "internal": ingest_internal}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into the manifest")
    parser.add_argument("--source", choices=sorted(INGESTORS), required=True)
    parser.add_argument("--root", type=Path, required=True, help="Corpus directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop existing records from this source before adding (default: append)",
    )
    args = parser.parse_args()

    manifest = (
        Manifest.load(args.manifest) if Path(args.manifest).exists() else Manifest(root=DATA_ROOT)
    )
    if args.replace:
        manifest.records = [r for r in manifest.records if r.source != args.source]

    new_records = INGESTORS[args.source](args.root, manifest.root)
    existing = {r.image_path for r in manifest.records}
    added = [r for r in new_records if r.image_path not in existing]
    manifest.extend(added)
    manifest.save(args.manifest)

    print(f"Ingested {len(added)} new record(s) from {args.source} ({len(new_records)} seen).")
    print(json.dumps(manifest.stats(), indent=2))
    if args.source == "internal":
        print("\nReference profile (Phase 0 discovery):")
        print(json.dumps(reference_profile(manifest.records), indent=2))
    if any(r.split is None for r in manifest.records):
        print("\nRecords are unsplit. Next: python -m ml.data.manifest split")


if __name__ == "__main__":
    main()
