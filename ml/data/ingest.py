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

from ml.config import DATA_ROOT, SCRIPTS, TRACK_A, TRACK_B
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


def ingest_generic(
    root: Path,
    manifest_root: Path = DATA_ROOT,
    *,
    source: str,
    layout: str = "folder",
    script: str = SCRIPTS.default,
    licence_track: str = TRACK_A.name,
    signer_depth: int = 0,
    no_forgeries: bool = False,
) -> list[Record]:
    """Ingest a third-party dataset laid out as folders of images.

    Handles the two layouts most public signature datasets use. Run
    :mod:`ml.data.inspect` first to see which one applies.

    Args:
        layout: ``"folder"`` when genuine/forged is a directory name somewhere
            in the path, ``"filename"`` when it is encoded in the file name.
        signer_depth: which path component under ``root`` identifies the
            signer, counting from 0. The default assumes ``<root>/<signer>/…``.
        no_forgeries: the dataset contains only genuine samples. Everything is
            labelled genuine and the corpus is usable for writer-identity
            pretraining but not for measuring skilled-forgery EER.

    Defaults to Track A. Third-party datasets are research-licensed far more
    often than their upload page suggests — a permissive licence on a
    *re-upload* does not relicense the underlying corpus. Pass
    ``licence_track`` explicitly only after checking the original source.
    """
    from ml.data.inspect import classify

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    if script not in SCRIPTS.values:
        raise ValueError(f"Unknown script {script!r}; expected one of {SCRIPTS.values}")

    records: list[Record] = []
    unlabelled: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        parts = path.relative_to(root).parts
        if len(parts) <= signer_depth:
            continue
        signer = parts[signer_depth]

        if no_forgeries:
            label = "genuine"
        else:
            label = None
            if layout == "folder":
                # Search the path from the deepest folder outward, so a
                # per-signer "forgeries" subfolder wins over a top-level
                # "train" folder.
                for part in reversed(parts[:-1]):
                    label = classify(part)
                    if label:
                        break
            else:
                label = classify(path.name)

            if label is None:
                unlabelled.append(str(path.relative_to(root)))
                continue

        records.append(
            Record(
                image_path=_relative(path, manifest_root),
                signer_id=f"{source}:{signer}",
                label=label,  # type: ignore[arg-type]
                script=script,
                source=source,
                licence_track=licence_track,
            )
        )

    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} image(s) could not be classified as genuine or forged, "
            f"e.g. {unlabelled[:3]}.\n"
            "Refusing to guess: labelling a forgery as genuine teaches the model to "
            "accept it, and that mistake raises validation accuracy rather than lowering "
            "it, so nothing downstream will catch it. Run `python -m ml.data.inspect "
            f"--root {root}` and pick the right --layout, or pass --no-forgeries if the "
            "dataset genuinely contains only genuine samples."
        )
    if not records:
        raise FileNotFoundError(f"No images found under {root}")
    return records


INGESTORS = {
    "synthetic": ingest_synthetic,
    "cedar": ingest_cedar,
    "internal": ingest_internal,
    "generic": ingest_generic,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into the manifest")
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "One of " + ", ".join(sorted(INGESTORS)) + ", or any name of your own when "
            "using --layout (the name becomes the signer-id prefix and the per-source "
            "tag in the accuracy report)"
        ),
    )
    parser.add_argument("--root", type=Path, required=True, help="Corpus directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop existing records from this source before adding (default: append)",
    )
    parser.add_argument(
        "--layout",
        choices=["folder", "filename"],
        help="Use the generic ingester: is genuine/forged encoded in folder or file names?",
    )
    parser.add_argument("--script", default=SCRIPTS.default, choices=list(SCRIPTS.values))
    parser.add_argument(
        "--track",
        choices=["a", "b"],
        default="a",
        help=(
            "Licence track. Defaults to 'a' (research only). Only pass 'b' after "
            "confirming the ORIGINAL dataset permits commercial use - a permissive "
            "licence on a re-upload does not relicense the underlying corpus."
        ),
    )
    parser.add_argument(
        "--signer-depth",
        type=int,
        default=0,
        help="Which path component under --root names the signer (0 = <root>/<signer>/...)",
    )
    parser.add_argument(
        "--no-forgeries",
        action="store_true",
        help="Dataset holds only genuine samples; usable for writer-identity pretraining",
    )
    args = parser.parse_args()

    manifest = (
        Manifest.load(args.manifest) if Path(args.manifest).exists() else Manifest(root=DATA_ROOT)
    )
    if args.replace:
        manifest.records = [r for r in manifest.records if r.source != args.source]

    if args.layout:
        new_records = ingest_generic(
            args.root,
            manifest.root,
            source=args.source,
            layout=args.layout,
            script=args.script,
            licence_track=TRACK_B.name if args.track == "b" else TRACK_A.name,
            signer_depth=args.signer_depth,
            no_forgeries=args.no_forgeries,
        )
    elif args.source in INGESTORS:
        new_records = INGESTORS[args.source](args.root, manifest.root)
    else:
        raise SystemExit(
            f"Unknown source {args.source!r} and no --layout given. Either use a built-in "
            f"source ({', '.join(sorted(INGESTORS))}) or pass --layout to use the generic "
            "ingester. Run `python -m ml.data.inspect --root <dir>` first."
        )
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
