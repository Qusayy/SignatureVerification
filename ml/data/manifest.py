"""Dataset manifest, signer-level splits, and the guards that protect them.

Two invariants are enforced here, and both exist because breaking them produces
a system that looks excellent in evaluation and fails in a deployment site:

1. **Splits are by signer, never by image.** If any of a signer's signatures
   appear in training, none of that signer's signatures may appear in
   validation or test. A model that has seen a writer during training
   recognises them rather than verifying them, which inflates the reported
   accuracy by a large and unpredictable margin. :func:`assert_no_leakage` is
   cheap enough to run in CI and should be.

2. **Licence tracks do not mix.** Records from non-commercial sources are
   tagged Track A and can never contribute to a production weight. See
   ``docs/licensing.md``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ml.config import DATA_ROOT, NON_COMMERCIAL_SOURCES, SCRIPTS, TRACK_A, TRACK_B

__all__ = [
    "Record",
    "Manifest",
    "LeakageError",
    "LicenceError",
    "assert_no_leakage",
    "assert_track_b",
    "split_by_signer",
]

Label = Literal["genuine", "skilled_forgery", "random_forgery"]
Split = Literal["train", "val", "test"]

DEFAULT_MANIFEST_PATH = DATA_ROOT / "manifest.json"


class LeakageError(AssertionError):
    """Raised when a signer's samples straddle more than one split."""


class LicenceError(AssertionError):
    """Raised when non-commercial data would contribute to a shipped model."""


@dataclass
class Record:
    """One signature image and everything needed to use it correctly."""

    image_path: str  # relative to the manifest's root, POSIX separators
    signer_id: str  # globally unique — prefixed with its source
    label: Label
    script: str = SCRIPTS.default
    source: str = "unknown"
    licence_track: str = TRACK_A.name
    split: Split | None = None
    # True for images that came from the organisation's stored specimen database, as
    # opposed to freshly captured query signatures. The 1-ref vs N-ref
    # evaluation depends on this distinction.
    is_reference: bool = False
    captured_at: str | None = None  # ISO date; specimen age drives drift analysis
    notes: str = ""

    def __post_init__(self) -> None:
        if self.script not in SCRIPTS.values:
            raise ValueError(f"Unknown script {self.script!r}; expected one of {SCRIPTS.values}")
        if self.source.lower() in NON_COMMERCIAL_SOURCES:
            # Tagging is automatic and not overridable: a caller passing
            # track_b for CEDAR data is making exactly the mistake this guards.
            self.licence_track = TRACK_A.name


@dataclass
class Manifest:
    """A collection of records plus the root their paths are relative to."""

    records: list[Record] = field(default_factory=list)
    root: Path = DATA_ROOT

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MANIFEST_PATH) -> Manifest:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            records=[Record(**r) for r in payload["records"]],
            root=Path(payload.get("root", DATA_ROOT)),
        )

    def save(self, path: Path | str = DEFAULT_MANIFEST_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "counts": self.stats(),
            "records": [asdict(r) for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def extend(self, records: Iterable[Record]) -> None:
        self.records.extend(records)

    def resolve(self, record: Record) -> Path:
        return self.root / record.image_path

    def by_split(self, split: Split) -> list[Record]:
        return [r for r in self.records if r.split == split]

    def signers(self, split: Split | None = None) -> set[str]:
        return {r.signer_id for r in self.records if split is None or r.split == split}

    def by_signer(self, split: Split | None = None) -> dict[str, list[Record]]:
        out: dict[str, list[Record]] = defaultdict(list)
        for r in self.records:
            if split is None or r.split == split:
                out[r.signer_id].append(r)
        return dict(out)

    def stats(self) -> dict:
        return {
            "records": len(self.records),
            "signers": len(self.signers()),
            "by_label": dict(Counter(r.label for r in self.records)),
            "by_script": dict(Counter(r.script for r in self.records)),
            "by_source": dict(Counter(r.source for r in self.records)),
            "by_split": dict(Counter(r.split or "unassigned" for r in self.records)),
            "by_licence_track": dict(Counter(r.licence_track for r in self.records)),
            "signers_per_split": {
                s: len(self.signers(s)) for s in ("train", "val", "test")  # type: ignore[arg-type]
            },
        }


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def split_by_signer(
    manifest: Manifest,
    *,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 1337,
) -> Manifest:
    """Assign every record a split, partitioning **signers** rather than images.

    Signers are stratified by (source, script) so the Arabic and Latin
    proportions of the test set match the corpus. Without stratification a
    random draw can leave the test set with almost no Arabic signers, and the
    per-script accuracy breakdown then rests on a handful of writers.

    Returns the same manifest, mutated in place, for convenience.
    """
    if not 0 < test_frac < 1 or not 0 <= val_frac < 1 or test_frac + val_frac >= 1:
        raise ValueError("test_frac and val_frac must be fractions summing to less than 1")

    # Determine each signer's stratum from their records. A signer whose
    # records disagree on script is treated as "mixed".
    strata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in manifest.records:
        strata[record.signer_id].add((record.source, record.script))

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for signer, keys in strata.items():
        if len(keys) == 1:
            grouped[next(iter(keys))].append(signer)
        else:
            sources = {k[0] for k in keys}
            source = next(iter(sources)) if len(sources) == 1 else "mixed"
            grouped[(source, "mixed")].append(signer)

    assignment: dict[str, Split] = {}
    rng = random.Random(seed)
    for key in sorted(grouped):
        signers = sorted(grouped[key])  # sort first so the shuffle is reproducible
        rng.shuffle(signers)
        n = len(signers)
        n_test = max(1, round(n * test_frac)) if n > 2 else (1 if n > 1 else 0)
        n_val = max(1, round(n * val_frac)) if n - n_test > 2 and val_frac > 0 else 0
        for i, signer in enumerate(signers):
            if i < n_test:
                assignment[signer] = "test"
            elif i < n_test + n_val:
                assignment[signer] = "val"
            else:
                assignment[signer] = "train"

    for record in manifest.records:
        record.split = assignment[record.signer_id]

    assert_no_leakage(manifest)
    return manifest


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def assert_no_leakage(manifest: Manifest) -> None:
    """Raise if any signer's samples appear in more than one split.

    Run this in CI. It is the cheapest possible insurance against the single
    most common way signature verification results get silently inflated.
    """
    seen: dict[str, Split] = {}
    offenders: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        if record.split is None:
            continue
        if record.signer_id in seen and seen[record.signer_id] != record.split:
            offenders[record.signer_id].update({seen[record.signer_id], record.split})
        else:
            seen[record.signer_id] = record.split

    if offenders:
        detail = ", ".join(f"{s} in {sorted(v)}" for s, v in sorted(offenders.items())[:10])
        raise LeakageError(
            f"{len(offenders)} signer(s) appear in more than one split: {detail}. "
            "Splits must partition signers, not images."
        )


def assert_track_b(records: Iterable[Record]) -> None:
    """Raise if any record is non-commercial, blocking a production weight.

    Called by ``ml/embed/train.py`` before the first optimizer step when
    training with ``--track b``.
    """
    bad = [r for r in records if r.licence_track != TRACK_B.name]
    if bad:
        sources = sorted({r.source for r in bad})
        raise LicenceError(
            f"{len(bad)} record(s) from non-commercial source(s) {sources} cannot be used to "
            f"train a production (Track B) model. See docs/licensing.md. "
            f"Use --track a for research and benchmarking runs."
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_split(args: argparse.Namespace) -> None:
    manifest = Manifest.load(args.manifest)
    split_by_signer(manifest, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed)
    manifest.save(args.manifest)
    print(json.dumps(manifest.stats(), indent=2))
    print(f"\nSplits frozen in {Path(args.manifest).resolve()}")
    print("Do not re-split before the final evaluation — the test set must stay sealed.")


def _cmd_verify(args: argparse.Namespace) -> None:
    manifest = Manifest.load(args.manifest)
    assert_no_leakage(manifest)
    print(f"No signer leakage across splits ({len(manifest.signers())} signers).")
    if args.track == "b":
        assert_track_b(manifest.records)
        print("All records are Track B — cleared for a production model.")
    missing = [r.image_path for r in manifest.records if not manifest.resolve(r).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest path(s) missing on disk, e.g. {missing[:3]}")
    print(f"All {len(manifest)} image paths exist on disk.")


def _cmd_stats(args: argparse.Namespace) -> None:
    print(json.dumps(Manifest.load(args.manifest).stats(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manifest management")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Assign signer-level train/val/test splits")
    p_split.add_argument("--test-frac", type=float, default=0.2)
    p_split.add_argument("--val-frac", type=float, default=0.1)
    p_split.add_argument("--seed", type=int, default=1337)
    p_split.set_defaults(func=_cmd_split)

    p_verify = sub.add_parser("verify", help="Check leakage, licence track, and paths")
    p_verify.add_argument("--track", choices=["a", "b"], default="a")
    p_verify.set_defaults(func=_cmd_verify)

    p_stats = sub.add_parser("stats", help="Print corpus composition")
    p_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
