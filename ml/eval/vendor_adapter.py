"""Run a commercial engine over the same sealed test set, for the bake-off.

A build-vs-buy decision is only defensible if both systems are measured on
*identical* comparisons. This module exports the exact comparison list the
in-house model was scored on, so the vendor engine can be run against it and
the results compared like for like.

Three adapters:

``csv``
    Import scores the vendor produced offline. This is the realistic path for
    most engines: export the pair list, hand it to the vendor or run their
    desktop SDK, import the scores back. No integration work required.

``http``
    Call an on-premise vendor REST endpoint directly. Configure the URL and the
    request/response field names; no credentials are stored in the repository.

``stub``
    Fails with instructions. The default, so nobody accidentally reports
    fabricated vendor numbers.

Before running any of this, confirm the engine can operate **on premise**.
Sending customer signature images to a vendor cloud is a data protection
decision, not an engineering one — see docs/licensing.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.config import DOCS_ROOT, SCORING
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, Record
from ml.eval.metrics import compute_metrics

__all__ = ["export_pair_list", "load_vendor_scores", "compare_systems"]


@dataclass
class Pair:
    """One comparison: a query image against a customer's reference images."""

    pair_id: str
    signer_id: str
    query_path: str
    reference_paths: list[str]
    label: str  # "genuine" or "skilled_forgery"
    script: str


def build_pairs(manifest: Manifest, split: str = "test", *, seed: int = 1337) -> list[Pair]:
    """Build the same comparison list the in-house evaluation uses.

    Uses the same seed and the same reference/query division as
    ``ml.eval.benchmark.build_comparisons`` so the two systems see identical
    work. If that function's split logic changes, this must change with it.
    """
    rng = np.random.default_rng(seed)
    records = manifest.by_split(split)  # type: ignore[arg-type]

    by_writer: dict[str, dict[str, list[Record]]] = defaultdict(
        lambda: {"genuine": [], "forgery": []}
    )
    for record in records:
        key = "genuine" if record.label == "genuine" else "forgery"
        by_writer[record.signer_id][key].append(record)

    pairs: list[Pair] = []
    for writer in sorted(by_writer):
        genuine = by_writer[writer]["genuine"]
        if len(genuine) < 2:
            continue
        shuffled = [genuine[i] for i in rng.permutation(len(genuine))]
        n_ref = max(1, len(genuine) // 2)
        references = [r.image_path for r in shuffled[:n_ref]]

        for i, record in enumerate(shuffled[n_ref:]):
            pairs.append(
                Pair(
                    f"{writer}:g{i}", writer, record.image_path, references, "genuine", record.script
                )
            )
        for i, record in enumerate(by_writer[writer]["forgery"]):
            pairs.append(
                Pair(
                    f"{writer}:f{i}",
                    writer,
                    record.image_path,
                    references,
                    "skilled_forgery",
                    record.script,
                )
            )
    return pairs


def export_pair_list(pairs: list[Pair], path: Path) -> Path:
    """Write the comparison list as CSV for the vendor to score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_id", "signer_id", "query_path", "reference_paths", "label", "script"])
        for pair in pairs:
            writer.writerow(
                [
                    pair.pair_id,
                    pair.signer_id,
                    pair.query_path,
                    "|".join(pair.reference_paths),
                    pair.label,
                    pair.script,
                ]
            )
    return path


def load_vendor_scores(path: Path) -> dict[str, float]:
    """Read back a CSV of ``pair_id,score``. Higher score = more likely genuine.

    If the vendor reports a *distance* rather than a similarity, negate it
    before import — otherwise every metric comes out inverted and the
    comparison silently favours the in-house model.
    """
    scores: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pair_id = row.get("pair_id")
            raw = row.get("score")
            if pair_id is None or raw is None:
                raise ValueError(f"Vendor CSV must have pair_id and score columns; got {list(row)}")
            scores[pair_id] = float(raw)
    if not scores:
        raise ValueError(f"No scores found in {path}")
    return scores


def score_with_http(pairs: list[Pair], manifest: Manifest, config: dict) -> dict[str, float]:
    """Score pairs against an on-premise vendor REST endpoint.

    ``config`` keys: ``url``, optional ``headers``, ``query_field``,
    ``reference_field``, ``score_field``. Credentials come from the environment
    or the config file — never from this repository.
    """
    import base64
    import urllib.request

    url = config["url"]
    headers = config.get("headers", {})
    query_field = config.get("query_field", "query")
    reference_field = config.get("reference_field", "references")
    score_field = config.get("score_field", "score")

    def encode(path: str) -> str:
        return base64.b64encode((manifest.root / path).read_bytes()).decode("ascii")

    scores: dict[str, float] = {}
    for pair in pairs:
        body = json.dumps(
            {
                query_field: encode(pair.query_path),
                reference_field: [encode(p) for p in pair.reference_paths],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json", **headers}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            payload = json.loads(response.read())
        scores[pair.pair_id] = float(payload[score_field])
    return scores


def compare_systems(
    pairs: list[Pair],
    ours: dict[str, float],
    vendor: dict[str, float],
    *,
    by_script: bool = True,
) -> str:
    """Produce the bake-off report comparing two systems on identical pairs."""
    missing_ours = [p.pair_id for p in pairs if p.pair_id not in ours]
    missing_vendor = [p.pair_id for p in pairs if p.pair_id not in vendor]

    # Only pairs both systems scored are comparable. Silently dropping the rest
    # would let one system look better by declining the hard cases.
    common = [p for p in pairs if p.pair_id in ours and p.pair_id in vendor]
    if not common:
        raise ValueError("No pairs were scored by both systems")

    def metrics_for(scores: dict[str, float], subset: list[Pair]):
        genuine = [scores[p.pair_id] for p in subset if p.label == "genuine"]
        impostor = [scores[p.pair_id] for p in subset if p.label != "genuine"]
        if len(genuine) < 10 or len(impostor) < 10:
            return None
        return compute_metrics(genuine, impostor, SCORING.far_targets)

    lines = [
        "# Vendor Comparison",
        "",
        f"Both systems scored the same {len(common):,} comparisons from the sealed test set.",
        "",
    ]
    if missing_ours or missing_vendor:
        lines += [
            f"> ⚠️ {len(missing_ours)} pair(s) unscored by the in-house model and "
            f"{len(missing_vendor)} by the vendor. Only pairs scored by **both** are compared; "
            "a system that declines hard cases must not gain from doing so. Investigate the "
            "declines before drawing a conclusion.",
            "",
        ]

    slices = {"overall": common}
    if by_script:
        for script in sorted({p.script for p in common}):
            slices[script] = [p for p in common if p.script == script]

    lines += [
        "| Slice | System | EER (skilled) | TAR @ FAR 1% | TAR @ FAR 0.1% |",
        "|---|---|---|---|---|",
    ]
    for name, subset in slices.items():
        for system, scores in (("in-house", ours), ("vendor", vendor)):
            m = metrics_for(scores, subset)
            if m is None:
                lines.append(f"| {name} | {system} | insufficient data | — | — |")
                continue
            flag = " ⚠️" if 0.001 in m.unresolvable_far_targets else ""
            lines.append(
                f"| {name} | {system} | {m.eer:.2%} | {m.tar_at_far.get(0.01, 0):.2%} | "
                f"{m.tar_at_far.get(0.001, 0):.2%}{flag} |"
            )

    lines += [
        "",
        "## Reading this",
        "",
        "- Accuracy is one input. Weigh it against licence cost, on-premise feasibility, whether",
        "  the vendor can be retrained on this organisation's Arabic signatures, and what happens to the",
        "  system if the contract lapses.",
        "- ⚠️ marks a FAR target the test set is too small to resolve.",
        "- If the vendor reports distances rather than similarities, confirm they were negated",
        "  on import. An inverted metric makes the vendor look catastrophically bad.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor bake-off on the sealed test set")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--engine", choices=["csv", "http", "stub"], default="stub")
    parser.add_argument("--export-pairs", type=Path, help="Write the comparison list and exit")
    parser.add_argument("--vendor-scores", type=Path, help="CSV of pair_id,score (engine=csv)")
    parser.add_argument("--our-scores", type=Path, help="CSV of pair_id,score from the in-house model")
    parser.add_argument("--http-config", type=Path, help="JSON config (engine=http)")
    parser.add_argument("--report", type=Path, default=DOCS_ROOT / "vendor-comparison.md")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    manifest = Manifest.load(args.manifest)
    pairs = build_pairs(manifest, args.split, seed=args.seed)

    if args.export_pairs:
        path = export_pair_list(pairs, args.export_pairs)
        print(f"Exported {len(pairs):,} comparisons to {path.resolve()}")
        print("Score these with the vendor engine, then re-run with --engine csv --vendor-scores")
        return

    if args.engine == "stub":
        raise SystemExit(
            "No vendor engine configured. Either:\n"
            "  1. python -m ml.eval.vendor_adapter --export-pairs pairs.csv\n"
            "     then score them with the vendor and re-run with --engine csv, or\n"
            "  2. configure an on-premise endpoint and use --engine http --http-config cfg.json\n"
            "Vendor numbers are never fabricated by this tool."
        )

    if args.engine == "csv":
        if not args.vendor_scores:
            raise SystemExit("--engine csv requires --vendor-scores")
        vendor = load_vendor_scores(args.vendor_scores)
    else:
        vendor = score_with_http(pairs, manifest, json.loads(args.http_config.read_text()))

    if not args.our_scores:
        raise SystemExit(
            "--our-scores is required: export the in-house model's scores for the same pairs "
            "so both systems are measured on identical comparisons."
        )
    ours = load_vendor_scores(args.our_scores)

    report = compare_systems(pairs, ours, vendor)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
