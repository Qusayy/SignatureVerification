"""Evaluation harness — the only source of accuracy numbers for this project.

Run it before believing anything about how well the system works::

    python -m ml.eval.benchmark --split test --by-script

What it enforces, and why each guard exists:

* **Cohort signers are drawn from training, never from test.** Normalising test
  scores against a cohort that includes test signers leaks the test set into
  the scores.
* **The calibrator is fitted on validation, never on test.** Otherwise the
  reported confidence has effectively seen the answers.
* **Skilled and random forgeries are reported separately.** Random forgeries
  are easy; quoting a combined number is the most common way signature
  verification accuracy gets overstated.
* **Synthetic-only evaluation refuses to emit a headline number.** The
  synthetic corpus exists to test plumbing. An accuracy figure computed from it
  says nothing about production performance, so the report is watermarked and the
  headline is withheld unless ``--allow-synthetic-headline`` is passed
  explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.config import ARTIFACT_ROOT, DOCS_ROOT, SCORING, resolve_device
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, Record, assert_no_leakage
from ml.embed.dataset import SignatureDataset, collate
from ml.eval.metrics import VerificationMetrics, compute_metrics, det_curve
from ml.scoring.calibrate import ScoreCalibrator
from ml.scoring.compare import compare_to_references
from ml.scoring.znorm import CohortNormalizer

__all__ = ["evaluate", "ComparisonSet"]


@dataclass
class ComparisonSet:
    """Score populations for one evaluation slice."""

    genuine: list[float]
    skilled_forgery: list[float]
    random_forgery: list[float]

    def is_evaluable(self, minimum: int = 10) -> bool:
        return len(self.genuine) >= minimum and len(self.skilled_forgery) >= minimum


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


@torch.no_grad()
def embed_records(
    model: torch.nn.Module,
    manifest: Manifest,
    split: str,
    device: str,
    *,
    batch_size: int = 32,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, list[Record]]:
    """Embed every record in a split, preserving order."""
    dataset = SignatureDataset(manifest, split, augment=False, cache_dir=cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)

    model.eval()
    vectors: list[np.ndarray] = []
    order: list[int] = []
    for batch in loader:
        out = model(batch["image"].to(device))
        vectors.append(torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy())
        order.extend(batch["index"].tolist())

    embeddings = np.concatenate(vectors).astype(np.float64)
    # Restore manifest order so embeddings[i] corresponds to dataset.records[i].
    restored = np.empty_like(embeddings)
    restored[np.asarray(order)] = embeddings
    return restored, dataset.records


# --------------------------------------------------------------------------
# Building comparisons
# --------------------------------------------------------------------------


def build_comparisons(
    embeddings: np.ndarray,
    records: list[Record],
    cohort: CohortNormalizer | None,
    *,
    max_references: int | None = None,
    seed: int = 1337,
) -> tuple[dict[str, ComparisonSet], dict]:
    """Form every genuine / skilled / random comparison in a split.

    Mirrors live operation: each writer's genuine samples are split into a
    reference set (standing in for the stored specimens) and query samples.
    Queries are then scored against that writer's references.

    Args:
        max_references: cap the reference set size. Pass 1 to measure the
            single-specimen case, which is what an organisation storing one signature
            per customer actually experiences.
    """
    rng = np.random.default_rng(seed)

    by_writer: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"genuine": [], "forgery": []})
    for i, record in enumerate(records):
        key = "genuine" if record.label == "genuine" else "forgery"
        by_writer[record.signer_id][key].append(i)

    writers = sorted(by_writer)
    references: dict[str, np.ndarray] = {}
    queries: dict[str, list[int]] = {}

    for writer in writers:
        genuine = by_writer[writer]["genuine"]
        if len(genuine) < 2:
            continue
        # Half the genuine samples become specimens, capped as requested.
        n_ref = max(1, len(genuine) // 2)
        if max_references:
            n_ref = min(n_ref, max_references)
        shuffled = list(rng.permutation(genuine))
        ref_idx, query_idx = shuffled[:n_ref], shuffled[n_ref:]
        if not query_idx:
            continue
        references[writer] = embeddings[ref_idx]
        queries[writer] = query_idx

    def score(query_index: int, writer: str) -> float:
        raw = compare_to_references(embeddings[query_index], references[writer]).raw
        if cohort is None:
            return raw
        return cohort.snorm(raw, embeddings[query_index], references=references[writer])

    overall = ComparisonSet([], [], [])
    per_script: dict[str, ComparisonSet] = defaultdict(lambda: ComparisonSet([], [], []))
    eligible = list(references)

    for writer in eligible:
        script = records[queries[writer][0]].script

        for q in queries[writer]:
            value = score(q, writer)
            overall.genuine.append(value)
            per_script[script].genuine.append(value)

        for f in by_writer[writer]["forgery"]:
            value = score(f, writer)
            overall.skilled_forgery.append(value)
            per_script[script].skilled_forgery.append(value)

        # Random forgeries: another writer's genuine signature presented as
        # this customer. Trivially easy, and reported only as a sanity check.
        others = [w for w in eligible if w != writer]
        if others:
            for other in rng.choice(others, size=min(len(others), 4), replace=False):
                value = score(queries[str(other)][0], writer)
                overall.random_forgery.append(value)
                per_script[script].random_forgery.append(value)

    summary = {
        "writers_evaluated": len(eligible),
        "references_per_writer": {
            w: int(references[w].shape[0]) for w in list(eligible)[:5]
        },
        "max_references": max_references,
    }
    return {"overall": overall, **per_script}, summary


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _metrics_table(name: str, metrics: VerificationMetrics) -> list[str]:
    rows = [
        f"### {name}",
        "",
        f"- Genuine comparisons: {metrics.n_genuine:,}",
        f"- Impostor comparisons: {metrics.n_impostor:,}",
        f"- **EER: {metrics.eer:.2%}**",
        f"- ROC AUC: {metrics.auc:.4f}",
        "",
        "| FAR target | TAR | Threshold |",
        "|---|---|---|",
    ]
    for target, tar in metrics.tar_at_far.items():
        flag = " ⚠️ unresolvable" if target in metrics.unresolvable_far_targets else ""
        rows.append(f"| {target:.1%} | {tar:.2%}{flag} | {metrics.threshold_at_far[target]:.4f} |")
    if metrics.unresolvable_far_targets:
        rows += [
            "",
            f"> ⚠️ FAR resolution is {metrics.far_resolution:.4f} "
            f"({metrics.n_impostor:,} impostor comparisons). Targets marked unresolvable "
            "cannot be measured at this sample size — widen the test set before quoting them.",
        ]
    rows.append("")
    return rows


def _det_plot(populations: dict[str, ComparisonSet], path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, comparisons in populations.items():
        if not comparisons.is_evaluable():
            continue
        far, frr = det_curve(np.asarray(comparisons.genuine), np.asarray(comparisons.skilled_forgery))
        ax.plot(far * 100, frr * 100, label=name)
    ax.set_xlabel("False acceptance rate (%)")
    ax.set_ylabel("False rejection rate (%)")
    ax.set_title("DET — genuine vs skilled forgeries")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def build_report(
    populations: dict[str, ComparisonSet],
    single_ref: dict[str, ComparisonSet] | None,
    context: dict,
    *,
    synthetic_only: bool,
    allow_synthetic_headline: bool,
) -> str:
    lines = ["# Accuracy Report", ""]

    if synthetic_only:
        lines += [
            "> ## ⚠️ SYNTHETIC DATA — NOT A VALID ACCURACY CLAIM",
            ">",
            "> Every signer in this evaluation is synthetic. These numbers measure that the",
            "> pipeline runs end to end; they say **nothing** about production performance and must",
            "> not appear in a business case, a procurement document, or a vendor comparison.",
            "> Real genuine samples and real *skilled* forgeries are required first.",
            "",
        ]

    lines += [
        "## Context",
        "",
        *[f"- {k}: `{v}`" for k, v in context.items()],
        "",
        "## Headline",
        "",
    ]

    overall = populations.get("overall")
    if overall and overall.is_evaluable() and (allow_synthetic_headline or not synthetic_only):
        metrics = compute_metrics(overall.genuine, overall.skilled_forgery, SCORING.far_targets)
        lines += [
            f"**EER against skilled forgeries: {metrics.eer:.2%}**  ",
            f"**TAR at FAR = 1%: {metrics.tar_at_far.get(0.01, float('nan')):.2%}**",
            "",
            "These are the two numbers to commit to. A single blended 'accuracy' figure is not",
            "reported because it hides the trade-off that risk and operations actually negotiate.",
            "",
        ]
    elif synthetic_only:
        lines += [
            "_Headline withheld: synthetic corpus. Re-run on real data, or pass",
            "`--allow-synthetic-headline` if you genuinely need the plumbing number._",
            "",
        ]
    else:
        lines += ["_Not enough comparisons to compute a headline metric._", ""]

    lines += ["## Detail", ""]
    for name, comparisons in populations.items():
        if not comparisons.is_evaluable():
            lines += [f"### {name}", "", "_Too few comparisons to evaluate._", ""]
            continue
        lines += _metrics_table(
            f"{name} — skilled forgeries",
            compute_metrics(comparisons.genuine, comparisons.skilled_forgery, SCORING.far_targets),
        )
        if len(comparisons.random_forgery) >= 10:
            random_metrics = compute_metrics(
                comparisons.genuine, comparisons.random_forgery, SCORING.far_targets
            )
            lines += [
                f"- Random-forgery EER (sanity check only): {random_metrics.eer:.2%}",
                "",
            ]

    if single_ref and "overall" in single_ref and single_ref["overall"].is_evaluable():
        full = compute_metrics(
            populations["overall"].genuine, populations["overall"].skilled_forgery, SCORING.far_targets
        )
        one = compute_metrics(
            single_ref["overall"].genuine, single_ref["overall"].skilled_forgery, SCORING.far_targets
        )
        lines += [
            "## Cost of storing only one specimen per customer",
            "",
            "| References per customer | EER | TAR @ FAR 1% |",
            "|---|---|---|",
            f"| All available | {full.eer:.2%} | {full.tar_at_far.get(0.01, 0):.2%} |",
            f"| Exactly one | {one.eer:.2%} | {one.tar_at_far.get(0.01, 0):.2%} |",
            "",
            f"Moving from one stored specimen to several changes EER by "
            f"{(one.eer - full.eer):+.2%}. This is the evidence for or against a "
            "re-enrolment programme.",
            "",
        ]

    lines += [
        "## How to read this",
        "",
        "- **EER** compares models. It is not an operating point.",
        "- **TAR@FAR** is the operating point: risk fixes the FAR it will tolerate, and TAR is",
        "  what the business gets back.",
        "- Skilled-forgery numbers are the real ones. Random-forgery numbers are a sanity check;",
        "  if they are not near-perfect, something upstream is broken.",
        "- The system is advisory. These metrics describe the quality of the advice, not an",
        "  automated decision.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    manifest = Manifest.load(args.manifest)
    assert_no_leakage(manifest)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from ml.embed.models import build_model

    model = build_model(payload.get("architecture", "signet")).to(device)
    model.load_state_dict(payload["model_state"])

    # --- Cohort from TRAINING signers only --------------------------------
    cohort = None
    if not args.no_cohort:
        train_embeddings, train_records = embed_records(
            model, manifest, "train", device, cache_dir=args.cache_dir
        )
        by_signer: dict[str, list[np.ndarray]] = defaultdict(list)
        for vector, record in zip(train_embeddings, train_records, strict=True):
            if record.label == "genuine":
                by_signer[record.signer_id].append(vector)
        cohort = CohortNormalizer.from_embeddings_by_signer(
            {k: np.vstack(v) for k, v in by_signer.items()}, size=args.cohort_size
        )
        cohort.save(ARTIFACT_ROOT / "cohort.npz")

    # --- Calibrator fitted on VALIDATION only -----------------------------
    calibrator = None
    if not args.no_calibrate and manifest.by_split("val"):
        val_embeddings, val_records = embed_records(
            model, manifest, "val", device, cache_dir=args.cache_dir
        )
        val_populations, _ = build_comparisons(val_embeddings, val_records, cohort, seed=args.seed)
        val_overall = val_populations["overall"]
        if val_overall.is_evaluable():
            calibrator = ScoreCalibrator.fit(
                val_overall.genuine, val_overall.skilled_forgery, fitted_on="val"
            )
            calibrator.save(ARTIFACT_ROOT / "calibrator.json")

    # --- Evaluate the requested split -------------------------------------
    embeddings, records = embed_records(model, manifest, args.split, device, cache_dir=args.cache_dir)
    populations, summary = build_comparisons(embeddings, records, cohort, seed=args.seed)
    if not args.by_script:
        populations = {"overall": populations["overall"]}

    single_ref = None
    if args.single_reference_comparison:
        single_ref, _ = build_comparisons(
            embeddings, records, cohort, max_references=1, seed=args.seed
        )

    sources = {r.source for r in records}
    synthetic_only = sources == {"synthetic"}

    context = {
        "checkpoint": Path(args.checkpoint).name,
        "architecture": payload.get("architecture", "?"),
        "licence_track": payload.get("provenance", {}).get("licence_track", "?"),
        "split": args.split,
        "sources": ", ".join(sorted(sources)),
        "signers_in_split": len({r.signer_id for r in records}),
        "cohort": cohort.describe() if cohort else "disabled",
        "calibrator": "fitted on val" if calibrator else "not fitted",
        **{k: v for k, v in summary.items() if k != "references_per_writer"},
    }

    report = build_report(
        populations,
        single_ref,
        context,
        synthetic_only=synthetic_only,
        allow_synthetic_headline=args.allow_synthetic_headline,
    )

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot = _det_plot(populations, out_path.parent / "det_curve.png")
    if plot:
        report += f"\n![DET curve]({plot.name})\n"
    out_path.write_text(report, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "context": context,
                "metrics": {
                    name: compute_metrics(
                        c.genuine, c.skilled_forgery, SCORING.far_targets
                    ).to_dict()
                    for name, c in populations.items()
                    if c.is_evaluable()
                },
                "synthetic_only": synthetic_only,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(report)
    print(f"\nReport: {out_path.resolve()}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a signature verification model")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_ROOT / "signet_track_b.pt")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--report", type=Path, default=DOCS_ROOT / "accuracy-report.md")
    parser.add_argument("--by-script", action="store_true", help="Break metrics down per script")
    parser.add_argument("--cohort-size", type=int, default=SCORING.cohort_size)
    parser.add_argument("--no-cohort", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--single-reference-comparison", action="store_true", default=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow-synthetic-headline",
        action="store_true",
        help="Emit a headline metric even on synthetic data. Do not use for any business claim.",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
