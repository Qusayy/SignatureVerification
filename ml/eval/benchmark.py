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
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.config import ARTIFACT_ROOT, DOCS_ROOT, SCORING, resolve_device
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, Record, assert_no_leakage
from ml.embed.dataset import SignatureDataset, collate
from ml.eval.metrics import VerificationMetrics, compute_metrics, det_curve
from ml.scoring.calibrate import ScoreCalibrator
from ml.scoring.compare import (
    DEFAULT_MAX_WEIGHT,
    compare_to_references,
    intra_reference_mean,
)
from ml.scoring.znorm import CohortNormalizer

__all__ = ["evaluate", "ComparisonSet"]


@dataclass
class ComparisonSet:
    """Score populations for one evaluation slice.

    The parallel ``*_writers`` lists record which writer produced each score.
    They exist for the bootstrap in :mod:`ml.eval.metrics`: resampling
    *comparisons* would treat the several scores from one writer as independent
    and understate the uncertainty, sometimes badly, because a writer with an
    unusual hand contributes a whole cluster of correlated scores.
    """

    genuine: list[float]
    skilled_forgery: list[float]
    random_forgery: list[float]
    genuine_writers: list[str] = field(default_factory=list)
    skilled_forgery_writers: list[str] = field(default_factory=list)
    random_forgery_writers: list[str] = field(default_factory=list)

    def is_evaluable(self, minimum: int = 10) -> bool:
        return len(self.genuine) >= minimum and len(self.skilled_forgery) >= minimum

    def add(self, population: str, value: float, writer: str) -> None:
        getattr(self, population).append(value)
        getattr(self, f"{population}_writers").append(writer)

    def metrics(self, impostors: str = "skilled_forgery") -> VerificationMetrics:
        """Metrics against one impostor population, with a writer-level CI."""
        return compute_metrics(
            self.genuine,
            getattr(self, impostors),
            SCORING.far_targets,
            genuine_writers=self.genuine_writers,
            impostor_writers=getattr(self, f"{impostors}_writers"),
        )


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
    writer_normalise: bool = SCORING.writer_normalise,
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
        writer_normalise: must match what the service does, or the reported
            numbers describe a system nobody is running.
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
        # Half the genuine samples become specimens. The reference/query split
        # is computed BEFORE `max_references` is applied, so capping the
        # specimen count leaves the query set untouched.
        #
        # It did not used to: capping n_ref moved samples from the reference
        # side to the query side, so the 1-specimen and N-specimen conditions
        # were scored over different genuine populations. That confound is what
        # made one specimen appear to beat three (35.89% vs 36.75% EER). With
        # the query set pinned the ordering is the expected one — 1 ref 27.1%,
        # 2 refs 22.4%, 6 refs 21.5% — so more specimens do help.
        n_ref = max(1, len(genuine) // 2)
        shuffled = list(rng.permutation(genuine))
        ref_idx, query_idx = shuffled[:n_ref], shuffled[n_ref:]
        if not query_idx:
            continue
        if max_references:
            ref_idx = ref_idx[:max_references]
        references[writer] = embeddings[ref_idx]
        queries[writer] = query_idx

    # Precomputed once per writer, as production does at enrolment.
    reference_means = {w: intra_reference_mean(r) for w, r in references.items()}

    def score(query_index: int, writer: str) -> float:
        raw = compare_to_references(
            embeddings[query_index],
            references[writer],
            writer_normalise=writer_normalise,
            reference_mean=reference_means[writer],
        ).raw
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
            overall.add("genuine", value, writer)
            per_script[script].add("genuine", value, writer)

        for f in by_writer[writer]["forgery"]:
            value = score(f, writer)
            overall.add("skilled_forgery", value, writer)
            per_script[script].add("skilled_forgery", value, writer)

        # Random forgeries: another writer's genuine signature presented as
        # this customer. Trivially easy, and reported only as a sanity check.
        others = [w for w in eligible if w != writer]
        if others:
            for other in rng.choice(others, size=min(len(others), 4), replace=False):
                value = score(queries[str(other)][0], writer)
                overall.add("random_forgery", value, writer)
                per_script[script].add("random_forgery", value, writer)

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
    counts = [
        f"- Genuine comparisons: {metrics.n_genuine:,}",
        f"- Impostor comparisons: {metrics.n_impostor:,}",
    ]
    if metrics.n_writers:
        counts.append(f"- Writers: {metrics.n_writers:,}")
    if metrics.eer_ci95:
        # The interval is the point of reporting it: on a test set this size a
        # couple of EER points is not a difference anyone can act on.
        counts.append(
            f"- **EER: {metrics.eer:.2%}**  (95% CI {metrics.eer_ci95[0]:.2%} – "
            f"{metrics.eer_ci95[1]:.2%}, writer-level bootstrap)"
        )
    else:
        counts.append(f"- **EER: {metrics.eer:.2%}**")

    rows = [
        f"### {name}",
        "",
        *counts,
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
    alternate: ComparisonSet | None = None,
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
        metrics = overall.metrics()
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
            comparisons.metrics(),
        )
        if len(comparisons.random_forgery) >= 10:
            random_metrics = comparisons.metrics("random_forgery")
            lines += [
                f"- Random-forgery EER (sanity check only): {random_metrics.eer:.2%}",
                "",
            ]

    served = populations.get("overall")
    if alternate is not None and served and served.is_evaluable() and alternate.is_evaluable():
        applied = served.metrics()
        other = alternate.metrics()
        cohort_on = bool(SCORING.cohort_normalise)
        lines += [
            "## Cohort normalisation: applied vs not",
            "",
            "Both recipes, scored over the identical comparisons. The service uses the",
            f"row marked *served*. Cohort normalisation is currently **{'on' if cohort_on else 'off'}**",
            "(`ScoringConfig.cohort_normalise`).",
            "",
            "| Recipe | EER | 95% CI | AUC | TAR @ FAR 1% |",
            "|---|---|---|---|---|",
        ]
        for label, m, is_served in (
            ("cohort applied" if cohort_on else "no cohort", applied, True),
            ("no cohort" if cohort_on else "cohort applied", other, False),
        ):
            ci = f"{m.eer_ci95[0]:.2%} – {m.eer_ci95[1]:.2%}" if m.eer_ci95 else "—"
            mark = " *(served)*" if is_served else ""
            lines += [
                f"| {label}{mark} | {m.eer:.2%} | {ci} | {m.auc:.4f} | "
                f"{m.tar_at_far.get(0.01, 0):.2%} |"
            ]
        lines += [
            "",
            "If the two intervals overlap heavily the difference is not evidence of",
            "anything; prefer the simpler recipe. Re-check this on every new corpus —",
            "cohort normalisation is a net loss on synthetic writers and may not be on",
            "real ones.",
            "",
        ]

    if single_ref and "overall" in single_ref and single_ref["overall"].is_evaluable():
        full = populations["overall"].metrics()
        one = single_ref["overall"].metrics()
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

    from ml.embed.models import load_checkpoint
    from ml.embed.provenance import weights_id as compute_weights_id

    model, payload = load_checkpoint(args.checkpoint)
    model = model.to(device)
    # Stamped onto the cohort and calibrator written below, so the service can
    # refuse to pair them with any other weights.
    model_id = payload.get("weights_id") or compute_weights_id(payload["model_state"])

    # --- Cohort from TRAINING signers only --------------------------------
    #
    # Built and saved regardless, so the artifact exists and the with/without
    # comparison below can be made. Whether it is *applied* is a separate
    # decision, taken by SCORING.cohort_normalise and mirrored from the
    # service, so the headline describes the system that is actually running.
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
        cohort.save(ARTIFACT_ROOT / "cohort.npz", weights_id=model_id)

    # What actually gets applied when scoring.
    scoring_cohort = cohort if SCORING.cohort_normalise else None

    # --- Calibrator fitted on VALIDATION only -----------------------------
    calibrator = None
    if not args.no_calibrate and manifest.by_split("val"):
        val_embeddings, val_records = embed_records(
            model, manifest, "val", device, cache_dir=args.cache_dir
        )
        val_populations, _ = build_comparisons(
            val_embeddings, val_records, scoring_cohort, seed=args.seed
        )
        val_overall = val_populations["overall"]
        if val_overall.is_evaluable():
            calibrator = ScoreCalibrator.fit(
                val_overall.genuine,
                val_overall.skilled_forgery,
                fitted_on="val",
                weights_id=model_id,
            )
            calibrator.save(ARTIFACT_ROOT / "calibrator.json")

    # --- Evaluate the requested split -------------------------------------
    embeddings, records = embed_records(model, manifest, args.split, device, cache_dir=args.cache_dir)
    populations, summary = build_comparisons(embeddings, records, scoring_cohort, seed=args.seed)
    if not args.by_script:
        populations = {"overall": populations["overall"]}

    # The recipe the service does *not* use, measured on the same comparisons.
    # Cohort normalisation was the default until it was measured and found to
    # cost 10 EER points here; keeping both numbers in every report is what
    # stops that decision from being re-taken on intuition.
    alternate = build_comparisons(
        embeddings, records, None if scoring_cohort is not None else cohort, seed=args.seed
    )[0]["overall"]

    single_ref = None
    if args.single_reference_comparison:
        single_ref, _ = build_comparisons(
            embeddings, records, scoring_cohort, max_references=1, seed=args.seed
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
        "cohort": (
            cohort.describe() if scoring_cohort else "built but not applied"
        ) if cohort else "disabled",
        "writer_normalisation": SCORING.writer_normalise,
        "calibrator": "fitted on val" if calibrator else "not fitted",
        **{k: v for k, v in summary.items() if k != "references_per_writer"},
    }

    report = build_report(
        populations,
        single_ref,
        context,
        synthetic_only=synthetic_only,
        allow_synthetic_headline=args.allow_synthetic_headline,
        alternate=alternate,
    )

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot = _det_plot(populations, out_path.parent / "det_curve.png")
    if plot:
        report += f"\n![DET curve]({plot.name})\n"
    out_path.write_text(report, encoding="utf-8")

    # The machine-readable half. It used to carry skilled-vs-genuine only,
    # which meant `ml.eval.ablation` — its main consumer — could not see the
    # random-forgery sanity check that this module's own report says must be
    # near-perfect or something upstream is broken. Everything the Markdown
    # shows now reaches the JSON, plus the per-writer scores a paired bootstrap
    # needs to compare two runs.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "context": context,
                "scoring_recipe": {
                    "writer_normalise": SCORING.writer_normalise,
                    "cohort_normalise": bool(cohort is not None),
                    "max_weight_on_nearest_reference": DEFAULT_MAX_WEIGHT,
                },
                "metrics": {
                    name: c.metrics().to_dict()
                    for name, c in populations.items()
                    if c.is_evaluable()
                },
                "random_forgery_metrics": {
                    name: c.metrics("random_forgery").to_dict()
                    for name, c in populations.items()
                    if len(c.random_forgery) >= 10 and c.genuine
                },
                "single_reference_metrics": {
                    name: c.metrics().to_dict()
                    for name, c in (single_ref or {}).items()
                    if c.is_evaluable()
                },
                "scores": {
                    name: {
                        population: {
                            "values": [round(v, 6) for v in getattr(c, population)],
                            "writers": getattr(c, f"{population}_writers"),
                        }
                        for population in ("genuine", "skilled_forgery", "random_forgery")
                    }
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
    # The report is UTF-8 and the Windows console is not. Replace rather than
    # raise: losing a dash from a printed table is not a reason to fail a run
    # whose output files were already written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate a signature verification model")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_ROOT / "signet_track_b.pt")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--report", type=Path, default=DOCS_ROOT / "accuracy-report.md")
    parser.add_argument("--by-script", action="store_true", help="Break metrics down per script")
    parser.add_argument("--cohort-size", type=int, default=SCORING.cohort_size)
    parser.add_argument("--no-cohort", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument(
        "--single-reference-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also measure the one-specimen-per-customer case. Was declared "
            "store_true with default=True, so it could never be switched off."
        ),
    )
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
