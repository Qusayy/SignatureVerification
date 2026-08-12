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
from ml.scoring.compare import compare_to_references, intra_reference_mean
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

    def is_thin(self, threshold: int = 200) -> bool:
        """Too few comparisons for a calibration curve worth shipping."""
        return len(self.genuine) < threshold or len(self.skilled_forgery) < threshold

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
    seed: int = 1337,
) -> tuple[dict[str, ComparisonSet], dict]:
    """Form every genuine / skilled / random comparison in a split.

    Mirrors live operation: each writer's genuine samples are split into a
    reference set (standing in for the stored specimens) and query samples.
    Queries are then scored against that writer's references.

    Args:
        max_references: specimens per customer. Pass
            ``SCORING.calibration_references`` — 1 — for the served protocol.
            Other values exist for the sensitivity table, which measures what a
            future multi-specimen enrolment would be worth.
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
    # Diagnostic only — reported so a corpus with broken enrolments is visible,
    # never used in scoring. See ml/scoring/compare.py.
    reference_means = {w: intra_reference_mean(r) for w, r in references.items()}

    def score(query_index: int, writer: str) -> float:
        """Exactly what the service computes. No flags, by design.

        `build_comparisons` used to take `writer_normalise`, a population mean
        and a cohort, and `evaluate` passed different combinations to different
        calls. That is how the harness came to measure a system nobody was
        running: it never passed `similarity_floor`, so it used 0.60 while
        production used 0.92, and it never passed `max_references` when fitting
        the calibrator, so every shipped curve was built for six specimens and
        applied to one.
        """
        similarity = compare_to_references(
            embeddings[query_index], references[writer]
        ).similarity
        if cohort is None:
            return similarity
        # Retained only for the alternative-recipe table in the report.
        return cohort.snorm(similarity, embeddings[query_index], references=references[writer])

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

    measurable = [v for v in reference_means.values() if v > 0.0]
    summary = {
        "writers_evaluated": len(eligible),
        "references_per_customer": max_references or "all available",
        "specimen_agreement_median": (
            round(float(np.median(measurable)), 5) if measurable else None
        ),
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


def _band_mix(comparisons: ComparisonSet, calibrator) -> list[str]:
    """Where genuine and forged traffic lands under the derived edges.

    The operational contract, and the thing risk signs off. An EER says how
    separable the populations are; this says what an operator sees on a normal
    day.
    """
    green_min, red_max = calibrator.effective_edges()

    def mix(values: list[float]) -> tuple[float, float, float]:
        scores = np.array([calibrator.score_0_100(v) for v in values])
        n = max(len(scores), 1)
        return (
            float((scores >= green_min).sum()) / n,
            float(((scores > red_max) & (scores < green_min)).sum()) / n,
            float((scores <= red_max).sum()) / n,
        )

    rows = [
        "## Where the traffic lands",
        "",
        f"Bands from the calibrator: green from **{green_min}**, red at or below "
        f"**{red_max}**, derived to hold FAR <= "
        f"{calibrator.operating_points.get('green_max_far', 0):.0%} and FRR <= "
        f"{calibrator.operating_points.get('red_max_frr', 0):.0%} on validation.",
        "",
        "| | Green | Amber | Red |",
        "|---|---|---|---|",
    ]
    for label, values in (
        ("Genuine signatures", comparisons.genuine),
        ("Skilled forgeries", comparisons.skilled_forgery),
    ):
        g, a, r = mix(values)
        rows.append(f"| {label} | {g:.1%} | {a:.1%} | {r:.1%} |")
    rows += [
        "",
        "Amber means the operator compares manually, which is what they do for every",
        "signature today, so amber is not a failure. Green is the only band that saves",
        "time and red the only one that raises an alarm.",
        "",
    ]
    return rows


def build_report(
    populations: dict[str, ComparisonSet],
    sensitivity: dict | None,
    context: dict,
    *,
    synthetic_only: bool,
    allow_synthetic_headline: bool,
    alternate: ComparisonSet | None = None,
    calibrator=None,
    references_per_customer: int = 1,
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
        ci = (
            f" (95% CI {metrics.eer_ci95[0]:.2%} - {metrics.eer_ci95[1]:.2%})"
            if metrics.eer_ci95
            else ""
        )
        lines += [
            f"Measured at **{references_per_customer} stored specimen(s) per customer**, "
            "which is what the service runs.",
            "",
            f"**EER against skilled forgeries: {metrics.eer:.2%}**{ci}  ",
            f"**TAR at FAR = 1%: {metrics.tar_at_far.get(0.01, float('nan')):.2%}**",
            "",
            "These are the two numbers to commit to. A single blended 'accuracy' figure is not",
            "reported because it hides the trade-off that risk and operations actually negotiate.",
            "",
        ]
        if calibrator is not None:
            lines += _band_mix(overall, calibrator)
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
        lines += [
            "## Alternative recipe: cohort normalisation",
            "",
            "Not on the serving path. Kept measured so the decision to drop it stays",
            "evidence-led, and so a future multi-specimen pilot is one benchmark run away",
            "rather than a re-implementation.",
            "",
            "| Recipe | EER | 95% CI | AUC | TAR @ FAR 1% |",
            "|---|---|---|---|---|",
        ]
        for label, m in (
            ("plain similarity *(served)*", applied),
            ("cohort S-norm", other),
        ):
            ci = f"{m.eer_ci95[0]:.2%} - {m.eer_ci95[1]:.2%}" if m.eer_ci95 else "-"
            lines.append(
                f"| {label} | {m.eer:.2%} | {ci} | {m.auc:.4f} | "
                f"{m.tar_at_far.get(0.01, 0):.2%} |"
            )
        lines += [
            "",
            "If the intervals overlap heavily the difference is not evidence of anything;",
            "prefer the simpler recipe. Re-check on every new corpus — cohort",
            "normalisation is a net loss on synthetic writers and may not be on real ones.",
            "",
        ]

    if sensitivity:
        lines += [
            "## What more specimens would be worth",
            "",
            "Not deployable today — the customer base is too large for a specimen-collection",
            "programme — but this is the business case if that ever changes.",
            "",
            "| Specimens per customer | EER | 95% CI | TAR @ FAR 1% |",
            "|---|---|---|---|",
        ]
        for key, comparisons in sensitivity.items():
            m = comparisons.metrics()
            ci = f"{m.eer_ci95[0]:.2%} – {m.eer_ci95[1]:.2%}" if m.eer_ci95 else "—"
            served = " *(served)*" if key == references_per_customer else ""
            lines.append(
                f"| {key}{served} | {m.eer:.2%} | {ci} | {m.tar_at_far.get(0.01, 0):.2%} |"
            )
        lines.append("")

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
    # Not on the serving path — it cost 15.7 EER points here, because it answers
    # the random-impostor question, which is already solved at 0.00%. Built only
    # so the alternative-recipe table can keep reporting what it would score,
    # which is what stops that decision being re-taken on intuition.
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

    # --- Calibrator fitted on VALIDATION, at the SERVED protocol ----------
    #
    # `max_references=REFS` is the whole point. Fitting without it built every
    # shipped curve on six specimens per customer while production ran one, so
    # every score the system ever showed was a value from one protocol pushed
    # through a scale built for another.
    refs = SCORING.calibration_references
    calibrator = None
    if not args.no_calibrate and manifest.by_split("val"):
        val_embeddings, val_records = embed_records(
            model, manifest, "val", device, cache_dir=args.cache_dir
        )
        val_populations, _ = build_comparisons(
            val_embeddings, val_records, None, max_references=refs, seed=args.seed
        )
        val_overall = val_populations["overall"]
        if val_overall.is_evaluable():
            if val_overall.is_thin() and not args.allow_thin_calibration:
                raise SystemExit(
                    f"Validation yields only {len(val_overall.genuine)} genuine / "
                    f"{len(val_overall.skilled_forgery)} impostor comparisons. A curve fitted "
                    "on that is coarse enough that the score stops discriminating.\n"
                    "The fix is more validation *writers* — not more comparisons per writer, "
                    "and never borrowing from train, whose writers the model memorised.\n"
                    "Pass --allow-thin-calibration to proceed anyway."
                )
            calibrator = ScoreCalibrator.fit(
                val_overall.genuine,
                val_overall.skilled_forgery,
                protocol_references=refs,
                fitted_on="val",
                weights_id=model_id,
            )
            # Band edges from operating points, on val. Never on test.
            calibrator.derive_band_edges(val_overall.genuine, val_overall.skilled_forgery)
            calibrator.save(ARTIFACT_ROOT / "calibrator.json")

    # --- Evaluate the requested split, at the same protocol ---------------
    embeddings, records = embed_records(model, manifest, args.split, device, cache_dir=args.cache_dir)
    populations, summary = build_comparisons(
        embeddings, records, None, max_references=refs, seed=args.seed
    )
    if not args.by_script:
        populations = {"overall": populations["overall"]}

    # The guard that makes the original bug unrepresentable: the protocol the
    # curve was fitted for, the protocol the headline measures, and the protocol
    # the service will run must be the same number.
    if calibrator is not None and calibrator.protocol_references != summary["max_references"]:
        raise SystemExit(
            f"Calibrator fitted for {calibrator.protocol_references} specimen(s) but the "
            f"headline measures {summary['max_references']}. These must agree or the "
            "reported accuracy describes a system nobody runs."
        )

    # Recipes the service does *not* use, on the identical comparisons. Kept so
    # a future multi-specimen pilot is one benchmark run away rather than a
    # re-implementation, and so the decision to drop them stays evidence-led.
    alternate = build_comparisons(embeddings, records, cohort, max_references=refs, seed=args.seed)[
        0
    ]["overall"]

    # How much a second and third specimen would be worth. Never reaches a
    # screen; it is the business case for an enrolment programme.
    sensitivity = {}
    if args.reference_sensitivity:
        for n in (1, 2, 3, None):
            pops, _ = build_comparisons(
                embeddings, records, None, max_references=n, seed=args.seed
            )
            if pops["overall"].is_evaluable():
                sensitivity[n or "all"] = pops["overall"]

    sources = {r.source for r in records}
    synthetic_only = sources == {"synthetic"}

    context = {
        "checkpoint": Path(args.checkpoint).name,
        "architecture": payload.get("architecture", "?"),
        "licence_track": payload.get("provenance", {}).get("licence_track", "?"),
        "split": args.split,
        "sources": ", ".join(sorted(sources)),
        "signers_in_split": len({r.signer_id for r in records}),
        "calibrator": (
            f"fitted on val for {calibrator.protocol_references} specimen(s), "
            f"green from {calibrator.green_min}, red at or below {calibrator.red_max}"
            if calibrator
            else "not fitted"
        ),
        **{k: v for k, v in summary.items() if k != "references_per_writer"},
    }

    report = build_report(
        populations,
        sensitivity,
        context,
        synthetic_only=synthetic_only,
        allow_synthetic_headline=args.allow_synthetic_headline,
        alternate=alternate,
        calibrator=calibrator,
        references_per_customer=refs,
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
                    "score_input": "similarity",
                    "pooling": SCORING.pooling,
                    "references_per_customer": refs,
                    "green_min": calibrator.green_min if calibrator else None,
                    "red_max": calibrator.red_max if calibrator else None,
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
                "reference_sensitivity": {
                    str(key): c.metrics().to_dict()
                    for key, c in (sensitivity or {}).items()
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
        "--reference-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also measure 2 and 3 specimens per customer. Never reaches a screen; "
            "it is the business case for an enrolment programme."
        ),
    )
    parser.add_argument(
        "--allow-thin-calibration",
        action="store_true",
        help=(
            "Write a calibrator fitted on fewer than 200 comparisons per class. "
            "The curve will be coarse enough that the score stops discriminating."
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
