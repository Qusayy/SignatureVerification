"""How much accuracy does another thousand writers actually buy?

The organisation has to decide how large a signature-collection programme to fund.
That decision should rest on a measured curve, not on a guess, because the
collection is the most expensive part of the project and the returns flatten at
some point.

This trains the same model at several training-set sizes and plots EER against
writer count.

Two things make the comparison valid, and both are easy to get wrong:

* **Only the training split varies.** Validation and test are pinned, so every
  point on the curve is scored on the identical sealed test set. Resampling the
  test set per run would make the curve meaningless.
* **Gradient steps are held constant** via ``--batches-per-epoch``. Otherwise a
  larger corpus receives both more writers *and* more training, and the two
  effects cannot be separated.

**Constant steps must still be *enough* steps.** Holding the budget fixed is
necessary but not sufficient: a larger corpus needs at least as many updates to
converge, so if the shared budget is too small the big-corpus points are simply
undertrained and the curve slopes the wrong way. A CPU run of this sweep at 416
total steps produced exactly that artefact — 250 writers scored *worse* than 84
while its identity loss was still 40% higher, i.e. it had not converged.
:func:`check_step_budget` refuses to start such a run.

Rule of thumb: each writer wants roughly 40-80 gradient steps of exposure, so
budget ``max(1500, 60 * n_writers / writers_per_batch)`` steps in total. For
1,700 writers that is several thousand steps — a GPU job, not a laptop job.

Usage::

    # Laptop-scale smoke test of the machinery
    python -m ml.eval.ablation --writer-counts 84 250 --epochs 8 --allow-undertrained

    # The real sweep
    python -m ml.eval.ablation --writer-counts 84 250 600 1700 \\
        --epochs 60 --batches-per-epoch 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ml.config import ARTIFACT_ROOT, DOCS_ROOT
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, assert_no_leakage

__all__ = ["subset_training_writers", "run_ablation"]


def subset_training_writers(
    manifest: Manifest, n_writers: int, *, seed: int = 1337
) -> Manifest:
    """Return a manifest whose *training* split holds at most ``n_writers``.

    Validation and test records pass through untouched, so every ablation point
    is evaluated against the same sealed test set.
    """
    import random

    train_signers = sorted(manifest.signers("train"))
    if n_writers >= len(train_signers):
        keep = set(train_signers)
    else:
        rng = random.Random(seed)
        keep = set(rng.sample(train_signers, n_writers))

    records = [
        r for r in manifest.records if r.split != "train" or r.signer_id in keep
    ]
    subset = Manifest(records=records, root=manifest.root)
    assert_no_leakage(subset)
    return subset


def recommended_steps(n_writers: int, writers_per_batch: int = 8) -> int:
    """Gradient steps needed for a corpus of ``n_writers`` to converge.

    Each writer needs repeated exposure before the identity head separates it
    from the rest. Batches cover ``writers_per_batch`` writers at a time, so the
    requirement scales with the corpus.
    """
    return max(1500, int(60 * n_writers / max(writers_per_batch, 1)))


def check_step_budget(
    writer_counts: list[int], total_steps: int, *, writers_per_batch: int = 8
) -> list[str]:
    """Return a warning per writer count the step budget cannot converge."""
    problems = []
    for n in writer_counts:
        needed = recommended_steps(n, writers_per_batch)
        if total_steps < needed:
            problems.append(
                f"{n} writers needs about {needed:,} gradient steps to converge; "
                f"this run budgets {total_steps:,}"
            )
    return problems


def run_ablation(args: argparse.Namespace) -> Path:
    total_steps = args.epochs * args.batches_per_epoch
    problems = check_step_budget(args.writer_counts, total_steps, writers_per_batch=8)
    if problems and not args.allow_undertrained:
        raise SystemExit(
            "Refusing to run: the step budget cannot converge every point, so the curve "
            "would slope the wrong way and read as 'more writers hurts'.\n  - "
            + "\n  - ".join(problems)
            + "\n\nRaise --epochs or --batches-per-epoch, drop the larger writer counts, "
            "or pass --allow-undertrained if you only want to smoke-test the machinery."
        )
    if problems:
        print("WARNING: undertrained points, results are not a valid curve:")
        for problem in problems:
            print(f"  - {problem}")

    manifest = Manifest.load(args.manifest)
    available = len(manifest.signers("train"))
    results: list[dict] = []

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for n_writers in args.writer_counts:
        if n_writers > available:
            print(f"Skipping {n_writers}: only {available} training writers in the manifest")
            continue

        subset = subset_training_writers(manifest, n_writers, seed=args.seed)
        subset_path = work_dir / f"manifest_{n_writers}.json"
        subset.save(subset_path)

        checkpoint = work_dir / f"model_{n_writers}.pt"
        report = work_dir / f"report_{n_writers}.md"

        train_cmd = [
            sys.executable, "-u", "-m", "ml.embed.train",
            "--arch", args.arch,
            "--manifest", str(subset_path),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--track", "b",
            "--forgery-weight", str(args.forgery_weight),
            "--batches-per-epoch", str(args.batches_per_epoch),
            "--val-every", str(args.val_every),
            "--out", str(checkpoint),
            "--notes", f"ablation: {n_writers} training writers",
        ]
        print(f"\n=== {n_writers} writers ===\n{' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)

        eval_cmd = [
            sys.executable, "-u", "-m", "ml.eval.benchmark",
            "--checkpoint", str(checkpoint),
            "--manifest", str(subset_path),
            "--split", "test",
            "--report", str(report),
            "--allow-synthetic-headline",
        ]
        subprocess.run(eval_cmd, check=True)

        metrics = json.loads(report.with_suffix(".json").read_text())
        overall = metrics["metrics"].get("overall", {})
        results.append(
            {
                "writers": n_writers,
                "eer": overall.get("eer"),
                "tar_at_far_1pct": overall.get("tar_at_far", {}).get("0.01"),
                "checkpoint": str(checkpoint),
            }
        )
        print(f"{n_writers} writers → EER {overall.get('eer')}")

    return _write_report(results, Path(args.report), available)


def _write_report(results: list[dict], path: Path, available: int) -> Path:
    lines = [
        "# Writer-Count Ablation",
        "",
        "How EER against skilled forgeries changes with the number of training writers.",
        "Only the training split varies; validation and test are pinned, and gradient steps",
        "are held constant, so writer count is the sole variable.",
        "",
        "| Training writers | EER (skilled) | TAR @ FAR 1% |",
        "|---|---|---|",
    ]
    for row in results:
        eer = f"{row['eer']:.2%}" if row["eer"] is not None else "—"
        tar = f"{row['tar_at_far_1pct']:.2%}" if row.get("tar_at_far_1pct") else "—"
        lines.append(f"| {row['writers']:,} | {eer} | {tar} |")

    lines += [
        "",
        f"Training writers available in this manifest: {available:,}.",
        "",
        "## How to read this",
        "",
        "- If EER is still falling steeply at the largest point, the corpus is the binding",
        "  constraint and more writers is the highest-value investment available.",
        "- If it has flattened, further collection buys little and the bottleneck has moved",
        "  to the model, the preprocessing, or the capture channel.",
        "- These points are on synthetic writers unless the manifest says otherwise. Synthetic",
        "  writers are more self-similar than real ones, so treat the curve as a lower bound on",
        "  how many *real* writers are needed, not an upper bound.",
        "",
    ]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        usable = [r for r in results if r["eer"] is not None]
        if len(usable) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([r["writers"] for r in usable], [r["eer"] * 100 for r in usable], marker="o")
            ax.set_xlabel("Training writers")
            ax.set_ylabel("EER vs skilled forgeries (%)")
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title("Accuracy vs corpus size")
            fig.tight_layout()
            plot_path = path.parent / "writer_count_ablation.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            lines.append(f"![Writer-count ablation]({plot_path.name})")
    except ImportError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure accuracy against training-corpus size")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--writer-counts", type=int, nargs="+", default=[84, 250, 500])
    parser.add_argument("--arch", default="signet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--forgery-weight", type=float, default=0.5)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=52,
        help="Held constant across points so writer count is the only variable",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=4,
        help="Validation is expensive on a large corpus and only picks the checkpoint here",
    )
    parser.add_argument("--work-dir", type=Path, default=ARTIFACT_ROOT / "ablation")
    parser.add_argument("--report", type=Path, default=DOCS_ROOT / "writer-count-ablation.md")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow-undertrained",
        action="store_true",
        help="Run even when the step budget cannot converge every point. Smoke tests only — "
        "the resulting curve is an artefact, not a measurement.",
    )
    run_ablation(parser.parse_args())


if __name__ == "__main__":
    main()
