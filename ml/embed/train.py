"""Train the signature embedding model.

Usage::

    python -m ml.embed.train --arch signet --epochs 30 --track a
    python -m ml.embed.train --arch hybrid --epochs 60 --track b   # production

``--track b`` enforces the licensing policy before the first optimizer step:
every training record must come from a commercially clean source, and no
pretrained initialisation is allowed. The resulting checkpoint carries a
provenance record that ``ml/embed/provenance.py --gate`` can verify at deploy
time.

Validation reports EER against skilled forgeries — not accuracy, and not EER
against random forgeries, both of which look far better than the system really
is.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.config import (
    ARTIFACT_ROOT,
    MODEL,
    SCORING,
    TRACK_A,
    TRACK_B,
    TRAIN,
    ModelConfig,
    resolve_device,
)
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, assert_no_leakage, assert_track_b
from ml.embed.dataset import SignatureDataset, WriterBatchSampler, collate
from ml.embed.losses import CombinedLoss
from ml.embed.models import build_model
from ml.embed.provenance import Provenance, weights_id
from ml.eval.metrics import compute_metrics

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@torch.no_grad()
def embed_split(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, list]:
    """Embed every sample in a loader. Returns (embeddings, records)."""
    model.eval()
    vectors: list[np.ndarray] = []
    indices: list[int] = []
    for batch in loader:
        out = model(batch["image"].to(device))
        vectors.append(F.normalize(out, p=2, dim=1).cpu().numpy())
        indices.extend(batch["index"].tolist())
    return np.concatenate(vectors), indices


def validation_scores(
    embeddings: np.ndarray,
    records: list,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build genuine and skilled-forgery comparison scores for a split.

    For each writer, every genuine sample is compared against that writer's
    other genuine samples (genuine population) and against that writer's
    forgeries (impostor population). Comparisons are always *within* a writer,
    which is what the live system does: it never asks "is this anyone's
    signature", only "is this this customer's signature".
    """
    by_writer: dict[str, dict[str, list[int]]] = {}
    for position, index in enumerate(indices):
        record = records[index]
        bucket = by_writer.setdefault(record.signer_id, {"genuine": [], "forgery": []})
        key = "genuine" if record.label == "genuine" else "forgery"
        bucket[key].append(position)

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    for bucket in by_writer.values():
        gen, forg = bucket["genuine"], bucket["forgery"]
        if len(gen) < 2:
            continue
        for a_i, a in enumerate(gen):
            for b in gen[a_i + 1 :]:
                genuine_scores.append(float(embeddings[a] @ embeddings[b]))
            for f in forg:
                impostor_scores.append(float(embeddings[a] @ embeddings[f]))

    return np.asarray(genuine_scores), np.asarray(impostor_scores)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = Manifest.load(args.manifest)
    assert_no_leakage(manifest)

    train_records = manifest.by_split("train")
    if not train_records:
        raise SystemExit("No training records. Run: python -m ml.data.manifest split")

    # --- Licence gate, before any work is done -----------------------------
    if args.track == "b":
        assert_track_b(train_records)
        if args.pretrained:
            raise SystemExit(
                "--pretrained cannot be combined with --track b. See docs/licensing.md."
            )
    licence_track = TRACK_B.name if args.track == "b" else TRACK_A.name

    train_ds = SignatureDataset(
        manifest, "train", augment=not args.no_augment, seed=args.seed, cache_dir=args.cache_dir
    )
    val_ds = SignatureDataset(
        manifest,
        "val",
        augment=False,
        writer_index=train_ds.writer_index,
        cache_dir=args.cache_dir,
    )

    sampler = WriterBatchSampler(
        train_ds,
        writers_per_batch=args.writers_per_batch,
        samples_per_writer=args.samples_per_writer,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )

    from dataclasses import replace

    model_cfg = MODEL
    if args.forgery_weight is not None:
        model_cfg = replace(MODEL, forgery_loss_weight=args.forgery_weight)

    model = build_model(args.arch, model_cfg, pretrained=args.pretrained, track=args.track).to(device)
    criterion = CombinedLoss(model.embedding_dim, train_ds.n_writers, model_cfg).to(device)

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        momentum=TRAIN.momentum,
        weight_decay=TRAIN.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.out or ARTIFACT_ROOT / f"{args.arch}_track_{args.track}.pt")

    print(
        f"Training {args.arch} on {len(train_ds)} images / {train_ds.n_writers} writers "
        f"({device}, track {args.track}). Validating on {len(val_ds)} images."
    )

    best_eer = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        totals: Counter = Counter()
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            writers = batch["writer"].to(device)
            genuine = batch["is_genuine"].to(device)

            embeddings = model(images)
            loss, parts = criterion(embeddings, writers, genuine)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip the criterion too. The ArcFace weight matrix is in the
            # optimizer (see above) but was excluded here, leaving the one
            # parameter with the largest and spikiest gradients unclipped.
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()), 5.0
            )
            optimizer.step()

            totals.update(parts)
            n_batches += 1

        scheduler.step()
        means = {k: v / max(n_batches, 1) for k, v in totals.items()}

        entry = {"epoch": epoch, "seconds": round(time.time() - started, 1), **{k: round(v, 4) for k, v in means.items()}}

        if epoch % args.val_every == 0 and len(val_ds):
            embeddings, indices = embed_split(model, val_loader, device)
            genuine, impostor = validation_scores(embeddings, val_ds.records, indices)
            if len(genuine) and len(impostor):
                metrics = compute_metrics(genuine, impostor, SCORING.far_targets)
                entry["val_eer_skilled"] = round(metrics.eer, 4)
                entry["val_tar_at_far_1pct"] = round(metrics.tar_at_far[0.01], 4)
                if metrics.eer < best_eer:
                    best_eer = metrics.eer
                    _save(
                        checkpoint_path, model, criterion, args, manifest, train_ds,
                        licence_track, metrics.eer, model_cfg,
                    )
                    entry["saved"] = True

        history.append(entry)
        print(json.dumps(entry))

    if best_eer == float("inf"):  # no validation split — save the final weights
        _save(
            checkpoint_path, model, criterion, args, manifest, train_ds,
            licence_track, None, model_cfg,
        )

    (checkpoint_path.with_suffix(".history.json")).write_text(json.dumps(history, indent=2))
    print(f"\nBest validation EER (skilled forgeries): {best_eer:.4f}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    return checkpoint_path


def _save(
    path: Path,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    args: argparse.Namespace,
    manifest: Manifest,
    dataset: SignatureDataset,
    licence_track: str,
    val_eer: float | None,
    model_cfg: ModelConfig,
) -> None:
    train_records = manifest.by_split("train")
    provenance = Provenance(
        licence_track=licence_track,
        sources=sorted({r.source for r in train_records}),
        architecture=args.arch,
        n_writers=dataset.n_writers,
        n_train_images=len(train_records),
        train_signers=len({r.signer_id for r in train_records}),
        scripts=dict(Counter(r.script for r in train_records)),
        pretrained_init="imagenet/resnet34" if args.pretrained else None,
        manifest_path=str(args.manifest),
        notes=args.notes,
        hyperparameters={
            "lr": args.lr,
            "epochs": args.epochs,
            "seed": args.seed,
            "writers_per_batch": args.writers_per_batch,
            "samples_per_writer": args.samples_per_writer,
            "batches_per_epoch": args.batches_per_epoch,
            "forgery_weight": model_cfg.forgery_loss_weight,
            "augment": not args.no_augment,
        },
    )
    state = model.state_dict()
    torch.save(
        {
            "model_state": state,
            "criterion_state": criterion.state_dict(),
            "architecture": args.arch,
            "embedding_dim": model.embedding_dim,
            "writer_index": dataset.writer_index,
            # `model_cfg`, not `MODEL`: the module default records
            # forgery_loss_weight=0.95 even for a run that passed 0.5, as every
            # checkpoint in artifacts/ demonstrates.
            "config": {
                "model": dict(model_cfg.__dict__),
                "crop_size": list(dataset[0]["image"].shape[1:]),
            },
            "weights_id": weights_id(state),
            "val_eer_skilled": val_eer,
            "provenance": provenance.to_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the signature embedding model")
    parser.add_argument("--arch", choices=["signet", "hybrid"], default="signet")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--writers-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-writer", type=int, default=4)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help=(
            "Fix the number of gradient steps per epoch. Defaults to scaling with corpus "
            "size. Set it explicitly for writer-count ablations, otherwise a larger corpus "
            "silently gets more training as well as more writers, and the two effects "
            "cannot be separated."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=TRAIN.num_workers)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-every", type=int, default=TRAIN.val_every)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache preprocessed canvases")
    parser.add_argument(
        "--track",
        choices=["a", "b"],
        default="a",
        help="a = research/POC, b = production (enforces the licensing policy)",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="ImageNet stem for --arch hybrid (Track A only)"
    )
    parser.add_argument(
        "--forgery-weight",
        type=float,
        default=None,
        help=(
            "Weight on the forgery-triplet term vs writer identity, 0-1 "
            f"(default {MODEL.forgery_loss_weight}). Raise it when skilled-forgery EER is the "
            "failing metric; lower it when the embedding space is not separating writers at all. "
            "Re-tune on real data — the value that works on synthetic signatures will not transfer."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--notes", default="")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
