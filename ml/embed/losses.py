"""Training objectives for the embedding model.

The combined objective has two jobs that pull in different directions:

* **ArcFace over writer identity** spreads different writers apart. It gives
  the embedding space its global structure and is what makes the model
  writer-independent — a new customer lands somewhere sensible without any
  retraining.
* **A forgery-aware triplet term** tightens each writer's own cluster against
  *skilled forgeries of that same writer*. This is the part that matters for
  the organisation. Random negatives are nearly free to separate; the forgery of the
  customer standing at the desk is the negative the system actually faces, so
  it is the negative trained against.

Training on identity alone produces a model that is excellent at telling
customer A from customer B and mediocre at telling customer A from someone
imitating customer A. That failure mode does not show up in an accuracy number
computed against random forgeries, which is why the evaluation harness reports
skilled-forgery EER separately.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import MODEL, ModelConfig

__all__ = [
    "ArcFaceHead",
    "ForgeryTripletLoss",
    "CombinedLoss",
    "GlobalThresholdPairLoss",
    "batch_hard_triplet",
]


class ArcFaceHead(nn.Module):
    """Additive angular margin softmax over writer identities.

    Used only during training; at inference the embedding is taken directly and
    this head is discarded, which is what keeps enrolment retraining-free.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_writers: int,
        *,
        scale: float = MODEL.arcface_scale,
        margin: float = MODEL.arcface_margin,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_writers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.n_writers = n_writers

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return logits ready for cross-entropy against ``labels``."""
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply the margin only where it keeps the target monotonic; beyond
        # (pi - margin) the penalty would fold back and push the sample toward
        # the wrong class.
        target = torch.cos(theta + self.margin)
        safe = theta + self.margin < math.pi
        target = torch.where(safe, target, cosine - self.margin * math.sin(self.margin))

        return self.scale * (one_hot * target + (1.0 - one_hot) * cosine)


class ForgeryTripletLoss(nn.Module):
    """Triplet loss whose negatives are skilled forgeries of the anchor writer.

    Operates on L2-normalised embeddings with squared Euclidean distance, which
    on the unit sphere is a monotone function of cosine distance — the same
    quantity the scoring layer thresholds at inference.
    """

    def __init__(self, margin: float = MODEL.triplet_margin):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        a, p, n = (F.normalize(t) for t in (anchor, positive, negative))
        d_pos = (a - p).pow(2).sum(dim=1)
        d_neg = (a - n).pow(2).sum(dim=1)
        return F.relu(d_pos - d_neg + self.margin).mean()


def batch_hard_triplet(
    embeddings: torch.Tensor,
    writer_labels: torch.Tensor,
    is_genuine: torch.Tensor,
    margin: float = MODEL.triplet_margin,
) -> torch.Tensor:
    """Batch-hard mining variant, forming triplets inside a batch.

    For each genuine anchor: the hardest positive is the *furthest* genuine
    sample of the same writer, and the hardest negative is the *closest*
    forgery of that same writer, falling back to the closest sample of another
    writer when the batch happens to contain no forgery for them.

    Mining inside the batch avoids materialising an explicit triplet dataset,
    which for N writers with F forgeries each grows unmanageably fast.
    """
    emb = F.normalize(embeddings)
    dist = torch.cdist(emb, emb, p=2).pow(2)

    same_writer = writer_labels[:, None] == writer_labels[None, :]
    genuine_col = is_genuine[None, :].expand_as(same_writer)
    eye = torch.eye(len(emb), dtype=torch.bool, device=emb.device)

    pos_mask = same_writer & genuine_col & ~eye
    neg_mask_forgery = same_writer & ~genuine_col
    neg_mask_other = ~same_writer

    anchors = is_genuine & pos_mask.any(dim=1)
    if not anchors.any():
        return embeddings.sum() * 0.0  # keeps the graph connected

    hardest_pos = (dist * pos_mask).max(dim=1).values

    big = dist.max().detach() + 1.0
    forgery_dist = torch.where(neg_mask_forgery, dist, big)
    other_dist = torch.where(neg_mask_other, dist, big)
    has_forgery = neg_mask_forgery.any(dim=1)
    hardest_neg = torch.where(has_forgery, forgery_dist.min(dim=1).values, other_dist.min(dim=1).values)

    losses = F.relu(hardest_pos - hardest_neg + margin)[anchors]
    return losses.mean()


class GlobalThresholdPairLoss(nn.Module):
    """Force one decision threshold to work for every writer.

    **Why this exists, and why it matters most with a single specimen.**

    The triplet term optimises a *relative* ordering: for each writer, genuine
    samples should sit closer than forgeries. It says nothing about where the
    boundary falls in absolute terms, so two writers can each be perfectly
    separable and yet need different thresholds — one at cosine 0.90, another
    at 0.75.

    With several specimens on file that is recoverable at scoring time: the
    customer's own specimen agreement supplies a per-customer baseline, which
    is worth ~15 EER points (see :mod:`ml.scoring.compare`). With **one**
    specimen there is no such baseline to compute, the score falls back to a
    population median, and every writer is judged on the same absolute scale —
    the exact thing nothing in the objective was asking the model to provide.

    This term asks for it directly. Every pair in the batch is classified as
    same-writer-genuine or not, through a single learnable scale and bias
    shared by all writers::

        p(same) = sigmoid(scale * (cosine - bias))

    Because ``scale`` and ``bias`` are global, the only way to drive the loss
    down is to make cosine values mean the same thing across writers. That is
    precisely the property single-specimen verification depends on.

    After training ``bias`` is a useful artefact in itself: it is the model's
    own estimate of the natural global threshold.
    """

    def __init__(self, scale: float = 10.0, bias: float = 0.5):
        super().__init__()
        # Parameterised in log space so the scale stays positive under SGD.
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(scale))))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    @property
    def threshold(self) -> float:
        """The learned global decision boundary, as a cosine."""
        return float(self.bias.detach())

    def forward(
        self,
        embeddings: torch.Tensor,
        writer_labels: torch.Tensor,
        is_genuine: torch.Tensor,
    ) -> torch.Tensor:
        emb = F.normalize(embeddings)
        cosine = emb @ emb.t()

        same_writer = writer_labels[:, None] == writer_labels[None, :]
        genuine_row = is_genuine[:, None]
        genuine_col = is_genuine[None, :]
        eye = torch.eye(len(emb), dtype=torch.bool, device=emb.device)

        # Positive: two genuine samples from one writer — what a real
        # verification looks like when the customer is who they say.
        positive = same_writer & genuine_row & genuine_col & ~eye
        # Negative: this writer's forgeries (the hard case) and other writers'
        # signatures (which keeps the absolute scale honest; without them the
        # model can satisfy the loss by inflating every cosine).
        negative = (same_writer & (genuine_row ^ genuine_col)) | ~same_writer

        pairs = positive | negative
        # Upper triangle only: the matrix is symmetric, so counting both halves
        # would double-weight every pair for no benefit.
        pairs = pairs & torch.triu(torch.ones_like(pairs), diagonal=1).bool()
        if not pairs.any():
            return embeddings.sum() * 0.0

        logits = self.log_scale.exp() * (cosine[pairs] - self.bias)
        targets = positive[pairs].float()

        # Negatives vastly outnumber positives in a P x K batch, so weight the
        # two classes evenly. Otherwise the cheapest way down is to call
        # everything a non-match.
        n_pos = targets.sum().clamp(min=1.0)
        n_neg = (targets.numel() - targets.sum()).clamp(min=1.0)
        weights = torch.where(targets > 0, n_neg / n_pos, torch.ones_like(targets))
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


class CombinedLoss(nn.Module):
    """ArcFace identity loss plus the forgery-aware triplet term.

    ``cfg.forgery_loss_weight`` mirrors the lambda in SigNet-F: it controls how
    much of the objective is spent on separating forgeries versus separating
    writers. Raise it when skilled-forgery EER is the failing metric.
    """

    def __init__(self, embedding_dim: int, n_writers: int, cfg: ModelConfig = MODEL):
        super().__init__()
        self.cfg = cfg
        self.arcface = ArcFaceHead(
            embedding_dim, n_writers, scale=cfg.arcface_scale, margin=cfg.arcface_margin
        )
        self.identity_loss = nn.CrossEntropyLoss()
        self.pair_loss = GlobalThresholdPairLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        writer_labels: torch.Tensor,
        is_genuine: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the scalar loss and a dict of components for logging."""
        # Only genuine samples define a writer's identity. Training the
        # identity head on forgeries would teach it that a forgery *is* the
        # writer, which is precisely backwards.
        genuine = is_genuine.bool()
        if genuine.any():
            logits = self.arcface(embeddings[genuine], writer_labels[genuine])
            identity = self.identity_loss(logits, writer_labels[genuine])
        else:
            identity = embeddings.sum() * 0.0

        forgery = batch_hard_triplet(embeddings, writer_labels, genuine, self.cfg.triplet_margin)

        w = self.cfg.forgery_loss_weight
        total = (1.0 - w) * identity + w * forgery

        # Added rather than blended into the convex combination above, so the
        # existing identity/forgery balance is untouched and this run stays
        # comparable with earlier ones.
        parts = {
            "identity": float(identity.detach()),
            "forgery": float(forgery.detach()),
        }
        if self.cfg.pair_loss_weight > 0.0:
            pair = self.pair_loss(embeddings, writer_labels, genuine)
            total = total + self.cfg.pair_loss_weight * pair
            parts["pair"] = float(pair.detach())
            parts["threshold"] = self.pair_loss.threshold

        parts["loss"] = float(total.detach())
        return total, parts
