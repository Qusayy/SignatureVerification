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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import MODEL, ModelConfig

__all__ = ["ArcFaceHead", "ForgeryTripletLoss", "CombinedLoss"]


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
        return total, {
            "loss": float(total.detach()),
            "identity": float(identity.detach()),
            "forgery": float(forgery.detach()),
        }
