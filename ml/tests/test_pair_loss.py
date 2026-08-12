"""Tests for the global-threshold pair objective.

Why this term exists: with several specimens on file, a score can be expressed
relative to how consistently that customer signs, and inconsistent absolute
scales across writers do not matter. With **one** specimen there is no such
baseline, every writer is judged on the same absolute cosine scale, and nothing
in the identity or triplet objectives was asking the model to provide one.

The property under test is therefore not "does the loss go down" but "does it
punish an embedding where two writers need different thresholds".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.config import MODEL
from ml.embed.losses import CombinedLoss, GlobalThresholdPairLoss

DIM = 32


def _writer_cluster(centre: np.ndarray, spread: float, n: int, rng) -> np.ndarray:
    out = centre[None, :] + spread * rng.normal(size=(n, len(centre)))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _batch(tight_spread: float, loose_spread: float, seed: int = 0):
    """Two writers, each internally separable, at different absolute scales.

    Writer 0's genuine samples sit very close together; writer 1's are spread
    out. Both are separable from their own forgeries, but the cosine at which
    the boundary falls differs — exactly the situation a single shared
    threshold cannot handle.
    """
    rng = np.random.default_rng(seed)
    a, b = np.zeros(DIM), np.zeros(DIM)
    a[0], b[1] = 1.0, 1.0

    emb, writers, genuine = [], [], []
    for w, (centre, spread) in enumerate(((a, tight_spread), (b, loose_spread))):
        emb.append(_writer_cluster(centre, spread, 3, rng))
        writers += [w] * 3
        genuine += [True] * 3
        # Forgeries: same direction, further out.
        emb.append(_writer_cluster(centre, spread * 3.0, 2, rng))
        writers += [w] * 2
        genuine += [False] * 2

    return (
        torch.tensor(np.vstack(emb), dtype=torch.float32),
        torch.tensor(writers),
        torch.tensor(genuine),
    )


def test_mismatched_scales_cost_more_than_matched_ones():
    """The property the term is for.

    Two writers separable at *different* absolute cosines must score worse than
    two separable at the same one, even though both are internally separable.
    """
    loss = GlobalThresholdPairLoss()

    matched = loss(*_batch(tight_spread=0.15, loose_spread=0.15))
    mismatched = loss(*_batch(tight_spread=0.02, loose_spread=0.45))

    assert mismatched > matched


def test_the_threshold_is_learnable():
    loss = GlobalThresholdPairLoss()
    before = loss.threshold

    optimiser = torch.optim.SGD(loss.parameters(), lr=1.0)
    for _ in range(20):
        optimiser.zero_grad()
        loss(*_batch(0.1, 0.1)).backward()
        optimiser.step()

    assert loss.threshold != before


def test_gradient_reaches_the_embeddings():
    """Otherwise the term only tunes its own scale and teaches the model nothing."""
    emb, writers, genuine = _batch(0.1, 0.3)
    emb.requires_grad_(True)
    GlobalThresholdPairLoss()(emb, writers, genuine).backward()

    assert emb.grad is not None
    assert float(emb.grad.abs().sum()) > 0


def test_a_batch_with_no_usable_pair_stays_connected():
    """One sample per writer: no positive pairs, and it must not crash."""
    emb = torch.randn(2, DIM, requires_grad=True)
    out = GlobalThresholdPairLoss()(emb, torch.tensor([0, 1]), torch.tensor([True, True]))
    out.backward()
    assert emb.grad is not None


def test_classes_are_balanced():
    """Negatives outnumber positives heavily; unweighted, the cheapest move is
    to call everything a non-match."""
    loss = GlobalThresholdPairLoss()
    emb, writers, genuine = _batch(0.1, 0.1)

    # An embedding where every pair is identical: all positives satisfied, all
    # negatives violated. If negatives dominated the objective this would be
    # cheap; balanced, it is expensive.
    collapsed = torch.ones(len(writers), DIM)
    collapsed = collapsed / collapsed.norm(dim=1, keepdim=True)

    assert loss(collapsed, writers, genuine) > loss(emb, writers, genuine)


# --------------------------------------------------------------------------
# Integration with the combined objective
# --------------------------------------------------------------------------


def test_combined_loss_reports_the_pair_term_and_threshold():
    criterion = CombinedLoss(DIM, 2, MODEL)
    emb, writers, genuine = _batch(0.1, 0.2)

    _total, parts = criterion(emb, writers, genuine)

    assert "pair" in parts
    assert "threshold" in parts
    assert "identity" in parts and "forgery" in parts


def test_pair_term_is_off_when_its_weight_is_zero():
    from dataclasses import replace

    cfg = replace(MODEL, pair_loss_weight=0.0)
    criterion = CombinedLoss(DIM, 2, cfg)
    emb, writers, genuine = _batch(0.1, 0.2)

    total, parts = criterion(emb, writers, genuine)

    assert "pair" not in parts
    # And the total is exactly the old objective, so an existing recipe is
    # reproducible.
    expected = (1 - cfg.forgery_loss_weight) * parts["identity"] + (
        cfg.forgery_loss_weight * parts["forgery"]
    )
    assert float(total.detach()) == pytest.approx(expected, abs=1e-5)
