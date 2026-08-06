"""Tests for the embedding models, losses, and verification metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.config import MODEL, PREPROCESS
from ml.embed.losses import ArcFaceHead, CombinedLoss, ForgeryTripletLoss, batch_hard_triplet
from ml.embed.models import build_model, embed_batch
from ml.eval.metrics import compute_metrics, equal_error_rate, roc_auc, tar_at_far

BATCH = 8


def _images(n: int = BATCH) -> torch.Tensor:
    return torch.rand(n, 1, *PREPROCESS.crop_size)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def test_signet_output_shape():
    model = build_model("signet")
    out = model(_images())
    assert out.shape == (BATCH, MODEL.embedding_dim)


def test_hybrid_output_shape():
    model = build_model("hybrid")
    out = model(_images(4))
    assert out.shape == (4, MODEL.embedding_dim)


def test_embed_batch_returns_unit_vectors_and_restores_mode():
    model = build_model("signet")
    model.train()
    out = embed_batch(model, _images())
    assert torch.allclose(out.norm(dim=1), torch.ones(BATCH), atol=1e-5)
    assert model.training, "embed_batch must restore the model's original mode"


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("resnet9000")


def test_pretrained_is_blocked_on_the_production_track():
    """The licence policy is enforced in code, not just in documentation."""
    with pytest.raises(ValueError, match="Track B"):
        build_model("hybrid", pretrained=True, track="b")


def test_model_is_deterministic_in_eval_mode():
    model = build_model("signet").eval()
    images = _images()
    with torch.no_grad():
        assert torch.allclose(model(images), model(images))


# --------------------------------------------------------------------------
# Spatial transformer (opt-in)
# --------------------------------------------------------------------------


def test_spatial_transformer_is_off_by_default():
    """Preprocessing already removes scale; the STN must be opted into."""
    assert build_model("signet").stn is None
    assert build_model("signet", spatial_transformer=True).stn is not None


def test_spatial_transformer_starts_as_the_identity():
    """Training must begin from 'change nothing', not from a random warp."""
    model = build_model("signet", spatial_transformer=True).eval()
    images = _images(2)
    with torch.no_grad():
        assert torch.allclose(model.stn(images), images, atol=1e-3)


def test_spatial_transformer_warp_is_bounded():
    """An unbounded STN can collapse every input onto one patch.

    That drives training loss down while destroying the signal — a failure that
    looks like success in the loss curve, so the bound is asserted here.
    """
    model = build_model("signet", spatial_transformer=True)
    stn = model.stn
    # Drive the regressor hard in both directions and confirm saturation.
    with torch.no_grad():
        stn.regressor.weight.fill_(50.0)
        out_high = stn(_images(2))
        stn.regressor.weight.fill_(-50.0)
        out_low = stn(_images(2))
    assert torch.isfinite(out_high).all() and torch.isfinite(out_low).all()
    # Some ink must survive: a collapsed transform would blank the image.
    assert out_high.abs().sum() > 0 and out_low.abs().sum() > 0


def test_spatial_transformer_model_trains():
    model = build_model("signet", spatial_transformer=True)
    out = model(_images(2))
    out.sum().backward()
    assert any(p.grad is not None for p in model.stn.parameters())


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def test_arcface_logits_shape_and_margin_effect():
    head = ArcFaceHead(16, n_writers=5)
    embeddings = torch.randn(6, 16)
    labels = torch.randint(0, 5, (6,))
    logits = head(embeddings, labels)
    assert logits.shape == (6, 5)
    assert torch.isfinite(logits).all()

    # The margin must make the target class harder than plain cosine would.
    no_margin = ArcFaceHead(16, n_writers=5, margin=0.0)
    no_margin.weight.data.copy_(head.weight.data)
    plain = no_margin(embeddings, labels)
    target = labels.view(-1, 1)
    assert (logits.gather(1, target) <= plain.gather(1, target) + 1e-4).all()


def test_triplet_loss_is_zero_when_margin_is_satisfied():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])
    assert ForgeryTripletLoss(margin=0.3)(anchor, positive, negative).item() == 0.0


def test_triplet_loss_is_positive_when_negative_is_too_close():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.0, 1.0]])
    negative = torch.tensor([[1.0, 0.01]])
    assert ForgeryTripletLoss(margin=0.3)(anchor, positive, negative).item() > 0.0


def test_batch_hard_triplet_handles_a_batch_with_no_valid_anchor():
    """One sample per writer means no positive pair — must not crash."""
    embeddings = torch.randn(4, 8, requires_grad=True)
    writers = torch.tensor([0, 1, 2, 3])
    genuine = torch.tensor([True, True, True, True])
    loss = batch_hard_triplet(embeddings, writers, genuine)
    assert loss.item() == 0.0
    loss.backward()  # graph must stay connected


def test_combined_loss_runs_and_reports_components():
    criterion = CombinedLoss(embedding_dim=32, n_writers=4)
    embeddings = torch.randn(8, 32, requires_grad=True)
    writers = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    genuine = torch.tensor([True, False, True, True, False, True, True, True])

    loss, parts = criterion(embeddings, writers, genuine)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "identity", "forgery"}
    assert embeddings.grad is not None


def test_combined_loss_ignores_forgeries_for_identity():
    """A forgery must never teach the identity head that it IS the writer."""
    criterion = CombinedLoss(embedding_dim=16, n_writers=3)
    embeddings = torch.randn(4, 16)
    writers = torch.tensor([0, 0, 1, 1])
    all_forged = torch.tensor([False, False, False, False])
    _, parts = criterion(embeddings, writers, all_forged)
    assert parts["identity"] == 0.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_perfectly_separated_scores_give_zero_eer():
    genuine = np.array([0.9, 0.95, 0.99])
    impostor = np.array([0.1, 0.2, 0.05])
    eer, _ = equal_error_rate(genuine, impostor)
    assert eer == pytest.approx(0.0, abs=1e-6)


def test_identical_distributions_give_eer_near_a_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 4000)
    eer, _ = equal_error_rate(scores, rng.normal(0, 1, 4000))
    assert 0.4 < eer < 0.6


def test_eer_threshold_actually_balances_the_errors():
    rng = np.random.default_rng(1)
    genuine = rng.normal(0.7, 0.15, 3000)
    impostor = rng.normal(0.3, 0.15, 3000)
    eer, threshold = equal_error_rate(genuine, impostor)
    far = (impostor >= threshold).mean()
    frr = (genuine < threshold).mean()
    assert abs(far - frr) < 0.02
    assert abs((far + frr) / 2 - eer) < 0.02


def test_tar_at_far_respects_the_constraint():
    rng = np.random.default_rng(2)
    genuine = rng.normal(0.8, 0.1, 5000)
    impostor = rng.normal(0.2, 0.1, 5000)
    tar, threshold = tar_at_far(genuine, impostor, 0.01)
    assert (impostor >= threshold).mean() <= 0.011
    assert tar == pytest.approx((genuine >= threshold).mean(), abs=1e-6)


def test_far_targets_finer_than_the_impostor_set_are_flagged():
    """A 0.1% FAR claim needs >= 1000 impostor pairs; smaller sets must say so.

    The function still returns a number (the FAR = 0 operating point), which is
    exactly why the caveat has to be surfaced rather than inferred.
    """
    rng = np.random.default_rng(5)
    metrics = compute_metrics(
        rng.normal(0.8, 0.1, 500), rng.normal(0.2, 0.1, 100), (0.10, 0.01, 0.001)
    )
    assert metrics.far_resolution == pytest.approx(0.01)
    assert metrics.unresolvable_far_targets == [0.001]
    assert "0.001" in metrics.to_dict()["unresolvable_far_targets"]


def test_far_targets_are_not_flagged_when_the_set_is_large_enough():
    rng = np.random.default_rng(6)
    metrics = compute_metrics(
        rng.normal(0.8, 0.1, 5000), rng.normal(0.2, 0.1, 20000), (0.01, 0.001)
    )
    assert metrics.unresolvable_far_targets == []


def test_auc_is_one_for_perfect_separation_and_half_for_noise():
    assert roc_auc(np.array([1.0, 0.9]), np.array([0.1, 0.2])) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    assert roc_auc(rng.normal(size=2000), rng.normal(size=2000)) == pytest.approx(0.5, abs=0.05)


def test_compute_metrics_reports_every_requested_far_target():
    rng = np.random.default_rng(4)
    metrics = compute_metrics(
        rng.normal(0.7, 0.12, 2000), rng.normal(0.3, 0.12, 2000), (0.10, 0.01, 0.001)
    )
    assert set(metrics.tar_at_far) == {0.10, 0.01, 0.001}
    # A tighter FAR can never buy a higher TAR.
    assert metrics.tar_at_far[0.10] >= metrics.tar_at_far[0.01] >= metrics.tar_at_far[0.001]
    assert 0.0 <= metrics.eer <= 1.0
    assert metrics.to_dict()["n_genuine"] == 2000


def test_compute_metrics_requires_both_populations():
    with pytest.raises(ValueError, match="impostor"):
        compute_metrics([0.9, 0.8], [])
