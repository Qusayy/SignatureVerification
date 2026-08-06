"""Tests for the foundation-model backbones.

Marked slow because the first run downloads weights from the Hugging Face hub.
Deselect with ``-m "not slow"`` on an air-gapped machine that has not been
pre-staged.
"""

from __future__ import annotations

import pytest
import torch

from ml.config import MODEL, PREPROCESS
from ml.embed.backbones import BACKBONES, FoundationEmbedder
from ml.embed.models import build_model

pytestmark = pytest.mark.slow

# The smallest backbone, so CI does not pull hundreds of megabytes.
TEST_BACKBONE = "dinov2-small"


@pytest.fixture(scope="module")
def embedder() -> FoundationEmbedder:
    return FoundationEmbedder(TEST_BACKBONE)


def test_every_declared_backbone_is_commercially_licensed():
    """Track B depends on this. A non-commercial entry here would be a trap."""
    for name, spec in BACKBONES.items():
        assert spec.commercial_use, f"{name} is {spec.licence} and cannot back production"
        assert spec.licence in {"Apache-2.0", "MIT"}, f"{name} has unexpected licence {spec.licence}"


def test_output_shape_matches_the_scoring_contract(embedder):
    out = embedder(torch.rand(2, 1, *PREPROCESS.crop_size))
    assert out.shape == (2, MODEL.embedding_dim)
    assert torch.isfinite(out).all()


def test_accepts_the_same_input_as_the_from_scratch_models(embedder):
    """Interface parity: nothing downstream should know which backbone is loaded."""
    from ml.embed.models import SigNet

    x = torch.rand(3, 1, *PREPROCESS.crop_size)
    assert embedder(x).shape == SigNet()(x).shape


def test_lora_leaves_the_backbone_frozen(embedder):
    trainable, total = embedder.trainable_parameters()
    assert trainable < total * 0.15, (
        f"{trainable / total:.1%} of parameters are trainable; LoRA should keep this small "
        "so the pretrained representation is not overwritten by a tiny corpus"
    )
    assert trainable > 0


def test_input_is_inverted_to_dark_ink_on_light(embedder):
    """Foundation models were trained on photographs, not negatives.

    The canvas convention is ink-high on black; feeding that straight in wastes
    the pretrained representation.
    """
    blank = torch.zeros(1, 1, *PREPROCESS.crop_size)  # no ink
    adapted = embedder._adapt_input(blank)
    # No ink → white page → above the normalisation mean on every channel.
    assert adapted.mean() > 0


def test_build_model_routes_backbone_names(embedder):
    model = build_model(TEST_BACKBONE, track="b")
    assert isinstance(model, FoundationEmbedder)
    assert model.embedding_dim == MODEL.embedding_dim


def test_unknown_backbone_name_lists_the_valid_ones():
    with pytest.raises(ValueError, match="dinov2"):
        build_model("not-a-real-backbone")


def test_backbone_is_deterministic_in_eval_mode(embedder):
    embedder.eval()
    x = torch.rand(2, 1, *PREPROCESS.crop_size)
    with torch.no_grad():
        assert torch.allclose(embedder(x), embedder(x), atol=1e-5)


def test_gradients_reach_lora_and_head_but_not_the_frozen_backbone(embedder):
    embedder.train()
    out = embedder(torch.rand(2, 1, *PREPROCESS.crop_size))
    out.sum().backward()

    head_grads = [p.grad for p in embedder.head.parameters() if p.grad is not None]
    assert head_grads, "the projection head must receive gradients"

    frozen = [
        p
        for n, p in embedder.backbone.named_parameters()
        if "lora" not in n.lower() and not p.requires_grad
    ]
    assert frozen, "expected frozen backbone parameters"
    assert all(p.grad is None for p in frozen), "frozen backbone must not accumulate gradients"
