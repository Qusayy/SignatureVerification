"""Foundation-model backbones with LoRA adapters.

The point of this module, in one sentence: with only a few hundred writers,
fine-tuning a large model pretrained on general images beats training a small
model from scratch — and unlike the obvious signature-specific weights, these
are commercially licensed.

**Licensing.** This is the distinction that matters and the one that catches
people out:

===========================  ==================  ====================
Asset                        Licence             Production use
===========================  ==================  ====================
SigNet / ``sigver`` weights  GPDS, non-commercial  ❌ Never
DINOv2                       Apache-2.0            ✅ Yes
SigLIP                       Apache-2.0            ✅ Yes
CLIP                         MIT                   ✅ Yes
ImageNet torchvision weights ImageNet terms        ⚠️  Avoid
===========================  ==================  ====================

Signature-*specific* pretrained weights are licence-poisoned. General vision
foundation models are not. So these backbones are Track B — they may back a
production model — while ``--pretrained`` on the from-scratch architectures
remains blocked.

**Why LoRA rather than full fine-tuning.** The backbone stays frozen and only
small low-rank adapters train. With a small corpus, full fine-tuning erases the
pretrained representation faster than it learns a signature-specific one; LoRA
keeps what the model already knows about strokes and edges. It also means a
training run fits comfortably on a single modern GPU.

**Air-gapped deployment.** Weights are downloaded from the Hugging Face hub on
first use. An organisation datacentre with no internet must pre-stage them: fetch once
on a connected machine, then set ``HF_HOME`` to the copied cache and
``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import MODEL, ModelConfig

__all__ = ["FoundationEmbedder", "BACKBONES", "BackboneSpec"]


@dataclass(frozen=True)
class BackboneSpec:
    """A pretrained backbone and everything needed to adapt it."""

    model_id: str
    licence: str
    commercial_use: bool
    # Modules LoRA attaches to. Names differ between architectures.
    lora_targets: tuple[str, ...]
    image_size: int = 224
    # ImageNet statistics for DINOv2; SigLIP was trained on [-1, 1].
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


BACKBONES: dict[str, BackboneSpec] = {
    "dinov2": BackboneSpec(
        model_id="facebook/dinov2-base",
        licence="Apache-2.0",
        commercial_use=True,
        lora_targets=("query", "key", "value"),
    ),
    "dinov2-small": BackboneSpec(
        model_id="facebook/dinov2-small",
        licence="Apache-2.0",
        commercial_use=True,
        lora_targets=("query", "key", "value"),
    ),
    "siglip": BackboneSpec(
        model_id="google/siglip-base-patch16-224",
        licence="Apache-2.0",
        commercial_use=True,
        lora_targets=("q_proj", "k_proj", "v_proj"),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class FoundationEmbedder(nn.Module):
    """A frozen vision foundation model plus LoRA, producing signature embeddings.

    Accepts the same input as the from-scratch models — a single-channel batch
    of preprocessed canvases in [0, 1] with ink high — and adapts it internally,
    so nothing downstream needs to know which backbone is in use.
    """

    def __init__(
        self,
        backbone: str = "dinov2",
        cfg: ModelConfig = MODEL,
        *,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}")

        self.spec = BACKBONES[backbone]
        self.backbone_name = backbone
        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim

        from transformers import AutoModel

        base = AutoModel.from_pretrained(self.spec.model_id)
        # SigLIP is a two-tower model; only the vision side is wanted.
        if hasattr(base, "vision_model"):
            base = base.vision_model

        if freeze_backbone:
            from peft import LoraConfig, get_peft_model

            for param in base.parameters():
                param.requires_grad = False
            base = get_peft_model(
                base,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(self.spec.lora_targets),
                    bias="none",
                ),
            )

        self.backbone = base
        hidden = self._infer_hidden_size()

        # Projection head onto the embedding space the scoring layer expects.
        # Trained from scratch regardless of whether the backbone is frozen.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.embedding_dim),
        )

        self.register_buffer("pixel_mean", torch.tensor(self.spec.mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(self.spec.std).view(1, 3, 1, 1))

    def _infer_hidden_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.spec.image_size, self.spec.image_size)
            out = self.backbone(pixel_values=dummy)
        return int(out.last_hidden_state.shape[-1])

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Single-channel signature canvas → the 3-channel tensor the backbone wants.

        Ink is inverted to dark-on-light first. Foundation models were trained
        on photographs, where objects are dark against a lighter background;
        feeding them a photographic negative wastes the pretrained
        representation for no reason.
        """
        x = 1.0 - x  # ink high → ink dark on white
        x = F.interpolate(
            x,
            size=(self.spec.image_size, self.spec.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=self._adapt_input(x))
        hidden = out.last_hidden_state

        # DINOv2 exposes a CLS token at position 0; SigLIP has no CLS, so mean
        # pooling over patches is the documented way to get a global vector.
        pooled = hidden[:, 0] if self.backbone_name.startswith("dinov2") else hidden.mean(dim=1)
        return self.head(pooled)

    def trainable_parameters(self) -> tuple[int, int]:
        """(trainable, total) parameter counts — LoRA should make this a small fraction."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
