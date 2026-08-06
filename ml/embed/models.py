"""Stage B — writer-independent signature embedding models.

Both architectures map a preprocessed signature to a fixed-length vector where
distance means "different writer". Writer-independence is the point: enrolling
a new customer must never require retraining, because a deployment site onboards
customers daily.

``signet``
    The Hafemann et al. CNN. Small, fast on CPU, thoroughly documented, and the
    sensible baseline to build the evaluation harness against. Note the
    architecture is BSD-licensed but the *published weights* are GPDS-trained
    and therefore Track A — see ``docs/licensing.md``.

``hybrid``
    A CNN stem for local stroke texture followed by a transformer encoder for
    global shape, mirroring the design that current published work (HTCSigNet,
    PAST, SignatureGuard) converges on. Slower, and worth the cost only once
    there is enough real data to train it.

ImageNet-pretrained stems are available for the hybrid but default to off:
ImageNet's own terms are research-oriented, so a Track B run starts from
scratch. :func:`build_model` refuses the combination rather than leaving it to
be noticed at procurement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import MODEL, PREPROCESS, ModelConfig

__all__ = ["SigNet", "HybridSigNet", "build_model", "embed_batch"]


# --------------------------------------------------------------------------
# SigNet baseline
# --------------------------------------------------------------------------


class SpatialTransformer(nn.Module):
    """Learned geometric normalisation, applied before the feature extractor.

    Predicts an affine warp per image and resamples. In principle this cleans
    up whatever geometric nuisance the preprocessing missed.

    **Default off, deliberately.** Two reasons:

    1. Moment normalisation in :mod:`ml.preprocess.pipeline` already removes
       absolute scale to within 0.3% (see :mod:`ml.eval.diagnostics`), so the
       main job an STN was wanted for is done.
    2. An unconstrained STN can learn to normalise away a writer's *slant*,
       which is discriminative and which the preprocessing preserves on
       purpose. That would quietly cost accuracy in a way no shape metric
       reveals.

    It is kept because it may still earn its place on real real-world captures,
    where geometric nuisance is larger than in synthetic data. Enable it and
    measure; do not assume it helps. The transform is initialised to the
    identity and the predicted warp is bounded, so at worst it starts as a
    no-op.
    """

    def __init__(self, input_size: tuple[int, int], max_shift: float = 0.15):
        super().__init__()
        self.max_shift = max_shift

        self.localiser = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),
        )
        self.regressor = nn.Linear(32 * 4 * 6, 6)

        # Start as the identity so training begins from "change nothing".
        nn.init.zeros_(self.regressor.weight)
        self.regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.regressor(self.localiser(x).flatten(1)).view(-1, 2, 3)

        # Bound the warp. An unbounded STN can collapse every input onto the
        # same patch, which drives the training loss down while destroying the
        # signal — a failure that looks like success in the loss curve.
        identity = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=x.device).view(1, 2, 3)
        theta = identity + torch.tanh(theta - identity) * self.max_shift

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


class SigNet(nn.Module):
    """Convolutional feature extractor following Hafemann et al.

    Input is a single-channel image of ``PREPROCESS.crop_size``, values in
    [0, 1] with ink high.
    """

    def __init__(
        self,
        cfg: ModelConfig = MODEL,
        input_size: tuple[int, int] | None = None,
        *,
        spatial_transformer: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        input_size = input_size or PREPROCESS.crop_size

        self.stn = SpatialTransformer(input_size) if spatial_transformer else None

        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Derive the flattened width from a dummy pass rather than hard-coding
        # it, so changing PREPROCESS.crop_size does not silently break the net.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_size)
            flat = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.embedding = nn.Linear(2048, cfg.embedding_dim)
        self.embedding_dim = cfg.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalised embeddings. Callers normalise where needed."""
        if self.stn is not None:
            x = self.stn(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return self.embedding(x)


# --------------------------------------------------------------------------
# CNN + transformer hybrid
# --------------------------------------------------------------------------


class HybridSigNet(nn.Module):
    """ResNet stem for stroke texture, transformer encoder for global shape.

    The stem keeps a moderate stride so the token grid stays informative: a
    signature is mostly whitespace, and pooling it too aggressively before the
    attention layers throws away exactly the fine stroke detail that separates
    a skilled forgery from a genuine sample.
    """

    def __init__(
        self,
        cfg: ModelConfig = MODEL,
        input_size: tuple[int, int] | None = None,
        *,
        pretrained: bool = False,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
    ):
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34

        self.cfg = cfg
        input_size = input_size or PREPROCESS.crop_size

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        # Single-channel input: average the RGB filters so a pretrained stem
        # keeps its learned edge detectors instead of being reinitialised.
        stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                stem.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = stem

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        stem_channels = 256  # resnet34 layer3 output

        self.project = nn.Conv2d(stem_channels, d_model, kernel_size=1)

        with torch.no_grad():
            dummy = self.project(self.stem(torch.zeros(1, 1, *input_size)))
            _, _, gh, gw = dummy.shape
        self.grid = (gh, gw)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would be
        # silently disabled with a warning; turn it off explicitly.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.embedding = nn.Linear(d_model, cfg.embedding_dim)
        self.embedding_dim = cfg.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(self.stem(x))
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : h * w + 1]
        tokens = self.encoder(tokens)
        return self.embedding(self.norm(tokens[:, 0]))


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_model(
    architecture: str = MODEL.architecture,
    cfg: ModelConfig = MODEL,
    *,
    pretrained: bool = False,
    track: str = "a",
    spatial_transformer: bool = False,
) -> nn.Module:
    """Construct an embedding model.

    Args:
        architecture: ``"signet"``, ``"hybrid"``, or a foundation backbone name
            from :data:`ml.embed.backbones.BACKBONES` (``"dinov2"``,
            ``"siglip"``, …).
        pretrained: use ImageNet weights for the hybrid stem. Rejected for
            Track B, where every weight must have a commercially clean origin.
            Note this flag does *not* govern the foundation backbones — those
            are pretrained by definition, and permissively licensed, so they
            are allowed on Track B. See :mod:`ml.embed.backbones`.
        track: ``"a"`` (research) or ``"b"`` (production).
    """
    from ml.embed.backbones import BACKBONES, FoundationEmbedder

    if architecture in BACKBONES:
        spec = BACKBONES[architecture]
        if track == "b" and not spec.commercial_use:
            raise ValueError(
                f"Backbone {architecture!r} is licensed {spec.licence}, which does not permit "
                "commercial use, so it cannot back a Track B model. See docs/licensing.md."
            )
        return FoundationEmbedder(architecture, cfg)

    if track == "b" and pretrained:
        raise ValueError(
            "ImageNet-pretrained weights cannot back a Track B (production) model. "
            "Train the stem from scratch, or run with --track a for research. "
            "See docs/licensing.md."
        )

    if architecture == "signet":
        return SigNet(cfg, spatial_transformer=spatial_transformer)
    if architecture == "hybrid":
        return HybridSigNet(cfg, pretrained=pretrained)
    raise ValueError(
        f"Unknown architecture {architecture!r}; expected 'signet', 'hybrid', "
        f"or one of {sorted(BACKBONES)}"
    )


@torch.no_grad()
def embed_batch(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Run the model in eval mode and return (optionally L2-normalised) vectors.

    Normalised embeddings are what the scoring layer expects: cosine distance
    on unit vectors is what all thresholds are calibrated against.
    """
    was_training = model.training
    model.eval()
    out = model(batch)
    if normalize:
        out = F.normalize(out, p=2, dim=1)
    if was_training:
        model.train()
    return out
