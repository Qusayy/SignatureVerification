"""Central ML configuration.

Single source of truth for image geometry, model defaults, and decision bands.
Both the training code and the API import from here so a preprocessing change
cannot silently desynchronise training from inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("SV_DATA_ROOT", REPO_ROOT / "data"))
ARTIFACT_ROOT = Path(os.environ.get("SV_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
DOCS_ROOT = REPO_ROOT / "docs"


# --------------------------------------------------------------------------
# Image geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Geometry and thresholds for the preprocessing pipeline.

    The canvas is deliberately larger than the network input: signatures are
    centred on a fixed canvas first (preserving aspect ratio and relative
    stroke scale), then resized. Doing it the other way round distorts wide
    signatures and is a common source of silent accuracy loss.
    """

    # Canvas the ink is centred on before resizing (H, W). Ratio ~1:1.43,
    # matching the wide aspect of a typical signature.
    canvas_size: tuple[int, int] = (952, 1360)

    # Network input after resize (H, W). 170x242 follows the SigNet
    # convention; the ViT hybrid re-resizes to 224x224 internally.
    input_size: tuple[int, int] = (170, 242)

    # Random crop applied during training; centre crop at inference.
    crop_size: tuple[int, int] = (150, 220)

    # Sauvola local-threshold window (odd) and sensitivity.
    sauvola_window: int = 51
    sauvola_k: float = 0.15

    # Ruled-line / form-line removal: minimum run length as a fraction of the
    # image dimension for a horizontal or vertical structure to count as a
    # printed line rather than part of a signature stroke.
    line_removal_ratio: float = 0.45

    # Ink is padded by this fraction of the tight bounding box before centring,
    # so strokes are never flush against the canvas edge.
    ink_pad_ratio: float = 0.04

    # --- Size normalisation -------------------------------------------
    # Signatures are rescaled so the RMS spread of ink about its centroid
    # equals this fraction of the canvas height. Absolute size therefore stops
    # being a feature, while the width-to-height ratio is preserved.
    #
    # This replaces an earlier "centre but never upscale" policy that kept
    # absolute size deliberately. That policy made the model reject genuine
    # signatures written 20% smaller than the stored specimen — see
    # ml/eval/diagnostics.py and the plan's Test A.
    #
    # RMS about the centroid is used rather than the bounding box because a
    # single long flourish moves the bounding box a great deal and the RMS
    # hardly at all.
    normalize_size: bool = True
    target_ink_rms_ratio: float = 0.155

    # Guard against a nearly-empty crop being blown up into pure noise.
    max_upscale: float = 6.0

    # Below this fraction of ink pixels the crop is considered blank and
    # verification is refused rather than scored on noise.
    min_ink_fraction: float = 0.0015


PREPROCESS = PreprocessConfig()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "signet"  # "signet" | "hybrid"
    embedding_dim: int = 512
    dropout: float = 0.3
    # ArcFace hyperparameters for the writer-identity head.
    arcface_scale: float = 30.0
    arcface_margin: float = 0.30
    # Weight of the forgery-contrastive term relative to writer identity.
    forgery_loss_weight: float = 0.95
    triplet_margin: float = 0.35
    # Weight on the global-threshold pair term. See
    # :class:`ml.embed.losses.GlobalThresholdPairLoss`: the triplet objective
    # only orders distances *within* a writer, so two writers can each be
    # separable at different absolute cosines. With several specimens on file
    # that is recoverable at scoring time from the customer's own specimen
    # agreement; with a single specimen it is not, and every writer is judged
    # on one shared scale. This term asks the model to provide that scale.
    #
    # Added on top of the identity/forgery combination rather than folded into
    # it, so an existing recipe keeps its balance. Set to 0.0 to disable.
    pair_loss_weight: float = 0.3


MODEL = ModelConfig()


# --------------------------------------------------------------------------
# Scoring and decision bands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringConfig:
    """Scoring behaviour.

    Bands are ADVISORY. The system never accepts or rejects on its own; the
    employee always decides. Thresholds are placeholders until calibrated on
    the organisation's sealed test set by ml/eval/benchmark.py.
    """

    # How a similarity becomes a score. Measured on the sealed 200-signer test
    # split of manifest_large with signet_v3_geom.pt — one checkpoint, one set
    # of comparisons, only the recipe varying:
    #
    #   recipe                            EER     95% CI       AUC     TAR@FAR1%
    #   cohort S-norm  (previous default) 35.80%  34.0-37.8    0.703   4.10%
    #   raw similarity                    23.90%  21.9-25.6    0.849   9.50%
    #   writer-internal  (current)        20.10%  18.4-21.4    0.878   4.90%
    #   writer-internal + cohort          30.40%  28.6-32.3    0.769   6.30%
    #
    # Reproduce with `python -m ml.eval.benchmark`, which prints the served
    # recipe against its alternative on every run.

    # Express each score relative to how consistently the customer signs, by
    # subtracting the mean pairwise similarity among their own specimens.
    # Worth 15.7 EER points over the previous default, with non-overlapping
    # intervals.
    #
    # The caveat, which the EER hides: at FAR = 1% the plain raw similarity is
    # better (9.50% TAR vs 4.90%). Writer normalisation improves the middle of
    # the curve more than the strict tail. EER and AUC are the right target
    # *for this system* because it is advisory and bands uncertain cases amber
    # rather than auto-accepting — but a deployment that wanted a
    # high-precision auto-accept threshold should re-take this decision, and
    # neither recipe supports one today.
    writer_normalise: bool = True

    # --- Guards on the relative score -----------------------------------
    #
    # Writer normalisation is a *relative* judgement, and on its own it has no
    # floor: a customer whose stored specimens do not resemble each other has a
    # baseline near zero, so any query at all clears it. A signature matching
    # the specimen 6.8% scored 88/100 that way. Both guards below can only ever
    # lower a score, never raise one, so neither can introduce a false accept.

    # No relative margin rescues an absolute similarity this low. A genuine
    # signature sits at 0.90-0.99 and even a good forgery at 0.79-0.95, so this
    # never binds on real comparisons — it exists to stop nonsense, not to
    # tune the operating point.
    absolute_similarity_floor: float = 0.60

    # Specimen agreement below this means the stored specimens do not look like
    # each other. That is a broken enrolment — wrong customer, mis-cropped
    # scan, two different people — and it must not be read as "this customer is
    # easy to match". The population baseline is used instead, and the response
    # says so.
    min_specimen_agreement: float = 0.50

    # Cohort z/t/s-normalisation against a background population.
    #
    # OFF by default, which reverses the original design. It was adopted for
    # the standard speaker-verification reason and measured afterwards: it
    # costs 15.7 EER points on its own, and 10.3 even on top of writer
    # normalisation.
    #
    # The reason it hurts here is that it answers "does this signature look
    # more like this customer than like the population?", which is the
    # *random-impostor* question. That one is already solved — random-impostor
    # EER is 0.00% on this corpus. The question that matters is whether a
    # deliberate imitation of this customer is genuine, and against that a
    # background population is noise.
    #
    # Left switchable rather than deleted: it may earn its place on a real
    # corpus with genuinely diverse writers. `ml.eval.benchmark` reports both.
    cohort_normalise: bool = False

    # Size of the impostor cohort used for z-normalisation.
    cohort_size: int = 200

    # Advisory band edges on the calibrated 0-100 score.
    green_min: float = 75.0
    red_max: float = 40.0

    # A calibrated score at or above this, combined with near-pixel-identical
    # geometry, indicates a photocopy of the stored reference rather than a
    # fresh signature. Flagged as suspected fraud, not as a strong match.
    duplicate_score_min: float = 99.0
    duplicate_iou_min: float = 0.985

    # FAR operating points reported by the evaluation harness.
    far_targets: tuple[float, ...] = (0.10, 0.05, 0.01, 0.001)


SCORING = ScoringConfig()


# --------------------------------------------------------------------------
# Data provenance / licensing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LicenceTrack:
    """Licence tracks, enforced at training time.

    Track A assets (GPDS-derived weights, CEDAR, AGPL detectors) may be used
    for the POC and for benchmarking, but MUST NOT contribute to any weight
    shipped to production. See docs/licensing.md.
    """

    name: str
    commercial_use: bool
    description: str


TRACK_A = LicenceTrack(
    name="track_a_research",
    commercial_use=False,
    description="Research/non-commercial datasets and weights. POC and benchmarking only.",
)

TRACK_B = LicenceTrack(
    name="track_b_production",
    commercial_use=True,
    description="Locally owned data plus permissively-licensed or locally generated synthetic data.",
)

# Datasets known to be non-commercial. Ingest tags them Track A automatically.
NON_COMMERCIAL_SOURCES: frozenset[str] = frozenset(
    {"gpds", "gpds_synthetic", "cedar", "utsig", "bhsig260", "sigcomp", "mcyt", "sid"}
)


@dataclass(frozen=True)
class ScriptLabels:
    """Signature scripts tracked separately so per-script bias stays visible."""

    values: tuple[str, ...] = ("latin", "arabic", "mixed", "unknown")
    default: str = "unknown"


SCRIPTS = ScriptLabels()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    seed: int = 1337
    num_workers: int = 0  # Windows-safe default; raise on Linux training boxes
    val_every: int = 1
    device: str = field(default_factory=lambda: os.environ.get("SV_DEVICE", "auto"))


TRAIN = TrainConfig()


def resolve_device(preference: str | None = None) -> str:
    """Resolve the configured device preference to a concrete torch device."""
    pref = preference or TRAIN.device
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
