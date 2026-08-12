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
    employee always decides.

    **The deployment protocol is one stored specimen per customer.** That is not
    a degraded case to be worked around — it is the design point, because the
    customer base is too large for a specimen-collection programme. Everything
    here follows from it.

    An earlier design expressed each score relative to the agreement among a
    customer's *own* specimens, which was worth 15.7 EER points at two or more
    specimens and exactly nothing at one, where the "baseline" degenerates to a
    corpus constant. It has been removed from the serving path.
    `ml.eval.benchmark` still reports it as an alternative recipe, so a future
    multi-specimen pilot needs one benchmark run rather than a
    re-implementation.
    """

    # Specimens per customer, in both the calibration protocol and production.
    # The calibrator records the protocol it was fitted for and the service
    # refuses a mismatch — a curve fitted on six specimens and applied to one is
    # a value from one measurement pushed through a scale built for another,
    # which is what every score before this rework was.
    calibration_references: int = 1

    # How several specimens are pooled, on the rare occasion there are several.
    # `max` keeps the score's meaning invariant to specimen count: at one
    # specimen it is exactly the calibrated condition, and an extra specimen can
    # only raise a genuine customer's score. A blended max/mean statistic
    # changes shape with the count in a way that is neither bounded nor
    # reportable.
    pooling: str = "max"

    # --- Band edges ------------------------------------------------------
    #
    # Derived from operating points on validation and stored in the calibrator,
    # not fixed here. The 0-100 score is P(genuine) under the prior the
    # benchmark constructs (~45% genuine); at a counter the forgery base rate is
    # nearer 0.1%. So the number is a monotone rank statistic wearing a
    # probability's clothes, and a fixed edge at 75 has no operational meaning —
    # change the forgeries-per-writer in the validation split and the same
    # signature lands in a different band.
    #
    # FAR and FRR are conditional on class and therefore invariant to that mix.
    # Green admits at most `green_max_far` of forgeries; red rejects at most
    # `red_max_frr` of genuine signatures.
    #
    # At one specimen on the sealed test set, green at FAR 5% admits roughly 39%
    # of genuine traffic: about 40% green, 55% amber, 5% red. Amber means the
    # teller compares manually, which is what they do for everything today.
    green_max_far: float = 0.05
    red_max_frr: float = 0.05

    # Used only when a calibrator carries no derived edges. Under the refusal
    # policy in ml/scoring/verifier.py such a calibrator cannot reach
    # production, so these serve tests and the pre-benchmark state.
    green_min_fallback: float = 75.0
    red_max_fallback: float = 40.0

    # --- Photocopy detection ---------------------------------------------
    #
    # A genuine signature is never reproduced exactly. Near-pixel-identical
    # geometry indicates a copy of the stored specimen pasted onto the form,
    # which is a fraud signal rather than the strongest possible match.
    #
    # The similarity half of the test lives in the calibrator
    # (`duplicate_similarity_min`, the 99.9th percentile of genuine) rather than
    # here. It used to be a threshold on the calibrated *score* (99.0), which
    # stopped firing the moment honest calibration lowered the ceiling below it
    # — a fraud control disabled by an unrelated improvement.
    duplicate_iou_min: float = 0.985

    # FAR operating points reported by the evaluation harness.
    far_targets: tuple[float, ...] = (0.10, 0.05, 0.01, 0.001)

    # Size of the impostor cohort, for the benchmark's alternative-recipe
    # comparison only. Cohort normalisation is not on the serving path: it cost
    # 15.7 EER points here, because it answers the random-impostor question,
    # which is already solved at 0.00% EER.
    cohort_size: int = 200


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
