"""Blocking gate: the model must recognise a signature at any plausible size.

A customer does not write at a fixed scale. If the same signature written 20%
smaller than the stored specimen scores like a forgery, the system rejects
genuine customers — and no aggregate metric reveals it, which is exactly how
this defect shipped in the first version.

These tests operate on the *preprocessing* contract as well as the trained
model, so the geometry guarantee holds even before a model exists.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.config import ARTIFACT_ROOT
from ml.data.synth import make_signer, render_signature
from ml.eval.diagnostics import DEFAULT_SCALES, load_embedder, scale_sweep
from ml.preprocess.pipeline import ink_rms_radius, preprocess_signature

CHECKPOINT = ARTIFACT_ROOT / "signet_track_b.pt"


@pytest.fixture
def signature() -> np.ndarray:
    rng = np.random.default_rng(31)
    return render_signature(make_signer("SCALE", "latin", rng), rng, kind="genuine")


def _rescale(image: np.ndarray, factor: float) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (max(8, int(w * factor)), max(8, int(h * factor))),
        interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR,
    )


# --------------------------------------------------------------------------
# Preprocessing contract — holds without any trained model
# --------------------------------------------------------------------------


def _geometry(canvas: np.ndarray) -> tuple[float, int, int]:
    ys, xs = np.nonzero(canvas)
    return (
        ink_rms_radius(canvas),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def _tolerant_overlap(a: np.ndarray, b: np.ndarray, tolerance: int = 9) -> float:
    """Fraction of ink in each image lying within ``tolerance`` px of the other.

    Plain binary IoU is the wrong instrument here: signature strokes are ~3px
    wide, so a 2px lateral shift drives IoU toward zero even when the shapes
    are identical. This measures whether the strokes land in the same *place*,
    which is the property that matters.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance))
    a_ink, b_ink = a > 0, b > 0
    a_near = cv2.dilate(a_ink.astype(np.uint8), kernel) > 0
    b_near = cv2.dilate(b_ink.astype(np.uint8), kernel) > 0
    total = np.count_nonzero(a_ink) + np.count_nonzero(b_ink)
    if total == 0:
        return 0.0
    return (np.count_nonzero(a_ink & b_near) + np.count_nonzero(b_ink & a_near)) / total


@pytest.mark.parametrize("factor", DEFAULT_SCALES)
def test_preprocessing_normalises_size_away(signature, factor):
    """The same signature at any plausible size must normalise to the same geometry.

    This is the property the embedding model depends on. If preprocessing does
    not deliver it, no amount of training will.
    """
    base = preprocess_signature(signature).image
    scaled = preprocess_signature(_rescale(signature, factor), strict=False).image

    base_rms, base_w, base_h = _geometry(base)
    rms, width, height = _geometry(scaled)

    assert abs(rms - base_rms) / base_rms < 0.06, (
        f"at {factor}x the ink spread is {rms:.1f} vs {base_rms:.1f}; "
        "absolute size is not being normalised away"
    )
    assert abs(width - base_w) / base_w < 0.06
    assert abs(height - base_h) / base_h < 0.06
    assert _tolerant_overlap(base, scaled) > 0.95


def test_preprocessing_preserves_aspect_ratio(signature):
    """Size is normalised away; the width-to-height ratio is not.

    Aspect is genuinely discriminative between writers. Absolute size is mostly
    pen and mood.
    """

    def aspect(image: np.ndarray) -> float:
        canvas = preprocess_signature(image, strict=False).image
        ys, xs = np.nonzero(canvas)
        return (xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1)

    reference = aspect(signature)
    for factor in (0.6, 1.6):
        assert abs(aspect(_rescale(signature, factor)) - reference) / reference < 0.15


def test_a_genuinely_wider_signature_still_looks_different(signature):
    """Normalisation must not erase real shape differences.

    Stretching only the width changes the signature's proportions — a real
    discriminative difference that must survive normalisation, unlike uniform
    scaling which must not.
    """
    h, w = signature.shape[:2]
    stretched = cv2.resize(signature, (int(w * 1.6), h))

    base = preprocess_signature(signature).image
    wide = preprocess_signature(stretched, strict=False).image
    _, base_w, base_h = _geometry(base)
    _, wide_w, wide_h = _geometry(wide)

    assert (wide_w / wide_h) > (base_w / base_h) * 1.2, (
        "a genuinely wider signature must stay wider after normalisation"
    )
    assert _tolerant_overlap(base, wide) < 0.95


# --------------------------------------------------------------------------
# End-to-end gate against the trained model
# --------------------------------------------------------------------------


@pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"No checkpoint at {CHECKPOINT}")
def test_trained_model_is_scale_invariant(signature):
    """The blocking gate from the remediation plan.

    Fails against the original model, which scored the same signature at 0.6x
    as 0.39 — far below the 0.74 a skilled forgery achieved.
    """
    sweep = scale_sweep(load_embedder(CHECKPOINT), signature)
    assert sweep.passes, (
        f"worst rescaled similarity {sweep.worst:.3f} < 0.95. "
        f"Per-scale: { {f'{k:g}x': round(v, 3) for k, v in sweep.similarities.items()} }"
    )
