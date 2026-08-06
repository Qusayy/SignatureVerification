"""Tests for the preprocessing pipeline.

These use the synthetic generator rather than fixture images so the suite stays
self-contained and no biometric data is ever committed to the repository.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.config import PREPROCESS
from ml.data.synth import make_signer, render_on_form, render_signature
from ml.preprocess.pipeline import (
    BlankSignatureError,
    center_on_canvas,
    deskew_page,
    estimate_page_skew,
    ink_bounding_box,
    normalize_illumination,
    preprocess_signature,
    remove_form_lines,
    sauvola_binarize,
    to_model_input,
)
from ml.tests.conftest import canvas_geometry, tolerant_overlap


@pytest.fixture
def signature() -> np.ndarray:
    rng = np.random.default_rng(7)
    style = make_signer("T0001", "latin", rng)
    return render_signature(style, np.random.default_rng(8), kind="genuine")


def test_blank_input_is_refused():
    blank = np.full((200, 400), 255, dtype=np.uint8)
    with pytest.raises(BlankSignatureError):
        preprocess_signature(blank)


def test_blank_input_non_strict_reports_warning():
    blank = np.full((200, 400), 255, dtype=np.uint8)
    result = preprocess_signature(blank, strict=False)
    assert "blank_or_near_blank" in result.warnings
    assert not result.is_usable


def test_output_geometry_and_ink(signature):
    result = preprocess_signature(signature)
    assert result.image.shape == PREPROCESS.canvas_size
    assert result.image.dtype == np.uint8
    assert result.is_usable
    # Ink is white (255) on black (0) after preprocessing.
    assert set(np.unique(result.image)).issubset({0, 255})
    assert 0 < result.ink_fraction < 0.5


def test_translation_invariance(signature):
    """The same signature at a different position must normalise identically."""
    padded = np.full((signature.shape[0] + 300, signature.shape[1] + 400), 255, dtype=np.uint8)
    padded[120 : 120 + signature.shape[0], 250 : 250 + signature.shape[1]] = signature

    a = preprocess_signature(signature).image
    b = preprocess_signature(padded).image

    assert tolerant_overlap(a, b) > 0.95, "centring is not translation invariant"


def test_absolute_size_is_normalised_away(signature):
    """A small signature must normalise to the same geometry as a large one.

    This reverses an earlier policy that deliberately preserved absolute size
    on the theory that size distinguishes writers. In practice it made the
    model reject genuine signatures written smaller than the stored specimen
    while accepting forgeries that matched the specimen's size — see
    ml/eval/diagnostics.py. Size is now removed; aspect ratio is kept, and is
    covered by ml/tests/test_scale_invariance.py.
    """
    small = cv2.resize(signature, (signature.shape[1] // 3, signature.shape[0] // 3))

    big_rms, big_w, big_h = canvas_geometry(preprocess_signature(signature).image)
    small_rms, small_w, small_h = canvas_geometry(preprocess_signature(small).image)

    assert abs(small_rms - big_rms) / big_rms < 0.08
    assert abs(small_w - big_w) / big_w < 0.08
    assert abs(small_h - big_h) / big_h < 0.08


def test_size_normalisation_can_be_disabled(signature):
    """The old behaviour remains reachable for ablation studies."""
    from dataclasses import replace

    cfg = replace(PREPROCESS, normalize_size=False)
    small = cv2.resize(signature, (signature.shape[1] // 3, signature.shape[0] // 3))

    normalised = canvas_geometry(preprocess_signature(small).image)[0]
    raw = canvas_geometry(preprocess_signature(small, cfg).image)[0]
    assert raw < normalised, "disabling normalisation should leave the small signature small"


def test_a_nearly_empty_crop_is_not_blown_up(signature):
    """The upscale cap stops a speck being magnified into apparent signal."""
    from dataclasses import replace

    cfg = replace(PREPROCESS, max_upscale=2.0)
    tiny = cv2.resize(signature, (signature.shape[1] // 8, signature.shape[0] // 8))
    result = preprocess_signature(tiny, cfg, strict=False)
    # With the cap in force the ink cannot reach the fully normalised spread.
    assert canvas_geometry(result.image)[0] <= PREPROCESS.target_ink_rms_ratio * PREPROCESS.canvas_size[0]


def test_illumination_normalisation_survives_a_gradient(signature):
    h, w = signature.shape
    gradient = np.linspace(0.55, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    shaded = np.clip(signature.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    clean = preprocess_signature(signature)
    lit = preprocess_signature(shaded)

    # A shadow across half the page must not be read as ink.
    assert lit.ink_fraction < clean.ink_fraction * 2.0
    flat = normalize_illumination(shaded)
    assert flat[:, :20].mean() > shaded[:, :20].mean()


def test_form_lines_are_removed_but_strokes_survive():
    canvas = np.zeros((300, 800), dtype=np.uint8)
    # A long printed rule spanning most of the width.
    cv2.line(canvas, (10, 200), (790, 200), 255, 2)
    line_pixels = np.count_nonzero(canvas)
    # A signature-like squiggle crossing it.
    pts = np.array([[100, 150], [200, 90], [300, 210], [400, 120], [500, 190]], dtype=np.int32)
    cv2.polylines(canvas, [pts], False, 255, 4)
    total = np.count_nonzero(canvas)
    stroke_pixels = total - line_pixels

    cleaned = remove_form_lines(canvas)

    remaining_on_rule = np.count_nonzero(cleaned[198:203, :])
    assert remaining_on_rule < line_pixels * 0.5, "printed rule was not removed"
    # The squiggle must largely survive.
    assert np.count_nonzero(cleaned) > stroke_pixels * 0.6


def test_signature_over_a_form_line_stays_connected():
    """Removing a rule must not sever a stroke that crosses it."""
    canvas = np.zeros((300, 800), dtype=np.uint8)
    cv2.line(canvas, (10, 150), (790, 150), 255, 2)
    cv2.line(canvas, (400, 60), (400, 250), 255, 5)  # vertical stroke crossing it

    cleaned = remove_form_lines(canvas)
    column = cleaned[60:250, 396:406]
    count, _ = cv2.connectedComponents((column > 0).astype(np.uint8))
    assert count - 1 <= 1, "stroke was cut into pieces by line removal"


def test_page_skew_is_estimated_and_corrected():
    rng = np.random.default_rng(11)
    style = make_signer("T0002", "latin", rng)
    sig = render_signature(style, rng, kind="genuine")
    page, _ = render_on_form(sig, rng)

    angle = 4.0
    h, w = page.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(page, matrix, (w, h), borderValue=250)

    # estimate_page_skew returns the correction to apply, which for a page
    # rotated counter-clockwise by +angle is about -angle.
    estimated = estimate_page_skew(rotated)
    assert abs(estimated + angle) < 1.5, f"skew correction {estimated} does not undo {angle}"

    corrected, applied = deskew_page(rotated)
    assert abs(applied - estimated) < 1e-6
    # After correction there is no meaningful skew left to detect.
    assert abs(estimate_page_skew(corrected)) < 1.0


def test_skew_estimate_is_zero_without_straight_structure(signature):
    """A bare signature has no reliable skew cue — must return 0, not a guess."""
    assert estimate_page_skew(signature) == 0.0


def test_center_on_canvas_centres_by_mass():
    binary = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(binary, (10, 10), (60, 60), 255, -1)
    canvas, bbox = center_on_canvas(binary)

    moments = cv2.moments(canvas, binaryImage=True)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    assert abs(cx - PREPROCESS.canvas_size[1] / 2) < 3
    assert abs(cy - PREPROCESS.canvas_size[0] / 2) < 3
    assert bbox[2] > 0 and bbox[3] > 0


def test_ink_bounding_box_on_empty_image():
    assert ink_bounding_box(np.zeros((50, 50), dtype=np.uint8)) == (0, 0, 0, 0)


def test_sauvola_falls_back_on_uniform_input():
    """A near-uniform crop must not come back as a field of noise."""
    uniform = np.full((200, 200), 200, dtype=np.uint8)
    uniform[90:110, 50:150] = 40  # one real dark mark
    binary = sauvola_binarize(uniform)
    assert np.count_nonzero(binary) < binary.size * 0.5


def test_to_model_input_shape_and_range(signature):
    result = preprocess_signature(signature)

    infer = to_model_input(result)
    assert infer.shape == (1, *PREPROCESS.crop_size)
    assert infer.dtype == np.float32
    assert 0.0 <= infer.min() and infer.max() <= 1.0

    train = to_model_input(result, train=True, rng=np.random.default_rng(3))
    assert train.shape == infer.shape


def test_to_model_input_centre_crop_is_deterministic(signature):
    result = preprocess_signature(signature)
    a = to_model_input(result)
    b = to_model_input(result)
    assert np.array_equal(a, b)


def test_arabic_signature_processes(signature):
    rng = np.random.default_rng(21)
    style = make_signer("T0003", "arabic", rng)
    img = render_signature(style, rng, kind="genuine")
    result = preprocess_signature(img)
    assert result.is_usable
    assert result.image.shape == PREPROCESS.canvas_size
