"""Tests for pipeline tracing.

The property that matters most is the one that is easiest to break by accident:
**tracing must not change the result**. A trace call that mutates the array it
was handed, or a code path that only runs when tracing is on, turns the
explanation into a lie — and a convincing one, because the displayed stages
would still look right.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocess.pipeline import preprocess_signature
from ml.preprocess.trace import MAX_STAGE_EDGE, PipelineTrace, vector_strip


@pytest.fixture
def signature() -> np.ndarray:
    """A crop with enough ink to survive the blank check."""
    image = np.full((160, 380), 245, dtype=np.uint8)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4 * np.pi, 320)
    y = (80 + 34 * np.sin(x) + rng.normal(0, 2, x.size)).astype(int)
    for i, (px, py) in enumerate(zip(np.linspace(30, 350, x.size).astype(int), y)):
        image[max(0, py - 3) : py + 3, max(0, px - 3) : px + 3] = 20
        if i % 40 == 0:
            image[max(0, py - 12) : py + 12, max(0, px - 2) : px + 2] = 30
    return image


# --------------------------------------------------------------------------
# The observational guarantee
# --------------------------------------------------------------------------


def test_tracing_does_not_change_the_canvas(signature):
    without = preprocess_signature(signature)
    with_trace = preprocess_signature(signature, trace=PipelineTrace())

    assert np.array_equal(without.image, with_trace.image)
    assert without.ink_fraction == with_trace.ink_fraction
    assert without.source_bbox == with_trace.source_bbox


def test_disabled_trace_records_nothing(signature):
    trace = PipelineTrace(enabled=False)
    preprocess_signature(signature, trace=trace)
    assert len(trace) == 0
    assert not trace


def test_trace_does_not_mutate_the_input(signature):
    original = signature.copy()
    preprocess_signature(signature, trace=PipelineTrace())
    assert np.array_equal(signature, original)


# --------------------------------------------------------------------------
# What gets recorded
# --------------------------------------------------------------------------


def test_preprocessing_records_every_step(signature):
    trace = PipelineTrace()
    preprocess_signature(signature, trace=trace)

    keys = [stage.key for stage in trace.stages]
    assert keys == [
        "grayscale",
        "illumination",
        "binarised",
        "lines_removed",
        "denoised",
        "normalised",
    ]
    assert all(stage.image is not None for stage in trace.stages)


def test_binary_stages_are_flagged_for_inverted_display(signature):
    """Ink-high-on-black masks read as a photographic negative to a person."""
    trace = PipelineTrace()
    preprocess_signature(signature, trace=trace)

    by_key = {stage.key: stage for stage in trace.stages}
    assert by_key["binarised"].invert_for_display
    assert by_key["normalised"].invert_for_display
    # The grayscale and illumination stages are already ink-dark-on-light.
    assert not by_key["grayscale"].invert_for_display
    assert not by_key["illumination"].invert_for_display


def test_skipping_line_removal_skips_its_stage(signature):
    trace = PipelineTrace()
    preprocess_signature(signature, strip_form_lines=False, trace=trace)
    assert "lines_removed" not in [s.key for s in trace.stages]


def test_stage_images_are_shrunk_for_transport():
    """A full page is megabytes; a dozen of them per request is not acceptable."""
    trace = PipelineTrace()
    trace.add("big", "Big", "", image=np.zeros((3000, 2000), dtype=np.uint8))
    assert max(trace.stages[0].image.shape[:2]) == MAX_STAGE_EDGE


def test_small_images_are_not_upscaled():
    trace = PipelineTrace()
    trace.add("small", "Small", "", image=np.zeros((40, 60), dtype=np.uint8))
    assert trace.stages[0].image.shape[:2] == (40, 60)


def test_none_metrics_are_dropped():
    """An absent measurement should not render as an empty chip."""
    trace = PipelineTrace()
    trace.add("s", "S", "", present=1.0, absent=None)
    assert trace.stages[0].metrics == {"present": 1.0}


# --------------------------------------------------------------------------
# Embedding strip
# --------------------------------------------------------------------------


def test_vector_strip_has_the_requested_shape():
    strip = vector_strip(np.random.default_rng(0).normal(size=256), height=64, width=400)
    assert strip.shape == (64, 400, 3)


def test_vector_strip_separates_sign():
    """Positive and negative dimensions must not render identically."""
    positive = vector_strip(np.ones(8))
    negative = vector_strip(-np.ones(8))
    assert not np.array_equal(positive, negative)


def test_vector_strip_survives_an_all_zero_vector():
    """A degenerate embedding must render as black, not divide by zero."""
    strip = vector_strip(np.zeros(16))
    assert strip.shape[2] == 3
    assert int(strip.max()) == 0


def test_vector_strip_handles_an_empty_vector():
    assert vector_strip(np.array([])).shape[2] == 3
