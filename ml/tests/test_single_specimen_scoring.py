"""The scoring contract, for a system that verifies against one specimen.

Production holds exactly one stored signature per customer and will continue
to — the customer base is too large for a collection programme. Every property
here follows from that, and each one is a defect that shipped.

The score used to be: a similarity, minus a baseline whose source varied, capped
by a floor, through a curve fitted for a different number of specimens. Each of
those stages failed in a way that produced a confident, wrong number rather than
an error. What replaced them is short enough that these tests can cover it
completely.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.scoring.calibrate import Band, ScoreCalibrator

DIM = 32


def _populations(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Similarity distributions roughly matching the measured model.

    Genuine near 0.93, skilled forgeries near 0.85, heavily overlapping — which
    is honest: at one specimen the measured EER is about 27%.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.93, 0.025, n), rng.normal(0.85, 0.045, n)


def _calibrator(seed: int = 0, n: int = 400) -> ScoreCalibrator:
    genuine, impostor = _populations(seed, n)
    calibrator = ScoreCalibrator.fit(genuine, impostor, protocol_references=1)
    calibrator.derive_band_edges(genuine, impostor)
    return calibrator


# --------------------------------------------------------------------------
# The score itself
# --------------------------------------------------------------------------


def test_score_is_monotone_in_similarity():
    """A closer match can never score lower.

    This replaces the whole family of guards that existed to stop a low
    similarity producing a high score — a floor, a minimum specimen agreement,
    an "inconsistent" flag on the panel. A curve fitted on similarity is
    monotone by construction, so the property is guaranteed rather than
    defended.
    """
    calibrator = _calibrator()
    scores = [calibrator.score_0_100(s) for s in np.linspace(0.5, 1.0, 400)]
    assert all(b >= a for a, b in zip(scores, scores[1:], strict=False))


def test_a_poor_match_scores_low_and_a_close_one_scores_high():
    calibrator = _calibrator()
    assert calibrator.score_0_100(0.60) < 10
    assert calibrator.score_0_100(0.98) > 70


def test_the_curve_emits_a_continuum_not_a_staircase():
    """The shipped curve emitted **eight** distinct values.

    Not a sample-size problem: isotonic keeps both ends of each flat block, so
    interpolating between them reproduces the steps literally. One knot per
    block, at the block's centroid, ramps between them instead.
    """
    assert _calibrator().distinct_scores >= 100


def test_the_ceiling_is_earned_rather_than_assumed():
    """Confidence scales with the evidence behind it.

    Isotonic assigns probability 1.0 to any top block containing only genuine
    comparisons, however few — which is how 99.5 printed from a model with AUC
    0.80. Laplace smoothing makes the ceiling a function of block size: a block
    of two genuine comparisons earns much less than a block of forty.
    """
    rng = np.random.default_rng(3)
    # Heavily overlapping in the body, with exactly two genuine comparisons
    # clear of everything else. Isotonic puts those two in the top block alone.
    genuine = np.concatenate([rng.uniform(0.70, 0.86, 60), [0.98, 0.99]])
    impostor = rng.uniform(0.70, 0.86, 62)

    calibrator = ScoreCalibrator.fit(genuine, impostor, protocol_references=1)

    # A top block of a few observations earns (k+1)/(k+2) — well under 0.9 for
    # any small k. Raw isotonic assigns 1.0 to the same block, which is how
    # 99.5 reached the screen from a model with AUC 0.80.
    ceiling = float(calibrator.y.max())
    assert ceiling < 0.9, f"a handful of comparisons earned {ceiling:.3f}"

    # More evidence earns more, and still never certainty.
    assert ceiling < float(_calibrator(n=2000).y.max()) < 0.995 + 1e-9


def test_a_score_is_always_available():
    """Never a dash at the counter.

    Withholding the number was the previous answer to "this similarity is not
    on the calibrator's scale". The operator has already captured the signature
    and waited; a non-answer is the least useful thing the screen can show.
    """
    calibrator = _calibrator()
    for similarity in (-0.5, 0.0, 0.42, 0.93, 1.0):
        score = calibrator.score_0_100(similarity)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0


# --------------------------------------------------------------------------
# Band edges
# --------------------------------------------------------------------------


def test_band_edges_come_from_the_calibrator_not_the_config():
    calibrator = _calibrator()
    assert calibrator.green_min > 0
    assert calibrator.effective_edges() == (calibrator.green_min, calibrator.red_max)


def test_the_green_edge_holds_the_false_accept_target():
    """Green means "at most this share of forgeries reach it", by construction.

    A fixed edge on the 0-100 score has no operational meaning: the score is a
    probability under the benchmark's genuine/impostor mix, not the base rate at
    a counter, so changing the number of forgeries per writer in validation
    moves every band. FAR is conditional on class and does not move.
    """
    genuine, impostor = _populations()
    calibrator = ScoreCalibrator.fit(genuine, impostor, protocol_references=1)
    calibrator.derive_band_edges(genuine, impostor, green_max_far=0.05)

    scores = np.array([calibrator.score_0_100(s) for s in impostor])
    assert float((scores >= calibrator.green_min).mean()) <= 0.05 + 1e-9


def test_the_red_edge_holds_the_false_reject_target():
    genuine, impostor = _populations()
    calibrator = ScoreCalibrator.fit(genuine, impostor, protocol_references=1)
    calibrator.derive_band_edges(genuine, impostor, red_max_frr=0.05)

    scores = np.array([calibrator.score_0_100(s) for s in genuine])
    assert float((scores <= calibrator.red_max).mean()) <= 0.05 + 1e-9


def test_a_tighter_far_target_raises_the_green_edge():
    genuine, impostor = _populations()

    def edge(far: float) -> float:
        c = ScoreCalibrator.fit(genuine, impostor, protocol_references=1)
        c.derive_band_edges(genuine, impostor, green_max_far=far)
        return c.green_min

    assert edge(0.01) >= edge(0.05) >= edge(0.10)


def test_bands_are_ordered():
    calibrator = _calibrator()
    assert calibrator.red_max < calibrator.green_min
    assert calibrator.band(0.99) is Band.GREEN
    assert calibrator.band(0.50) is Band.RED


# --------------------------------------------------------------------------
# Context for the operator
# --------------------------------------------------------------------------


def test_the_curve_reports_how_common_a_similarity_is():
    """A bare 46 is not actionable; 46 beside "1 in 4 genuine match this
    poorly" is."""
    calibrator = _calibrator()

    poor = calibrator.share_reaching(0.80, "genuine")
    good = calibrator.share_reaching(0.96, "genuine")

    assert poor is not None and good is not None
    # More genuine signatures clear a low bar than a high one.
    assert poor > good


def test_the_duplicate_threshold_tracks_the_model():
    """The photocopy check must survive an honest ceiling.

    It used to require a calibrated score of 99, which stopped being reachable
    the moment shrinkage lowered the top of the scale — a fraud control
    disabled by an unrelated improvement. It is now a similarity quantile, so
    it moves with the model rather than with the score's presentation.
    """
    calibrator = _calibrator()
    genuine, _ = _populations()

    assert calibrator.duplicate_similarity_min > float(np.quantile(genuine, 0.99))
    assert calibrator.duplicate_similarity_min <= 1.0


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


def test_an_extra_specimen_cannot_lower_a_genuine_score():
    """`max` pooling, so opportunistic enrolment is always safe.

    If a second specimen could lower a customer's score, adding one would be a
    risk rather than an improvement, and nobody would do it.
    """
    from ml.scoring.compare import compare_to_references

    rng = np.random.default_rng(5)
    base = rng.normal(size=DIM)
    base /= np.linalg.norm(base)
    query = base + 0.05 * rng.normal(size=DIM)

    one = compare_to_references(query, base.reshape(1, -1))
    two = compare_to_references(query, np.vstack([base, rng.normal(size=DIM)]))

    assert two.similarity >= one.similarity - 1e-12


def test_one_specimen_makes_similarity_the_only_statistic():
    from ml.scoring.compare import compare_to_references

    rng = np.random.default_rng(6)
    score = compare_to_references(rng.normal(size=DIM), rng.normal(size=(1, DIM)))

    assert score.is_single_reference
    assert score.similarity == pytest.approx(score.max_similarity)
    assert score.similarity == pytest.approx(score.mean_similarity)


def test_the_comparison_records_its_scoring_generation():
    """The audit trail is append-only and now holds two shapes of row."""
    from ml.scoring.compare import compare_to_references

    payload = compare_to_references(np.ones(DIM), np.ones((1, DIM))).to_dict()
    assert payload["scoring_version"] == 2
    assert "similarity" in payload
    # `raw` was a writer-normalised margin and must not be confused with one.
    assert "raw" not in payload
