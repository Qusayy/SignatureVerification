"""Tests for comparison, cohort normalisation, calibration, and the verifier."""

from __future__ import annotations

import numpy as np
import pytest

from ml.config import SCORING
from ml.scoring.calibrate import Band, ScoreCalibrator, band_for
from ml.scoring.compare import compare_to_references, l2_normalize
from ml.scoring.explain import difference_overlay, reason_text, side_by_side
from ml.scoring.verifier import VerificationResult, Verifier, _canvas_iou
from ml.scoring.znorm import CohortNormalizer

DIM = 32


def _unit(vector) -> np.ndarray:
    return l2_normalize(np.asarray(vector, dtype=float).reshape(1, -1))[0]


def _cohort(n: int = 120, seed: int = 0) -> CohortNormalizer:
    rng = np.random.default_rng(seed)
    return CohortNormalizer(rng.normal(size=(n, DIM)))


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def test_identical_signature_scores_one():
    query = _unit(np.arange(DIM))
    score = compare_to_references(query, query.reshape(1, -1))
    assert score.raw == pytest.approx(1.0)
    assert score.is_single_reference


def test_single_reference_collapses_max_and_mean():
    rng = np.random.default_rng(1)
    score = compare_to_references(rng.normal(size=DIM), rng.normal(size=(1, DIM)))
    assert score.max_similarity == pytest.approx(score.mean_similarity)
    assert score.raw == pytest.approx(score.max_similarity)


def test_multi_reference_uses_both_nearest_and_mean():
    """One close specimen must lift the score, but not all the way to its own."""
    query = _unit([1.0] + [0.0] * (DIM - 1))
    close = query
    far = _unit([0.0, 1.0] + [0.0] * (DIM - 2))
    score = compare_to_references(query, np.vstack([close, far]))

    assert score.max_similarity == pytest.approx(1.0)
    assert score.mean_similarity == pytest.approx(0.5, abs=1e-6)
    assert score.mean_similarity < score.raw < score.max_similarity


def test_comparison_rejects_an_empty_reference_set():
    with pytest.raises(ValueError, match="no stored reference"):
        compare_to_references(np.ones(DIM), np.empty((0, DIM)))


def test_comparison_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        compare_to_references(np.ones(DIM), np.ones((2, DIM + 1)))


# --------------------------------------------------------------------------
# Cohort normalisation
# --------------------------------------------------------------------------


def test_cohort_rejects_a_too_small_background_set():
    with pytest.raises(ValueError, match="too small"):
        CohortNormalizer(np.random.default_rng(0).normal(size=(5, DIM)))


def test_snorm_makes_scores_comparable_across_customers():
    """The point of S-norm: a distinctive and a generic signer become comparable.

    The generic signer's references sit close to the cohort, so a raw score of
    0.6 is unremarkable for them and notable for the distinctive signer. After
    normalisation the distinctive signer must come out ahead.
    """
    rng = np.random.default_rng(3)
    cohort_vectors = rng.normal(size=(200, DIM)) * 0.1
    cohort_vectors[:, 0] += 1.0  # cohort clusters along axis 0
    cohort = CohortNormalizer(cohort_vectors)

    generic_ref = l2_normalize(cohort_vectors.mean(axis=0).reshape(1, -1))
    distinctive_ref = l2_normalize(np.array([[0.0] * (DIM - 1) + [1.0]]))

    raw = 0.6
    generic_z = cohort.snorm(raw, generic_ref[0], references=generic_ref)
    distinctive_z = cohort.snorm(raw, distinctive_ref[0], references=distinctive_ref)

    assert distinctive_z > generic_z


def test_enrolment_stats_are_reusable_across_verifications():
    cohort = _cohort()
    rng = np.random.default_rng(4)
    refs = rng.normal(size=(3, DIM))
    query = rng.normal(size=DIM)

    stats = cohort.enrolment_stats(refs)
    a = cohort.snorm(0.5, query, enrolment=stats)
    b = cohort.snorm(0.5, query, references=refs)
    assert a == pytest.approx(b)


def test_snorm_requires_references_or_precomputed_stats():
    with pytest.raises(ValueError, match="precomputed enrolment stats"):
        _cohort().snorm(0.5, np.ones(DIM))


def test_cohort_round_trips_through_disk(tmp_path):
    cohort = _cohort()
    reloaded = CohortNormalizer.load(cohort.save(tmp_path / "cohort.npz"))
    assert reloaded.size == cohort.size
    assert np.allclose(reloaded.embeddings, cohort.embeddings, atol=1e-6)


def test_cohort_from_signers_takes_one_vector_per_writer():
    rng = np.random.default_rng(5)
    by_signer = {f"S{i}": rng.normal(size=(4, DIM)) for i in range(60)}
    cohort = CohortNormalizer.from_embeddings_by_signer(by_signer, size=40)
    assert cohort.size == 40
    assert len(set(cohort.signers)) == 40


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _fitted_calibrator(seed: int = 6) -> ScoreCalibrator:
    rng = np.random.default_rng(seed)
    return ScoreCalibrator.fit(
        rng.normal(2.5, 0.8, 500), rng.normal(-0.5, 0.8, 500), fitted_on="val"
    )


def test_calibrator_is_monotonic():
    calibrator = _fitted_calibrator()
    probabilities = [calibrator.probability(s) for s in np.linspace(-4, 6, 60)]
    # Deliberately offset lists, so the length mismatch is the point.
    assert all(b >= a - 1e-9 for a, b in zip(probabilities, probabilities[1:], strict=False))


def test_calibrator_separates_the_two_populations():
    calibrator = _fitted_calibrator()
    assert calibrator.score_0_100(3.5) > 75
    assert calibrator.score_0_100(-1.5) < 25


def test_calibrator_never_reports_absolute_certainty():
    """A verification system claiming 100% confidence is lying."""
    calibrator = _fitted_calibrator()
    assert 0.0 < calibrator.probability(-99.0)
    assert calibrator.probability(99.0) < 1.0


def test_calibrator_refuses_to_fit_on_too_little_data():
    with pytest.raises(ValueError, match="Too few scores"):
        ScoreCalibrator.fit([1.0, 2.0], [0.0, 0.5])


def test_calibrator_round_trips_without_sklearn(tmp_path):
    calibrator = _fitted_calibrator()
    reloaded = ScoreCalibrator.load(calibrator.save(tmp_path / "cal.json"))
    for s in (-2.0, 0.0, 1.5, 3.0):
        assert reloaded.probability(s) == pytest.approx(calibrator.probability(s))


def test_placeholder_calibrator_is_flagged():
    assert ScoreCalibrator.identity().is_placeholder
    assert not _fitted_calibrator().is_placeholder


def test_bands_follow_the_configured_edges():
    assert band_for(SCORING.green_min + 1) is Band.GREEN
    assert band_for(SCORING.red_max - 1) is Band.RED
    assert band_for((SCORING.green_min + SCORING.red_max) / 2) is Band.AMBER


def test_threshold_inversion_matches_the_forward_map():
    calibrator = _fitted_calibrator()
    threshold = calibrator.threshold_for_probability(0.5)
    assert calibrator.probability(threshold) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------
# Copy detection and explanations
# --------------------------------------------------------------------------


def test_canvas_iou_detects_an_exact_copy():
    canvas = np.zeros((60, 60), dtype=np.uint8)
    canvas[20:40, 20:40] = 255
    assert _canvas_iou(canvas, canvas) == pytest.approx(1.0)

    shifted = np.zeros_like(canvas)
    shifted[20:40, 30:50] = 255
    assert _canvas_iou(canvas, shifted) < 0.5


def test_canvas_iou_handles_mismatched_shapes_and_blanks():
    assert _canvas_iou(np.zeros((10, 10), np.uint8), np.zeros((20, 20), np.uint8)) == 0.0
    assert _canvas_iou(np.zeros((10, 10), np.uint8), np.zeros((10, 10), np.uint8)) == 0.0


def _result(**kwargs) -> VerificationResult:
    comparison = compare_to_references(_unit(np.arange(DIM)), _unit(np.arange(DIM)).reshape(1, -1))
    defaults = dict(
        score=90.0,
        band=Band.GREEN,
        guidance=Band.GREEN.guidance,
        comparison=comparison,
        normalized_score=3.0,
        query_canvas=np.zeros((40, 60), np.uint8),
        ink_fraction=0.05,
    )
    defaults.update(kwargs)
    return VerificationResult(**defaults)  # type: ignore[arg-type]


def test_reason_text_warns_about_a_single_specimen():
    assert "one specimen" in reason_text(_result()).lower()


def test_reason_text_leads_with_the_copy_warning():
    text = reason_text(_result(suspected_copy=True, score=99.5))
    assert "copy" in text.lower()
    assert "suspicious" in text.lower()


def test_reason_text_flags_an_uncalibrated_score():
    assert "uncalibrated" in reason_text(_result(calibrated=False)).lower()


def test_difference_overlay_marks_agreement_and_divergence():
    query = np.zeros((80, 120), dtype=np.uint8)
    query[30:50, 20:60] = 255
    reference = np.zeros_like(query)
    reference[30:50, 20:60] = 255  # identical region
    reference[30:50, 80:100] = 255  # specimen-only region

    overlay = difference_overlay(query, reference, align=False, trim=False)
    assert overlay.shape == (80, 120, 3)
    colours = {tuple(c) for c in overlay.reshape(-1, 3)}
    assert (255, 255, 255) in colours  # background survives
    assert len(colours) >= 3  # background, agreement, and specimen-only


def test_difference_overlay_trims_empty_canvas():
    """The working canvas is mostly blank; trimming keeps the strokes legible."""
    query = np.zeros((400, 600), dtype=np.uint8)
    query[190:210, 100:200] = 255
    reference = np.zeros_like(query)
    reference[190:210, 150:260] = 255

    trimmed = difference_overlay(query, reference, align=False, trim=True)
    untrimmed = difference_overlay(query, reference, align=False, trim=False)

    assert untrimmed.shape[:2] == (400, 600)
    assert trimmed.shape[0] < untrimmed.shape[0]
    assert trimmed.shape[1] < untrimmed.shape[1]
    # The strokes themselves must survive the crop.
    assert {tuple(c) for c in trimmed.reshape(-1, 3)} >= {(255, 255, 255)}
    assert (trimmed != 255).any()


def test_side_by_side_stacks_both_images():
    query = np.zeros((40, 60), dtype=np.uint8)
    reference = np.zeros((20, 30), dtype=np.uint8)
    stacked = side_by_side(query, reference, gap=10)
    assert stacked.shape == (40 + 10 + 40, 60, 3)


# --------------------------------------------------------------------------
# Verifier wiring
# --------------------------------------------------------------------------


class _StubModel:
    """Returns a fixed embedding per canvas, so wiring can be tested without training."""

    embedding_dim = DIM
    training = False

    def __init__(self, vectors: dict[int, np.ndarray]):
        self.vectors = vectors

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, tensor):
        import torch

        key = int(tensor.sum().item() * 1000) % max(len(self.vectors), 1)
        vector = list(self.vectors.values())[key]
        return torch.from_numpy(np.tile(vector, (tensor.shape[0], 1)).astype(np.float32))


def test_verifier_flags_missing_cohort_and_calibration():
    from ml.data.synth import make_signer, render_signature

    rng = np.random.default_rng(9)
    model = _StubModel({0: rng.normal(size=DIM)})
    verifier = Verifier(model, cohort=None, calibrator=None, device="cpu")  # type: ignore[arg-type]

    image = render_signature(make_signer("C1", "latin", rng), rng, kind="genuine")

    enrolment = verifier.enrol("C1", [image])
    result = verifier.verify(image, enrolment)

    assert "no_cohort_normalisation" in result.warnings
    assert "uncalibrated_score_placeholder" in result.warnings
    assert "single_reference_lower_confidence" in result.warnings
    assert result.calibrated is False
    assert result.to_dict()["advisory_only"] is True


def test_verifier_rejects_enrolment_with_no_references():
    verifier = Verifier(_StubModel({0: np.ones(DIM)}), device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no reference signatures"):
        verifier.enrol("C1", [])
