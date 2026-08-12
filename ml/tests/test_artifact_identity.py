"""Tests for artifact identity and the scoring recipe.

The bug these exist for was not a crash. ``artifacts/cohort.npz`` and
``artifacts/calibrator.json`` had been produced by one checkpoint while the
service loaded another, and the only visible symptom was that two thirds of
skilled forgeries scored 99.5 out of 100 — which looks exactly like a system
working perfectly.

Nothing detected it because ``model_version`` was
``f"{architecture}@{git_commit[:8]}"`` and every checkpoint reported
``signet@unknown``, so every staleness comparison was between two identical
meaningless strings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.embed.provenance import UNKNOWN_WEIGHTS_ID, weights_id
from ml.scoring.calibrate import ScoreCalibrator
from ml.scoring.compare import compare_to_references, intra_reference_mean
from ml.scoring.znorm import CohortNormalizer


# --------------------------------------------------------------------------
# weights_id
# --------------------------------------------------------------------------


def test_identical_weights_hash_identically():
    state = {"a": torch.ones(4, 4), "b": torch.zeros(3)}
    assert weights_id(state) == weights_id({"b": torch.zeros(3), "a": torch.ones(4, 4)})


def test_a_single_changed_element_changes_the_hash():
    state = {"a": torch.ones(4, 4)}
    other = {"a": torch.ones(4, 4)}
    other["a"][0, 0] = 1.001
    assert weights_id(state) != weights_id(other)


def test_shape_is_part_of_the_identity():
    assert weights_id({"a": torch.zeros(4)}) != weights_id({"a": torch.zeros(5)})


def test_parameter_names_are_part_of_the_identity():
    assert weights_id({"a": torch.ones(2)}) != weights_id({"b": torch.ones(2)})


def test_non_tensor_entries_do_not_break_hashing():
    """Some heads carry plain counters in their state dict."""
    assert weights_id({"a": torch.ones(2), "steps": 7}) != weights_id(
        {"a": torch.ones(2), "steps": 8}
    )


# --------------------------------------------------------------------------
# Artifact stamping
# --------------------------------------------------------------------------


def test_cohort_round_trips_its_weights_stamp(tmp_path):
    cohort = _cohort(dim=8)
    cohort.save(tmp_path / "cohort.npz", weights_id="deadbeefdeadbeef")
    assert CohortNormalizer.load(tmp_path / "cohort.npz").weights_id == "deadbeefdeadbeef"


def test_cohort_without_a_stamp_loads_as_unverifiable(tmp_path):
    """Cohorts written before the stamp existed must not masquerade as matching."""
    cohort = _cohort(dim=8)
    np.savez_compressed(
        tmp_path / "old.npz",
        embeddings=cohort.embeddings.astype(np.float32),
        signers=np.array(cohort.signers),
    )
    assert CohortNormalizer.load(tmp_path / "old.npz").weights_id == UNKNOWN_WEIGHTS_ID


def test_calibrator_round_trips_its_weights_stamp(tmp_path):
    rng = np.random.default_rng(0)
    calibrator = ScoreCalibrator.fit(
        rng.normal(1.0, 0.3, 60), rng.normal(0.0, 0.3, 60), weights_id="abc123"
    )
    calibrator.save(tmp_path / "cal.json")
    assert ScoreCalibrator.load(tmp_path / "cal.json").weights_id == "abc123"


def test_calibrator_warns_on_a_thin_fit():
    """The shipped curve was fitted on 72/96 and saturated at 0.995."""
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="isotonic fit degenerates"):
        ScoreCalibrator.fit(rng.normal(1.0, 0.3, 40), rng.normal(0.0, 0.3, 40))


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


def _cohort(dim: int = 512, n: int = 120) -> CohortNormalizer:
    """A cohort large enough to satisfy the impostor-distribution guard."""
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(n, dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return CohortNormalizer(vectors, [f"s{i:04d}" for i in range(n)])


def _write_checkpoint(path, state):
    torch.save(
        {
            "model_state": state,
            "architecture": "signet",
            "embedding_dim": 512,
            "weights_id": weights_id(state),
            "config": {},
            "provenance": {},
        },
        path,
    )


def test_verifier_refuses_a_cohort_from_other_weights(tmp_path):
    from ml.embed.models import build_model
    from ml.scoring.verifier import ArtifactMismatch, Verifier

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, build_model("signet").state_dict())

    _cohort().save(tmp_path / "cohort.npz", weights_id="0000000000000000")

    with pytest.raises(ArtifactMismatch, match="cohort.npz"):
        Verifier.from_artifacts(
            checkpoint,
            cohort_path=tmp_path / "cohort.npz",
            calibrator_path=tmp_path / "absent.json",
            device="cpu",
        )


def test_verifier_refuses_an_unstamped_cohort(tmp_path):
    """The exact state the shipped artifacts were in."""
    from ml.embed.models import build_model
    from ml.scoring.verifier import ArtifactMismatch, Verifier

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, build_model("signet").state_dict())

    unstamped = _cohort()
    np.savez_compressed(
        tmp_path / "cohort.npz",
        embeddings=unstamped.embeddings.astype(np.float32),
        signers=np.array(unstamped.signers),
    )

    with pytest.raises(ArtifactMismatch, match="predates this check"):
        Verifier.from_artifacts(
            checkpoint,
            cohort_path=tmp_path / "cohort.npz",
            calibrator_path=tmp_path / "absent.json",
            device="cpu",
        )


def test_verifier_accepts_matching_artifacts(tmp_path):
    from ml.embed.models import build_model
    from ml.scoring.verifier import Verifier

    model = build_model("signet")
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, model.state_dict())
    identity = weights_id(model.state_dict())

    _cohort().save(tmp_path / "cohort.npz", weights_id=identity)

    verifier = Verifier.from_artifacts(
        checkpoint,
        cohort_path=tmp_path / "cohort.npz",
        calibrator_path=tmp_path / "absent.json",
        device="cpu",
    )
    # The version string identifies the weights, not the commit that built them.
    assert verifier.model_version == f"signet@{identity}"
    assert "unknown" not in verifier.model_version


def test_missing_artifacts_are_still_tolerated(tmp_path):
    """The stack must start before a benchmark has ever been run."""
    from ml.embed.models import build_model
    from ml.scoring.verifier import Verifier

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, build_model("signet").state_dict())

    verifier = Verifier.from_artifacts(
        checkpoint,
        cohort_path=tmp_path / "absent.npz",
        calibrator_path=tmp_path / "absent.json",
        device="cpu",
    )
    assert verifier.cohort is None
    assert verifier.calibrator.is_placeholder


# --------------------------------------------------------------------------
# Writer-internal normalisation
# --------------------------------------------------------------------------


def test_intra_reference_mean_is_the_off_diagonal_mean():
    refs = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Two orthogonal specimens: every cross-pair similarity is 0.
    assert intra_reference_mean(refs) == pytest.approx(0.0, abs=1e-9)

    identical = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert intra_reference_mean(identical) == pytest.approx(1.0, abs=1e-9)


def test_intra_reference_mean_needs_two_specimens():
    assert intra_reference_mean(np.array([[1.0, 0.0]])) == 0.0


def test_writer_normalisation_subtracts_specimen_agreement():
    rng = np.random.default_rng(3)
    refs = rng.normal(size=(4, 32))
    query = rng.normal(size=32)

    plain = compare_to_references(query, refs, writer_normalise=False)
    normalised = compare_to_references(query, refs, writer_normalise=True)

    assert normalised.raw == pytest.approx(plain.raw - intra_reference_mean(refs))
    assert normalised.is_writer_normalised
    assert not plain.is_writer_normalised


def test_a_consistent_writer_is_held_to_a_stricter_standard():
    """The whole point: the same similarity means different things per customer.

    Two customers, both scoring 0.90 against their specimens. One signs almost
    identically every time; the other varies. The consistent one has produced
    something unusual for them, and the score has to say so.
    """
    dim = 64
    base = np.zeros(dim)
    base[0] = 1.0

    def rotated(angle: float) -> np.ndarray:
        vector = np.zeros(dim)
        vector[0] = np.cos(angle)
        vector[1] = np.sin(angle)
        return vector

    consistent = np.vstack([rotated(0.0), rotated(0.02), rotated(-0.02)])
    variable = np.vstack([rotated(0.0), rotated(0.6), rotated(-0.6)])

    query = rotated(0.30)
    strict = compare_to_references(query, consistent)
    lenient = compare_to_references(query, variable)

    assert strict.raw < lenient.raw


def test_single_specimen_falls_back_rather_than_subtracting_zero():
    refs = np.array([[1.0, 0.0, 0.0]])
    score = compare_to_references(np.array([1.0, 0.0, 0.0]), refs, writer_normalise=True)
    assert score.raw == pytest.approx(1.0)
    assert not score.is_writer_normalised


def test_precomputed_reference_mean_matches_recomputation():
    rng = np.random.default_rng(11)
    refs = rng.normal(size=(5, 16))
    query = rng.normal(size=16)

    cached = compare_to_references(query, refs, reference_mean=intra_reference_mean(refs))
    fresh = compare_to_references(query, refs)
    assert cached.raw == pytest.approx(fresh.raw)


# --------------------------------------------------------------------------
# The single-specimen ceiling
#
# With one specimen a customer's own consistency cannot be measured, so the
# baseline was 0 and `raw` came through as a bare similarity of ~0.8-1.0. The
# calibrator is fitted on margins whose domain ends near +0.04, so every one of
# those clipped to the top of the curve: every single-specimen verification
# returned ~100, forgeries included. That is the configuration a bank holding
# one signature per customer is actually in.
# --------------------------------------------------------------------------


def test_single_specimen_uses_the_population_baseline():
    refs = np.array([[1.0, 0.0, 0.0]])
    query = np.array([0.9, 0.436, 0.0])

    score = compare_to_references(query, refs, population_reference_mean=0.95)

    assert score.baseline_source == "population"
    assert score.intra_reference_mean == pytest.approx(0.95)
    # The margin, not the similarity: comfortably inside the calibrator domain.
    assert score.raw == pytest.approx(0.9 - 0.95, abs=1e-3)


def test_single_specimen_without_a_population_baseline_is_flagged():
    """Better to declare the score uncomparable than to quietly emit one."""
    refs = np.array([[1.0, 0.0, 0.0]])
    score = compare_to_references(np.array([1.0, 0.0, 0.0]), refs)

    assert score.baseline_source == "none"
    assert score.raw == pytest.approx(1.0)


def test_several_specimens_prefer_their_own_baseline():
    rng = np.random.default_rng(4)
    refs = rng.normal(size=(3, 16))
    query = rng.normal(size=16)

    score = compare_to_references(query, refs, population_reference_mean=0.95)

    assert score.baseline_source == "own"
    assert score.intra_reference_mean == pytest.approx(intra_reference_mean(refs))


def test_single_specimen_scores_do_not_all_clip_to_the_ceiling():
    """The regression itself, end to end through the calibrator.

    A good and a bad single-specimen query must not both return the same
    saturated score.
    """
    rng = np.random.default_rng(7)
    # A calibrator fitted on margins, as the real one is.
    genuine = rng.normal(0.01, 0.02, 300)
    impostor = rng.normal(-0.12, 0.05, 300)
    calibrator = ScoreCalibrator.fit(
        genuine, impostor, population_reference_mean=0.95
    )

    reference = np.zeros(8)
    reference[0] = 1.0
    refs = reference.reshape(1, -1)

    def score(cosine: float) -> float:
        query = np.zeros(8)
        query[0], query[1] = cosine, np.sqrt(max(0.0, 1 - cosine**2))
        raw = compare_to_references(
            query, refs, population_reference_mean=calibrator.population_reference_mean
        ).raw
        return calibrator.score_0_100(raw)

    close, distant = score(0.97), score(0.80)
    assert close > distant, "single-specimen scores are not discriminating"
    assert distant < 50, f"a poor single-specimen match still scored {distant}"


def test_population_baseline_survives_a_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    calibrator = ScoreCalibrator.fit(
        rng.normal(0.01, 0.02, 300),
        rng.normal(-0.12, 0.05, 300),
        population_reference_mean=0.9575,
    )
    calibrator.save(tmp_path / "cal.json")
    assert ScoreCalibrator.load(tmp_path / "cal.json").population_reference_mean == pytest.approx(
        0.9575
    )
