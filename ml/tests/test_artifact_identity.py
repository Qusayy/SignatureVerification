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
from ml.scoring.calibrate import CalibratorSchemaError


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
# Artifact stamping and refusal
#
# A calibrator is a function of a specific embedding space *and* a specific
# verification protocol. Paired with other weights, or with a different number
# of stored specimens, it produces numbers in an entirely normal range that mean
# nothing. Both have shipped here.
# --------------------------------------------------------------------------


def _fitted(seed: int = 0, refs: int = 1, weights: str = "") -> ScoreCalibrator:
    rng = np.random.default_rng(seed)
    return ScoreCalibrator.fit(
        rng.normal(0.93, 0.02, 400),
        rng.normal(0.85, 0.04, 400),
        protocol_references=refs,
        weights_id=weights,
        fitted_on="val",
    )


def test_calibrator_round_trips(tmp_path):
    original = _fitted(weights="abc123")
    original.derive_band_edges(
        np.random.default_rng(1).normal(0.93, 0.02, 400),
        np.random.default_rng(2).normal(0.85, 0.04, 400),
    )
    original.save(tmp_path / "cal.json")
    reloaded = ScoreCalibrator.load(tmp_path / "cal.json")

    assert reloaded.weights_id == "abc123"
    assert reloaded.protocol_references == 1
    assert reloaded.green_min == original.green_min
    assert reloaded.red_max == original.red_max
    for s in (0.80, 0.90, 0.95):
        assert reloaded.score_0_100(s) == pytest.approx(original.score_0_100(s))


def test_a_pre_rework_calibrator_is_refused(tmp_path):
    """Schema 1 was fitted on writer-normalised margins.

    Applying it to a similarity is not a degraded result, it is a different
    scale — which is how a 4.7% match came to read 69/100.
    """
    import json

    (tmp_path / "old.json").write_text(
        json.dumps({"x": [-0.7, 0.0, 0.04], "y": [0.005, 0.5, 0.995], "fitted_on": "val"})
    )
    with pytest.raises(CalibratorSchemaError, match="schema version 1"):
        ScoreCalibrator.load(tmp_path / "old.json")


def test_thin_fit_is_stamped_and_warned():
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="below 200"):
        calibrator = ScoreCalibrator.fit(
            rng.normal(0.93, 0.02, 80),
            rng.normal(0.85, 0.04, 80),
            protocol_references=1,
        )
    assert calibrator.thin_fit


def test_fit_refuses_a_set_too_small_to_mean_anything():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Too few scores"):
        ScoreCalibrator.fit(
            rng.normal(0.93, 0.02, 20), rng.normal(0.85, 0.04, 20), protocol_references=1
        )


# --------------------------------------------------------------------------
# Refusal at load
# --------------------------------------------------------------------------


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


def test_verifier_refuses_a_calibrator_from_other_weights(tmp_path):
    from ml.embed.models import build_model
    from ml.scoring.verifier import ArtifactMismatch, Verifier

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, build_model("signet").state_dict())
    _fitted(weights="0000000000000000").save(tmp_path / "cal.json")

    with pytest.raises(ArtifactMismatch, match="cal.json"):
        Verifier.from_artifacts(
            checkpoint, calibrator_path=tmp_path / "cal.json", device="cpu"
        )


def test_verifier_refuses_a_calibrator_fitted_for_another_protocol(tmp_path):
    """The bug that made every score meaningless, made unrepresentable.

    Every shipped curve was fitted on six specimens per customer while
    production ran one.
    """
    from dataclasses import replace

    from ml.config import SCORING
    from ml.embed.models import build_model
    from ml.scoring.verifier import CalibratorUnavailable, Verifier

    model = build_model("signet")
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, model.state_dict())
    _fitted(refs=6, weights=weights_id(model.state_dict())).save(tmp_path / "cal.json")

    with pytest.raises(CalibratorUnavailable, match="fitted for 6 specimen"):
        Verifier.from_artifacts(
            checkpoint,
            calibrator_path=tmp_path / "cal.json",
            device="cpu",
            cfg=replace(SCORING, calibration_references=1),
        )


def test_verifier_refuses_to_load_without_a_calibrator(tmp_path):
    """No dash at the counter. The refusal belongs where an engineer sees it."""
    from ml.embed.models import build_model
    from ml.scoring.verifier import CalibratorUnavailable, Verifier

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, build_model("signet").state_dict())

    with pytest.raises(CalibratorUnavailable, match="benchmark"):
        Verifier.from_artifacts(
            checkpoint, calibrator_path=tmp_path / "absent.json", device="cpu"
        )


def test_verifier_accepts_matching_artifacts(tmp_path):
    from ml.embed.models import build_model
    from ml.scoring.verifier import Verifier

    model = build_model("signet")
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, model.state_dict())
    identity = weights_id(model.state_dict())
    _fitted(weights=identity).save(tmp_path / "cal.json")

    verifier = Verifier.from_artifacts(
        checkpoint, calibrator_path=tmp_path / "cal.json", device="cpu"
    )
    assert verifier.model_version == f"signet@{identity}"
    assert "unknown" not in verifier.model_version
