"""Tests for the manifest, splits, guards, and capture augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from ml.config import TRACK_A, TRACK_B
from ml.data.augment import DEFAULT_AUGMENT, AugmentConfig, augment_capture
from ml.data.ingest import ingest_synthetic, reference_profile
from ml.data.manifest import (
    LeakageError,
    LicenceError,
    Manifest,
    Record,
    assert_no_leakage,
    assert_track_b,
    split_by_signer,
)
from ml.data.synth import generate_corpus, make_signer, render_signature
from ml.preprocess.pipeline import preprocess_signature


def _record(signer: str, label: str = "genuine", script: str = "latin", source: str = "synthetic") -> Record:
    return Record(
        image_path=f"{signer}/{label}.png",
        signer_id=signer,
        label=label,  # type: ignore[arg-type]
        script=script,
        source=source,
        licence_track=TRACK_B.name if source == "synthetic" else TRACK_A.name,
    )


def _manifest(n_signers: int = 40, arabic_every: int = 2) -> Manifest:
    records = []
    for i in range(n_signers):
        signer = f"S{i:03d}"
        script = "arabic" if i % arabic_every == 0 else "latin"
        for j in range(4):
            records.append(_record(signer, "genuine" if j < 3 else "skilled_forgery", script))
    return Manifest(records=records)


# --------------------------------------------------------------------------
# Licence tagging
# --------------------------------------------------------------------------


def test_non_commercial_source_is_forced_to_track_a():
    """A caller cannot mislabel CEDAR as production-safe, even explicitly."""
    record = Record(
        image_path="x.png",
        signer_id="cedar:0001",
        label="genuine",
        source="cedar",
        licence_track=TRACK_B.name,  # wrong, and must be overridden
    )
    assert record.licence_track == TRACK_A.name


def test_assert_track_b_rejects_research_data():
    records = [_record("S1"), _record("S2", source="cedar")]
    with pytest.raises(LicenceError, match="cedar"):
        assert_track_b(records)


def test_assert_track_b_accepts_clean_data():
    assert_track_b([_record("S1"), _record("S2")])


def test_unknown_script_is_rejected():
    with pytest.raises(ValueError, match="Unknown script"):
        Record(image_path="x.png", signer_id="S1", label="genuine", script="klingon")


# --------------------------------------------------------------------------
# Splitting and leakage
# --------------------------------------------------------------------------


def test_split_partitions_signers_not_images():
    manifest = split_by_signer(_manifest(), test_frac=0.2, val_frac=0.1, seed=7)
    train, val, test = (manifest.signers(s) for s in ("train", "val", "test"))
    assert train and val and test
    assert not (train & test) and not (train & val) and not (val & test)
    assert train | val | test == manifest.signers()


def test_split_is_deterministic_for_a_seed():
    a = split_by_signer(_manifest(), seed=99)
    b = split_by_signer(_manifest(), seed=99)
    assert {r.signer_id: r.split for r in a.records} == {r.signer_id: r.split for r in b.records}


def test_split_stratifies_by_script():
    """Both scripts must reach the test set, or per-script metrics are noise."""
    manifest = split_by_signer(_manifest(n_signers=60), test_frac=0.2, seed=3)
    test_scripts = {r.script for r in manifest.by_split("test")}
    assert {"arabic", "latin"} <= test_scripts


def test_leakage_is_detected():
    manifest = _manifest(n_signers=10)
    split_by_signer(manifest, seed=1)
    # Force one signer's record into a different split.
    victim = manifest.records[0].signer_id
    for record in manifest.records:
        if record.signer_id == victim:
            record.split = "train" if record.split == "test" else "test"
            break
    with pytest.raises(LeakageError, match=victim):
        assert_no_leakage(manifest)


def test_unsplit_records_do_not_trip_the_leakage_guard():
    assert_no_leakage(_manifest(n_signers=5))  # split is None everywhere


def test_split_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        split_by_signer(_manifest(), test_frac=0.8, val_frac=0.5)


def test_manifest_round_trips_through_disk(tmp_path):
    manifest = split_by_signer(_manifest(n_signers=12), seed=5)
    path = manifest.save(tmp_path / "manifest.json")
    reloaded = Manifest.load(path)
    assert len(reloaded) == len(manifest)
    assert reloaded.stats()["by_split"] == manifest.stats()["by_split"]
    assert_no_leakage(reloaded)


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def test_ingest_synthetic_round_trip(tmp_path):
    corpus = tmp_path / "synth"
    generate_corpus(corpus, n_signers=6, genuine_per_signer=4, forgeries_per_signer=2, forms_per_signer=0, seed=5)

    records = ingest_synthetic(corpus, manifest_root=tmp_path)
    assert len(records) == 6 * (4 + 2)
    assert all(r.licence_track == TRACK_B.name for r in records)
    assert all(r.signer_id.startswith("synthetic:") for r in records)
    assert {r.label for r in records} == {"genuine", "skilled_forgery"}

    manifest = Manifest(records=records, root=tmp_path)
    assert all(manifest.resolve(r).exists() for r in records)


def test_reference_profile_reports_the_phase_zero_question():
    records = [_record("S1"), _record("S2"), _record("S3")]
    records[0].is_reference = True
    records[1].is_reference = True
    profile = reference_profile(records)
    assert profile["signers"] == 3
    assert profile["signers_with_references"] == 2
    assert profile["signers_without_references"] == 1


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------


@pytest.fixture
def raw_signature() -> np.ndarray:
    rng = np.random.default_rng(4)
    return render_signature(make_signer("A1", "latin", rng), rng, kind="genuine")


def test_augmentation_output_is_still_a_usable_signature(raw_signature):
    """Every augmented sample must survive preprocessing, not become noise."""
    for seed in range(25):
        rng = np.random.default_rng(seed)
        augmented = augment_capture(raw_signature, rng)
        assert augmented.ndim == 2
        assert augmented.dtype == np.uint8
        result = preprocess_signature(augmented, strict=False)
        assert result.is_usable, f"seed {seed} produced an unusable crop"


def test_augmentation_actually_changes_the_image(raw_signature):
    always = AugmentConfig(
        p_rotate=1.0, p_scale=1.0, p_pen_width=1.0, p_blur=1.0, p_perspective=1.0,
        p_illumination=1.0, p_noise=1.0, p_jpeg=1.0, p_ruled_line=1.0, p_stamp=1.0, p_ink_bleed=1.0,
    )
    augmented = augment_capture(raw_signature, np.random.default_rng(0), always)
    assert augmented.shape != raw_signature.shape or not np.array_equal(augmented, raw_signature)


def test_augmentation_is_deterministic_per_seed(raw_signature):
    a = augment_capture(raw_signature, np.random.default_rng(11), DEFAULT_AUGMENT)
    b = augment_capture(raw_signature, np.random.default_rng(11), DEFAULT_AUGMENT)
    assert np.array_equal(a, b)


def test_rotation_stays_small_enough_to_preserve_identity():
    """Guard the config itself: large rotations would destroy the signal."""
    assert DEFAULT_AUGMENT.max_rotate_deg <= 5.0
