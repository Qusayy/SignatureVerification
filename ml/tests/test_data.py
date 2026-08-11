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


# --------------------------------------------------------------------------
# Embedding cache isolation
#
# The cache key used to be f"{split}_{index}". Position in a manifest is not a
# stable identity: `ml.data.ingest` appends and `manifest split` reshuffles, so
# both renumber records. Augmented reads bypass the cache, so a stale entry
# never reached training — it reached validation and test, which is every
# number the project reports, silently and with plausible-looking images.
# --------------------------------------------------------------------------


def test_cache_fingerprint_distinguishes_two_corpora(tmp_path):
    from ml.embed.dataset import corpus_fingerprint

    a = _manifest_with(tmp_path, ["a/one.png", "a/two.png"])
    b = _manifest_with(tmp_path, ["b/one.png", "b/two.png"])
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_cache_fingerprint_changes_when_the_split_changes(tmp_path):
    from ml.embed.dataset import corpus_fingerprint

    manifest = _manifest_with(tmp_path, ["x/one.png", "x/two.png"])
    before = corpus_fingerprint(manifest)
    manifest.records[0].split = "test"
    assert corpus_fingerprint(manifest) != before


def test_cache_fingerprint_is_order_independent(tmp_path):
    """Appending a source must not invalidate entries for the others."""
    from ml.embed.dataset import corpus_fingerprint

    forward = _manifest_with(tmp_path, ["p/one.png", "p/two.png"])
    reversed_ = _manifest_with(tmp_path, ["p/two.png", "p/one.png"])
    assert corpus_fingerprint(forward) == corpus_fingerprint(reversed_)


def _manifest_with(root, paths):
    from ml.data.manifest import Manifest, Record

    return Manifest(
        root=root,
        records=[
            Record(image_path=p, signer_id=f"s:{i}", label="genuine", split="train")
            for i, p in enumerate(paths)
        ],
    )


def test_two_manifests_sharing_a_cache_do_not_collide(tmp_path):
    """The real failure: same cache directory, different corpora, silent reuse."""
    import cv2

    from ml.data.manifest import Manifest, Record
    from ml.embed.dataset import SignatureDataset

    cache = tmp_path / "cache"

    def corpus(name: str, shape: str) -> Manifest:
        root = tmp_path / name
        (root / "s").mkdir(parents=True, exist_ok=True)
        image = np.full((80, 200), 255, np.uint8)
        if shape == "loop":
            cv2.ellipse(image, (100, 40), (70, 22), 0, 0, 300, 0, 4)
        else:
            # A visibly different mark: preprocessing normalises tone and size
            # away, so the two corpora have to differ in shape to differ at all.
            for x in range(20, 180, 20):
                cv2.line(image, (x, 15), (x + 10, 65), 0, 4)
        cv2.imwrite(str(root / "s" / "one.png"), image)
        return Manifest(
            root=root,
            records=[
                Record(image_path="s/one.png", signer_id="s:0", label="genuine", split="train")
            ],
        )

    first = SignatureDataset(corpus("first", "loop"), "train", augment=False, cache_dir=cache)
    second = SignatureDataset(corpus("second", "zigzag"), "train", augment=False, cache_dir=cache)

    a = first[0]["image"].numpy()
    b = second[0]["image"].numpy()

    # Different source images must not resolve to the same cached canvas.
    assert not np.array_equal(a, b)
    assert (cache / "cache_meta.json").exists()
