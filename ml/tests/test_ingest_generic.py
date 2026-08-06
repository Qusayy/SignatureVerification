"""Tests for third-party dataset inspection and ingestion.

The single most consequential mistake in ingesting a public signature dataset
is mislabelling forgeries as genuine. It does not surface as an error, and it
does not surface as worse validation accuracy — it surfaces as *better*
accuracy, because the model is being rewarded for accepting forgeries. So the
labelling rules get tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ml.config import TRACK_A, TRACK_B
from ml.data.ingest import ingest_generic
from ml.data.inspect import classify, inspect_root


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((40, 80), 255, dtype=np.uint8)).save(path)


# --------------------------------------------------------------------------
# Label classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "forgery", "forgeries", "full_forg", "forged", "skilled_forgery",
        "fake", "F", "_f_", "disguised", "simulated",
        "forgeries_12_04.png", "SKILLED", "Forgery",
    ],
)
def test_forgery_names_classify_as_forgery(name):
    assert classify(name) == "skilled_forgery"


@pytest.mark.parametrize(
    "name",
    ["genuine", "original", "originals", "full_org", "real", "authentic",
     "reference", "original_12_04.png", "Genuine"],
)
def test_genuine_names_classify_as_genuine(name):
    assert classify(name) == "genuine"


def test_forgery_hints_beat_genuine_substrings():
    """The bug this guards: 'org' is a substring of 'forgery' and 'full_forg'.

    Without forgery-first precedence both hints match, the result is ambiguous,
    and every forgery in the dataset is either dropped or — far worse — guessed
    genuine.
    """
    assert "org" in "forgery"
    assert "org" in "full_forg"
    assert classify("forgery") == "skilled_forgery"
    assert classify("full_forg") == "skilled_forgery"
    assert classify("full_org") == "genuine"


@pytest.mark.parametrize("name", ["person_0042", "train", "images", "S0001", "data"])
def test_neutral_names_are_unclassified(name):
    assert classify(name) is None


# --------------------------------------------------------------------------
# Generic ingestion
# --------------------------------------------------------------------------


@pytest.fixture
def folder_dataset(tmp_path):
    """<root>/<signer>/{genuine,forgery}/*.png — the most common layout."""
    for signer in ("p001", "p002", "p003"):
        for i in range(3):
            _write_image(tmp_path / signer / "genuine" / f"{i}.png")
        for i in range(2):
            _write_image(tmp_path / signer / "forgery" / f"{i}.png")
    return tmp_path


def test_folder_layout_ingests_with_correct_labels(folder_dataset, tmp_path):
    records = ingest_generic(folder_dataset, tmp_path, source="ds", layout="folder")

    assert len(records) == 15
    labels = {}
    for r in records:
        labels[r.label] = labels.get(r.label, 0) + 1
    assert labels == {"genuine": 9, "skilled_forgery": 6}
    assert {r.signer_id for r in records} == {"ds:p001", "ds:p002", "ds:p003"}


def test_source_namespaces_signer_ids(folder_dataset, tmp_path):
    """Two datasets must never merge two different people into one identity."""
    a = ingest_generic(folder_dataset, tmp_path, source="hf", layout="folder")
    b = ingest_generic(folder_dataset, tmp_path, source="kaggle", layout="folder")
    assert not ({r.signer_id for r in a} & {r.signer_id for r in b})


def test_defaults_to_research_track(folder_dataset, tmp_path):
    """Third-party data is Track A until someone checks the original licence."""
    records = ingest_generic(folder_dataset, tmp_path, source="ds", layout="folder")
    assert all(r.licence_track == TRACK_A.name for r in records)


def test_track_b_must_be_requested_explicitly(folder_dataset, tmp_path):
    records = ingest_generic(
        folder_dataset, tmp_path, source="ds", layout="folder", licence_track=TRACK_B.name
    )
    assert all(r.licence_track == TRACK_B.name for r in records)


def test_script_is_recorded_for_per_script_metrics(folder_dataset, tmp_path):
    records = ingest_generic(
        folder_dataset, tmp_path, source="ds", layout="folder", script="arabic"
    )
    assert all(r.script == "arabic" for r in records)


def test_unknown_script_is_rejected(folder_dataset, tmp_path):
    with pytest.raises(ValueError, match="Unknown script"):
        ingest_generic(folder_dataset, tmp_path, source="ds", layout="folder", script="klingon")


def test_unclassifiable_images_refuse_rather_than_guess(tmp_path):
    """Refusing is the whole point — a wrong guess is invisible downstream."""
    for i in range(4):
        _write_image(tmp_path / "person_01" / f"sig_{i}.png")

    with pytest.raises(ValueError, match="could not be classified"):
        ingest_generic(tmp_path, tmp_path, source="ds", layout="folder")


def test_no_forgeries_flag_allows_genuine_only_corpora(tmp_path):
    for signer in ("a", "b"):
        for i in range(3):
            _write_image(tmp_path / signer / f"sig_{i}.png")

    records = ingest_generic(tmp_path, tmp_path, source="ds", layout="folder", no_forgeries=True)
    assert len(records) == 6
    assert all(r.label == "genuine" for r in records)


def test_filename_layout(tmp_path):
    """Flat directory with the label encoded in the file name."""
    for signer in ("01", "02"):
        for i in range(3):
            _write_image(tmp_path / signer / f"original_{signer}_{i}.png")
        for i in range(2):
            _write_image(tmp_path / signer / f"forgeries_{signer}_{i}.png")

    records = ingest_generic(tmp_path, tmp_path, source="cedar-like", layout="filename")
    assert sum(r.label == "genuine" for r in records) == 6
    assert sum(r.label == "skilled_forgery" for r in records) == 4


def test_signer_depth_selects_the_right_path_component(tmp_path):
    """Datasets often nest signers under a split folder."""
    for split in ("train", "test"):
        for signer in ("p1", "p2"):
            _write_image(tmp_path / split / signer / "genuine" / "a.png")

    deep = ingest_generic(tmp_path, tmp_path, source="ds", layout="folder", signer_depth=1)
    assert {r.signer_id for r in deep} == {"ds:p1", "ds:p2"}

    shallow = ingest_generic(tmp_path, tmp_path, source="ds", layout="folder", signer_depth=0)
    assert {r.signer_id for r in shallow} == {"ds:train", "ds:test"}


def test_missing_root_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_generic(tmp_path / "nope", tmp_path, source="ds", layout="folder")


# --------------------------------------------------------------------------
# Inspector
# --------------------------------------------------------------------------


def test_inspector_finds_folder_encoded_labels(folder_dataset):
    report = inspect_root(folder_dataset)
    assert report["image_count"] == 15
    assert report["labels_from_folder_names"] == {"genuine": 9, "skilled_forgery": 6}


def test_inspector_reports_packed_datasets(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train-00000.parquet").write_bytes(b"not really parquet")
    report = inspect_root(tmp_path)
    assert report["image_count"] == 0
    assert report["tabular_files"]
