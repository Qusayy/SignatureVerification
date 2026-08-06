"""Repository-level invariants, intended to run in CI.

These are cheap and they guard the two mistakes that are hardest to notice
after the fact: a split that leaks signers between train and test, and a
production model trained on data that cannot legally back one.
"""

from __future__ import annotations

import pytest

from ml.config import ARTIFACT_ROOT, TRACK_B
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, assert_no_leakage


@pytest.mark.skipif(not DEFAULT_MANIFEST_PATH.exists(), reason="No manifest on this machine")
def test_manifest_has_no_signer_leakage():
    """No signer may appear in more than one split.

    A signer straddling train and test inflates every reported metric by an
    unpredictable margin, and nothing else in the pipeline will complain.
    """
    assert_no_leakage(Manifest.load(DEFAULT_MANIFEST_PATH))


@pytest.mark.skipif(not DEFAULT_MANIFEST_PATH.exists(), reason="No manifest on this machine")
def test_every_manifest_image_exists():
    manifest = Manifest.load(DEFAULT_MANIFEST_PATH)
    missing = [r.image_path for r in manifest.records if not manifest.resolve(r).exists()]
    assert not missing, f"{len(missing)} manifest path(s) missing, e.g. {missing[:3]}"


@pytest.mark.parametrize("checkpoint", sorted(ARTIFACT_ROOT.glob("*_track_b*.pt")))
def test_production_checkpoints_are_deployable(checkpoint):
    """Anything named track_b must actually pass the licence gate."""
    from ml.embed.provenance import assert_deployable

    provenance = assert_deployable(checkpoint)
    assert provenance.licence_track == TRACK_B.name
    assert provenance.pretrained_init is None


def test_no_biometric_data_is_tracked_by_git():
    """`data/` and `artifacts/` must be git-ignored — they hold biometric PII."""
    from ml.config import REPO_ROOT

    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "data/" in gitignore
    assert "artifacts/" in gitignore
    assert "*.pt" in gitignore
