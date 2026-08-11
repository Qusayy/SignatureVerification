"""Test fixtures: an isolated API instance backed by a real trained model.

These are end-to-end tests, not mocked ones. They run the actual preprocessing,
the actual embedding model, and the actual scoring chain, because the wiring
between those pieces is exactly where this system breaks — a mocked verifier
would pass while the real one returned nonsense.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.config import ARTIFACT_ROOT

def _matching_checkpoint() -> Path:
    """The checkpoint that `artifacts/cohort.npz` was actually built from.

    These are end-to-end tests against the real artifact directory, and the
    service now refuses to pair a cohort or calibrator with weights that did
    not produce it. Hard-coding a checkpoint name here made the whole suite
    depend on which model was benchmarked last — which is exactly the coupling
    that let a mismatched set reach the demo unnoticed.
    """
    from ml.embed.provenance import read_weights_id
    from ml.scoring.znorm import CohortNormalizer

    candidates = sorted(ARTIFACT_ROOT.glob("*.pt"))
    cohort_path = ARTIFACT_ROOT / "cohort.npz"
    if cohort_path.exists():
        wanted = getattr(CohortNormalizer.load(cohort_path), "weights_id", "")
        for path in candidates:
            try:
                if wanted and read_weights_id(path) == wanted:
                    return path
            except Exception:  # noqa: BLE001 - a broken checkpoint is simply not a match
                continue
    return candidates[0] if candidates else ARTIFACT_ROOT / "signet_track_b.pt"


CHECKPOINT = _matching_checkpoint()
requires_model = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"No checkpoint in {ARTIFACT_ROOT}. Run: python -m ml.embed.train",
)


@pytest.fixture
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the API at a throwaway database, storage root, and encryption key."""
    from cryptography.fernet import Fernet

    monkeypatch.setenv("SV_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("SV_STORAGE_LOCAL_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("SV_IMAGE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    # At least 32 bytes: HS256 rejects shorter keys as weak (RFC 7518 §3.2).
    monkeypatch.setenv("SV_JWT_SECRET", "test-secret-not-the-default-but-long-enough-for-hs256")
    monkeypatch.setenv("SV_CHECKPOINT_PATH", str(CHECKPOINT))
    monkeypatch.delenv("SV_STORAGE_ENDPOINT", raising=False)

    import api.db as db
    import api.services.inference as inference
    import api.services.storage as storage
    from api.settings import get_settings

    get_settings.cache_clear()
    db.reset_engine()
    storage.reset_store()
    inference.reset_service()

    yield tmp_path

    get_settings.cache_clear()
    db.reset_engine()
    storage.reset_store()
    inference.reset_service()


@pytest.fixture
def client(api_env):
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def employee(api_env):
    """A signed-up employee to authenticate as."""
    from api.db import get_sessionmaker, init_db
    from api.seed import seed_employee

    init_db()
    session = get_sessionmaker()()
    try:
        record = seed_employee(session, "tester", "test-password")
        session.commit()
        return {"username": record.username, "password": "test-password", "id": record.id}
    finally:
        session.close()


@pytest.fixture
def auth(client, employee) -> dict[str, str]:
    response = client.post(
        "/api/auth/token",
        data={"username": employee["username"], "password": employee["password"]},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def signatures():
    """Genuine samples and a skilled forgery for one synthetic signer."""
    from ml.data.synth import make_signer, render_signature

    rng = np.random.default_rng(2024)
    style = make_signer("TESTER", "latin", rng)
    genuine = [
        render_signature(style, np.random.default_rng(100 + i), kind="genuine") for i in range(5)
    ]
    forgery = render_signature(style, np.random.default_rng(500), kind="forgery")

    other_style = make_signer("OTHER", "arabic", np.random.default_rng(77))
    other = render_signature(other_style, np.random.default_rng(78), kind="genuine")

    return {"genuine": genuine, "forgery": forgery, "other_writer": other}


def as_upload(image: np.ndarray, name: str = "signature.png") -> tuple[str, bytes, str]:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return (name, buffer.tobytes(), "image/png")
