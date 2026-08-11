"""Tests for the failure modes that produce an unexplained 500.

Each of these was a real fault. They share a shape: nothing raises where the
mistake is made, and the error surfaces somewhere unrelated — an image request,
a customer lookup — with a message that names a symptom rather than a cause.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import text

from api import db as db_module
from api.settings import Settings


# --------------------------------------------------------------------------
# Encryption key stability
# --------------------------------------------------------------------------


def test_generated_encryption_key_is_stable_across_processes(tmp_path):
    """The bug: `seed` wrote under one key and `uvicorn` read under another.

    Nothing failed at write time. Every subsequent image request returned 500
    with "could not decrypt", and the cause — that the key is regenerated per
    process — was three files away from the symptom.
    """
    settings = Settings(dev_key_path=tmp_path / ".image_key", image_encryption_key=None)

    first = settings.resolved_encryption_key()
    # A separate Settings instance stands in for a separate process.
    second = Settings(
        dev_key_path=tmp_path / ".image_key", image_encryption_key=None
    ).resolved_encryption_key()

    assert first == second


def test_explicit_key_wins_over_the_cached_one(tmp_path):
    from cryptography.fernet import Fernet

    explicit = Fernet.generate_key().decode()
    (tmp_path / ".image_key").write_text(Fernet.generate_key().decode(), encoding="utf-8")

    settings = Settings(dev_key_path=tmp_path / ".image_key", image_encryption_key=explicit)
    assert settings.resolved_encryption_key() == explicit


def test_production_refuses_to_generate_a_key(tmp_path):
    """Silently encrypting production data under a generated key is worse than
    failing to start: the key is only as durable as the file beside it."""
    settings = Settings(
        environment="production",
        dev_key_path=tmp_path / ".image_key",
        image_encryption_key=None,
    )
    with pytest.raises(RuntimeError, match="SV_IMAGE_ENCRYPTION_KEY must be set"):
        settings.resolved_encryption_key()


def test_images_survive_a_restart_without_an_explicit_key(tmp_path, monkeypatch):
    """End to end: write with one store, read with a freshly built one."""
    from api.services import storage as storage_module

    monkeypatch.setattr(
        storage_module,
        "get_settings",
        lambda: Settings(
            dev_key_path=tmp_path / ".image_key",
            image_encryption_key=None,
            storage_local_root=tmp_path / "storage",
        ),
    )

    image = np.full((12, 12), 128, dtype=np.uint8)

    storage_module.reset_store()
    key = storage_module.get_store().put_image(image)

    storage_module.reset_store()  # stands in for a process restart
    assert np.array_equal(storage_module.get_store().get_image(key), image)

    storage_module.reset_store()


# --------------------------------------------------------------------------
# Schema drift
# --------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A throwaway SQLite database wired into the module-level engine."""
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setattr(
        db_module, "get_settings", lambda: Settings(database_url=url, dev_key_path=tmp_path / "k")
    )
    db_module.reset_engine()
    yield
    db_module.reset_engine()


def test_fresh_database_reports_no_drift(temp_db):
    db_module.init_db()
    drift = db_module.schema_drift()

    assert drift == {"missing_tables": [], "missing_columns": [], "unexpected_columns": []}
    assert not db_module.schema_is_stale()


def test_renamed_column_is_detected(temp_db):
    """The actual fault: `employees.branch` was renamed to `employees.location`.

    `create_all` does not alter an existing table, so the old column stayed and
    the new one never appeared. Every authenticated request failed, because the
    dependency that loads the current operator selects that column.
    """
    db_module.init_db()

    engine = db_module.get_engine()
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE employees DROP COLUMN location"))
        connection.execute(text("ALTER TABLE employees ADD COLUMN branch VARCHAR"))

    drift = db_module.schema_drift()
    assert "employees.location" in drift["missing_columns"]
    assert "employees.branch" in drift["unexpected_columns"]
    assert db_module.schema_is_stale()


def test_init_db_does_not_repair_drift(temp_db):
    """Guards the assumption that produced the bug.

    If `create_all` ever did add the column, the detection above would be dead
    code and could be removed. It does not, so it is not.
    """
    db_module.init_db()
    with db_module.get_engine().begin() as connection:
        connection.execute(text("ALTER TABLE employees DROP COLUMN location"))

    db_module.init_db()
    assert "employees.location" in db_module.schema_drift()["missing_columns"]


def test_missing_table_is_not_treated_as_staleness(temp_db):
    """`init_db` creates absent tables, so their absence is not a blocker."""
    drift = db_module.schema_drift()
    assert drift["missing_tables"]
    assert not db_module.schema_is_stale(drift)


def test_recreate_schema_clears_drift(temp_db):
    db_module.init_db()
    with db_module.get_engine().begin() as connection:
        connection.execute(text("ALTER TABLE employees DROP COLUMN location"))
    assert db_module.schema_is_stale()

    db_module.recreate_schema()
    assert not db_module.schema_is_stale()


def test_recreate_schema_drops_rows(temp_db):
    """It is destructive by design, and the seed warns before calling it."""
    from api.models.tables import Employee

    db_module.init_db()
    session = db_module.get_sessionmaker()()
    session.add(Employee(username="u", full_name="U", password_hash="x", location="HQ"))
    session.commit()
    session.close()

    db_module.recreate_schema()

    session = db_module.get_sessionmaker()()
    assert session.query(Employee).count() == 0
    session.close()


# --------------------------------------------------------------------------
# The report itself
# --------------------------------------------------------------------------


def test_doctor_reports_schema_failure_before_querying(temp_db):
    """A stale schema must be named, not buried under the traceback it causes."""
    from api.doctor import FAIL, Report, check_schema

    db_module.init_db()
    with db_module.get_engine().begin() as connection:
        connection.execute(text("ALTER TABLE employees DROP COLUMN location"))

    report = Report()
    assert check_schema(report) is False
    check = report.checks[-1]
    assert check.status == FAIL
    assert "employees.location" in check.detail
    assert any("--reset" in line for line in check.fix)
    assert report.failed


def test_doctor_passes_a_healthy_schema(temp_db):
    from api.doctor import OK, Report, check_schema

    db_module.init_db()
    report = Report()
    assert check_schema(report) is True
    assert report.checks[-1].status == OK
    assert not report.failed


# --------------------------------------------------------------------------
# The bug that started this
#
# `cohort.npz` and `calibrator.json` had been produced by one checkpoint while
# the service loaded another. The only symptom was that two thirds of skilled
# forgeries scored 99.5 out of 100 — indistinguishable, on screen, from a
# system working perfectly. Nothing detected it because `model_version` was
# `architecture@git_commit`, every checkpoint reported `signet@unknown`, and
# every staleness check compared two identical meaningless strings.
# --------------------------------------------------------------------------


def test_model_version_identifies_weights_not_the_commit():
    """Two checkpoints from the same commit must not report the same version."""
    from ml.embed.provenance import weights_id

    import torch

    from ml.embed.models import build_model

    torch.manual_seed(0)
    first = build_model("signet").state_dict()
    torch.manual_seed(1)
    second = build_model("signet").state_dict()

    assert weights_id(first) != weights_id(second)
    # And neither is the string that used to stand in for identity.
    assert "unknown" not in weights_id(first)


def test_doctor_flags_a_cohort_from_another_run(tmp_path, monkeypatch):
    """The check `api/doctor.py`'s docstring promised and did not implement."""
    import numpy as np
    import torch

    from api import doctor as doctor_module
    from api.doctor import FAIL, Report, check_artifacts
    from ml.embed.models import build_model
    from ml.embed.provenance import weights_id
    from ml.scoring.znorm import CohortNormalizer

    state = build_model("signet").state_dict()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": state,
            "architecture": "signet",
            "embedding_dim": 512,
            "weights_id": weights_id(state),
            "config": {},
            "provenance": {},
        },
        checkpoint,
    )

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(120, 512))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    CohortNormalizer(vectors, [f"s{i}" for i in range(120)]).save(
        tmp_path / "cohort.npz", weights_id="0000000000000000"
    )

    monkeypatch.setattr(
        doctor_module,
        "get_settings",
        lambda: Settings(
            checkpoint_path=checkpoint,
            cohort_path=tmp_path / "cohort.npz",
            calibrator_path=tmp_path / "absent.json",
            dev_key_path=tmp_path / "k",
        ),
    )

    report = Report()
    check_artifacts(report)

    failure = next(c for c in report.checks if c.status == FAIL)
    assert "cohort" in failure.name
    assert "0000000000000000" in failure.detail
    assert any("benchmark" in line for line in failure.fix)
