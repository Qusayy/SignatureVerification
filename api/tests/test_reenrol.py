"""Tests for re-enrolment after a model or preprocessing change."""

from __future__ import annotations

import numpy as np

from api.tests.conftest import as_upload, requires_model

pytestmark = requires_model


def _enrolled_customer(client, auth, signatures, number="C900"):
    assert client.post(
        "/api/customers", headers=auth, json={"customer_number": number, "full_name": "Re-enrol"}
    ).status_code == 201
    files = [("files", as_upload(img, f"r{i}.png")) for i, img in enumerate(signatures[:3])]
    assert client.post(
        f"/api/customers/{number}/references", headers=auth, files=files
    ).status_code == 201
    return number


def test_check_reports_a_clean_database_as_current(client, auth, signatures):
    from api.db import get_sessionmaker
    from api.reenrol import check
    from api.services.inference import get_service

    _enrolled_customer(client, auth, signatures["genuine"])

    session = get_sessionmaker()()
    try:
        report = check(session, get_service())
        assert report["stale_enrolments"] == 0
        assert report["customers_with_cached_enrolment"] >= 1
    finally:
        session.close()


def test_check_detects_a_stale_model_version(client, auth, signatures):
    from api.db import get_sessionmaker
    from api.models.tables import CustomerEnrolment
    from api.reenrol import check
    from api.services.inference import get_service

    _enrolled_customer(client, auth, signatures["genuine"])

    session = get_sessionmaker()()
    try:
        enrolment = session.query(CustomerEnrolment).first()
        enrolment.model_version = "signet@deadbeef"  # pretend an older model wrote it
        session.commit()

        report = check(session, get_service())
        assert report["stale_enrolments"] == 1
        assert "signet@deadbeef" in report["stale_versions"]
    finally:
        session.close()


def test_reenrol_rewrites_embeddings_and_clears_staleness(client, auth, signatures):
    from api.db import get_sessionmaker
    from api.models.tables import CustomerEnrolment, ReferenceSignature
    from api.reenrol import check, reenrol
    from api.services.inference import get_service
    from api.services.storage import get_store

    _enrolled_customer(client, auth, signatures["genuine"])

    session = get_sessionmaker()()
    try:
        # Corrupt the stored embeddings, as a stale model effectively does.
        for reference in session.query(ReferenceSignature).all():
            reference.embedding = [0.0] * len(reference.embedding)
        session.query(CustomerEnrolment).first().model_version = "signet@stale"
        session.commit()

        result = reenrol(session, get_service(), get_store(), dry_run=False)
        session.commit()

        assert result["references_re_embedded"] == 3
        assert result["customers_updated"] == 1
        assert not result["failures"]

        # Embeddings are real vectors again, recomputed from the stored images.
        for reference in session.query(ReferenceSignature).all():
            vector = np.asarray(reference.embedding)
            assert np.linalg.norm(vector) > 0.5

        assert check(session, get_service())["stale_enrolments"] == 0
    finally:
        session.close()


def test_dry_run_changes_nothing(client, auth, signatures):
    from api.db import get_sessionmaker
    from api.models.tables import ReferenceSignature
    from api.reenrol import reenrol
    from api.services.inference import get_service
    from api.services.storage import get_store

    _enrolled_customer(client, auth, signatures["genuine"])

    session = get_sessionmaker()()
    try:
        before = [list(r.embedding) for r in session.query(ReferenceSignature).all()]
        result = reenrol(session, get_service(), get_store(), dry_run=True)
        session.rollback()

        after = [list(r.embedding) for r in session.query(ReferenceSignature).all()]
        assert result["dry_run"] is True
        assert result["references_re_embedded"] == 3
        assert before == after
    finally:
        session.close()


def test_reenrolled_scores_stay_sane(client, auth, signatures):
    """The point of re-enrolment: a genuine signature still scores like one."""
    from api.db import get_sessionmaker
    from api.reenrol import reenrol
    from api.services.inference import get_service
    from api.services.storage import get_store

    number = _enrolled_customer(client, auth, signatures["genuine"])

    def score(image):
        response = client.post(
            "/api/verify",
            headers=auth,
            data={"customer_number": number, "is_full_page": "false"},
            files={"file": as_upload(image)},
        )
        assert response.status_code == 200, response.text
        return response.json()["comparison"]["similarity"]

    before = score(signatures["genuine"][3])

    session = get_sessionmaker()()
    try:
        reenrol(session, get_service(), get_store(), dry_run=False)
        session.commit()
    finally:
        session.close()

    after = score(signatures["genuine"][3])
    # Re-embedding with the same model must be a no-op within numerical noise.
    assert abs(after - before) < 0.02
    # And a genuine signature must still beat a different writer's.
    assert after > score(signatures["other_writer"])
