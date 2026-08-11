"""End-to-end API tests."""

from __future__ import annotations

import numpy as np
import pytest

from api.tests.conftest import as_upload, requires_model

pytestmark = requires_model


# --------------------------------------------------------------------------
# Health and auth
# --------------------------------------------------------------------------


def test_health_reports_model_and_advisory_status(client):
    body = client.get("/api/health").json()
    assert body["model_loaded"] is True
    assert body["advisory_only"] is True
    assert isinstance(body["warnings"], list)


def test_endpoints_require_authentication(client):
    assert client.get("/api/customers").status_code == 401
    assert client.get("/api/audit/events").status_code == 401
    assert client.post("/api/verify").status_code == 401
    assert client.get("/api/images/anything").status_code == 401


def test_bad_credentials_are_rejected(client, employee):
    response = client.post(
        "/api/auth/token", data={"username": employee["username"], "password": "wrong"}
    )
    assert response.status_code == 401


def test_token_identifies_the_employee(client, auth, employee):
    body = client.get("/api/auth/me", headers=auth).json()
    assert body["username"] == employee["username"]


# --------------------------------------------------------------------------
# Enrolment
# --------------------------------------------------------------------------


def _create_customer(client, auth, number="C001", script="latin"):
    response = client.post(
        "/api/customers",
        headers=auth,
        json={"customer_number": number, "full_name": "Test Customer", "script": script},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _enrol(client, auth, number, images):
    files = [("files", as_upload(img, f"ref{i}.png")) for i, img in enumerate(images)]
    response = client.post(f"/api/customers/{number}/references", headers=auth, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_duplicate_customer_number_is_rejected(client, auth):
    _create_customer(client, auth)
    response = client.post(
        "/api/customers", headers=auth, json={"customer_number": "C001", "full_name": "Dup"}
    )
    assert response.status_code == 409


def test_enrolment_stores_specimens_and_embeddings(client, auth, signatures):
    _create_customer(client, auth)
    body = _enrol(client, auth, "C001", signatures["genuine"][:3])
    assert body["n_references"] == 3
    assert len(body["references"]) == 3
    assert all(r["canvas_url"].startswith("/api/images/") for r in body["references"])


def test_blank_specimen_is_refused_at_enrolment(client, auth):
    _create_customer(client, auth)
    blank = np.full((200, 400), 255, dtype=np.uint8)
    response = client.post(
        "/api/customers/C001/references", headers=auth, files=[("files", as_upload(blank))]
    )
    assert response.status_code == 422
    assert "ink" in response.json()["detail"].lower()


def test_deleting_a_specimen_updates_the_count(client, auth, signatures):
    _create_customer(client, auth)
    body = _enrol(client, auth, "C001", signatures["genuine"][:3])
    reference_id = body["references"][0]["id"]
    response = client.delete(f"/api/customers/C001/references/{reference_id}", headers=auth)
    assert response.status_code == 200
    assert response.json()["n_references"] == 2


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def test_verification_without_specimens_is_refused(client, auth, signatures):
    _create_customer(client, auth)
    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": as_upload(signatures["genuine"][0])},
    )
    assert response.status_code == 409
    assert "no specimen" in response.json()["detail"].lower()


def test_verification_returns_an_advisory_result(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": as_upload(signatures["genuine"][3])},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["advisory_only"] is True
    assert 0 <= body["score"] <= 100
    assert body["band"] in {"green", "amber", "red"}
    assert body["comparison"]["n_references"] == 3
    assert len(body["comparison"]["per_reference"]) == 3
    assert body["reason"]
    assert body["crop_url"] and body["overlay_url"]
    # No field anywhere claims the system reached a verdict.
    assert "decision" not in body
    assert "approved" not in body


def test_genuine_signature_scores_above_a_different_writer(client, auth, signatures):
    """The core sanity check: the model must at least separate different people."""
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    def score(image):
        response = client.post(
            "/api/verify",
            headers=auth,
            data={"customer_number": "C001", "is_full_page": "false"},
            files={"file": as_upload(image)},
        )
        assert response.status_code == 200, response.text
        return response.json()["comparison"]["raw"]

    assert score(signatures["genuine"][3]) > score(signatures["other_writer"])


def test_blank_capture_is_refused_rather_than_scored(client, auth, signatures):
    """A blank crop must produce an error, never a confident-looking low score."""
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    blank = np.full((300, 600), 250, dtype=np.uint8)
    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": as_upload(blank)},
    )
    assert response.status_code == 422
    assert "rescan" in response.json()["detail"].lower()


def test_single_specimen_lowers_stated_confidence(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:1])

    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": as_upload(signatures["genuine"][2])},
    )
    body = response.json()
    assert body["comparison"]["single_reference"] is True
    assert "single_reference_lower_confidence" in body["warnings"]


def test_full_page_detection_finds_the_signature(client, auth, signatures):
    from ml.data.synth import render_on_form

    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    page, _bbox = render_on_form(signatures["genuine"][3], np.random.default_rng(9))
    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "true"},
        files={"file": as_upload(page, "form.png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["detection"] is not None
    assert body["detection"]["method"] == "heuristic"
    assert body["detection"]["bbox"]["width"] > 0


def test_employee_supplied_bbox_overrides_detection(client, auth, signatures):
    from ml.data.synth import render_on_form

    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    page, bbox = render_on_form(signatures["genuine"][3], np.random.default_rng(11))
    x, y, w, h = bbox
    response = client.post(
        "/api/verify",
        headers=auth,
        data={
            "customer_number": "C001",
            "is_full_page": "true",
            "bbox_x": str(x),
            "bbox_y": str(y),
            "bbox_width": str(w),
            "bbox_height": str(h),
        },
        files={"file": as_upload(page, "form.png")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["detection"]["method"] == "employee"


def test_unknown_customer_is_a_404(client, auth, signatures):
    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "NOPE", "is_full_page": "false"},
        files={"file": as_upload(signatures["genuine"][0])},
    )
    assert response.status_code == 404


def test_non_image_upload_is_rejected(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:2])
    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": ("notes.png", b"this is not an image", "image/png")},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# Decisions and audit
# --------------------------------------------------------------------------


def _verify(client, auth, image, number="C001"):
    response = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": number, "is_full_page": "false"},
        files={"file": as_upload(image)},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_decision_is_recorded_and_appears_in_the_audit_log(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])
    result = _verify(client, auth, signatures["genuine"][3])

    response = client.post(
        f"/api/verify/{result['event_id']}/decision",
        headers=auth,
        json={"decision": "accept", "note": "Matches the card", "seconds_to_decide": 8},
    )
    assert response.status_code == 201, response.text

    events = client.get("/api/audit/events", headers=auth).json()
    assert len(events) == 1
    assert events[0]["decision"] == "accept"
    assert events[0]["decision_note"] == "Matches the card"


def test_audit_trail_is_append_only(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])
    result = _verify(client, auth, signatures["genuine"][3])

    first = client.post(
        f"/api/verify/{result['event_id']}/decision", headers=auth, json={"decision": "accept"}
    )
    assert first.status_code == 201

    second = client.post(
        f"/api/verify/{result['event_id']}/decision", headers=auth, json={"decision": "reject"}
    )
    assert second.status_code == 409
    assert "append-only" in second.json()["detail"]


def test_agreement_summary_excludes_amber(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])
    result = _verify(client, auth, signatures["genuine"][3])
    client.post(
        f"/api/verify/{result['event_id']}/decision", headers=auth, json={"decision": "accept"}
    )

    summary = client.get("/api/audit/agreement", headers=auth).json()
    assert summary["total_decisions"] == 1
    assert "amber is inconclusive" in summary["note"]
    if result["band"] == "amber":
        assert summary["agreements"] == 0 and summary["disagreements"] == 0
    else:
        assert summary["agreements"] + summary["disagreements"] == 1


def test_decision_on_an_unknown_event_is_a_404(client, auth):
    response = client.post(
        "/api/verify/does-not-exist/decision", headers=auth, json={"decision": "accept"}
    )
    assert response.status_code == 404


def test_verification_is_durable_before_the_response_is_returned(client, auth, signatures):
    """The event must be committed by the endpoint, not by dependency teardown.

    FastAPI runs dependency cleanup *after* the response has been sent, so an
    endpoint relying on the teardown commit hands the client an ``event_id``
    that is not yet durable. A operator who clicks Accept quickly then gets a 404
    — intermittently, and only over real HTTP.

    TestClient completes teardown before returning, so it cannot reproduce the
    race. Instead the teardown commit is disabled here, which asserts the
    endpoint made the row durable on its own.
    """
    from api.db import get_session, get_sessionmaker
    from api.main import app
    from api.models.tables import VerificationEvent

    def session_without_teardown_commit():
        session = get_sessionmaker()()
        try:
            yield session  # deliberately no commit here
        finally:
            session.close()

    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    app.dependency_overrides[get_session] = session_without_teardown_commit
    try:
        result = _verify(client, auth, signatures["genuine"][3])

        # A brand-new session sees the row only if it was actually committed.
        probe = get_sessionmaker()()
        try:
            assert probe.get(VerificationEvent, result["event_id"]) is not None, (
                "verification event was not committed before the response was returned"
            )
        finally:
            probe.close()

        # And the follow-up decision therefore succeeds.
        decision = client.post(
            f"/api/verify/{result['event_id']}/decision",
            headers=auth,
            json={"decision": "accept"},
        )
        assert decision.status_code == 201, decision.text
    finally:
        app.dependency_overrides.pop(get_session, None)


# --------------------------------------------------------------------------
# Image serving
# --------------------------------------------------------------------------


def test_stored_images_are_served_but_never_cached(client, auth, signatures):
    _create_customer(client, auth)
    body = _enrol(client, auth, "C001", signatures["genuine"][:1])
    url = body["references"][0]["canvas_url"]

    response = client.get(url, headers=auth)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "no-store" in response.headers["cache-control"]


def test_image_paths_cannot_escape_the_storage_root(client, auth):
    response = client.get("/api/images/../../../../etc/passwd", headers=auth)
    assert response.status_code in {400, 404}


@pytest.mark.parametrize("missing", ["images/nope.png.enc", "canvases/absent.png.enc"])
def test_missing_images_are_404(client, auth, missing):
    assert client.get(f"/api/images/{missing}", headers=auth).status_code == 404


# --------------------------------------------------------------------------
# Pipeline trace
# --------------------------------------------------------------------------


def test_verification_returns_the_pipeline_trace(client, auth, signatures):
    """The explanation must be the real computation, not a reconstruction."""
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    body = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false", "explain": "true"},
        files={"file": as_upload(signatures["genuine"][3])},
    ).json()

    keys = [stage["key"] for stage in body["stages"]]
    # The chain the operator is shown, in the order it actually runs.
    assert keys == [
        "capture",
        "grayscale",
        "illumination",
        "binarised",
        "lines_removed",
        "denoised",
        "normalised",
        "model_input",
        "embedding",
        "compare",
        "cohort",
        "calibration",
        "overlay",
    ]
    assert all(stage["title"] and stage["caption"] for stage in body["stages"])


def test_trace_is_opt_out(client, auth, signatures):
    """A caller that only wants the number should not pay for a dozen PNGs."""
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    body = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false", "explain": "false"},
        files={"file": as_upload(signatures["genuine"][3])},
    ).json()

    assert body["stages"] == []


def test_trace_does_not_change_the_score(client, auth, signatures):
    """Tracing is observational. If it moved the score it would be a defect."""
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    def score(explain: str) -> float:
        return client.post(
            "/api/verify",
            headers=auth,
            data={"customer_number": "C001", "is_full_page": "false", "explain": explain},
            files={"file": as_upload(signatures["genuine"][3])},
        ).json()["score"]

    assert score("true") == score("false")


def test_trace_images_are_served_through_the_encrypted_store(client, auth, signatures):
    """Stage images are derived from a biometric image and are just as
    identifying as the original, so they get the same protection."""
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    body = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false", "explain": "true"},
        files={"file": as_upload(signatures["genuine"][3])},
    ).json()

    urls = [stage["image_url"] for stage in body["stages"] if stage["image_url"]]
    assert urls, "no stage produced an image"

    for url in urls:
        assert url.startswith("/api/images/stages/")
        assert client.get(url).status_code == 401  # no token, no image

        response = client.get(url, headers=auth)
        assert response.status_code == 200, url
        assert response.headers["content-type"] == "image/png"
        assert "no-store" in response.headers["cache-control"]


def test_scoring_stages_carry_the_numbers_behind_the_result(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    body = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false", "explain": "true"},
        files={"file": as_upload(signatures["genuine"][3])},
    ).json()
    stages = {stage["key"]: stage for stage in body["stages"]}

    compare = stages["compare"]
    assert compare["kind"] == "compare"
    assert compare["image_url"] is None
    assert len(compare["metrics"]["per_reference"]) == 3

    calibration = stages["calibration"]
    assert calibration["kind"] == "score"
    # The displayed score must be the score, not a separately rounded one.
    assert calibration["metrics"]["score"] == round(body["score"], 1)
    assert calibration["metrics"]["band"] == body["band"]


def test_comparison_reports_the_baseline_the_score_is_measured_against(
    client, auth, signatures
):
    """A client cannot explain the score without it.

    `raw` is the combined similarity minus the customer's own specimen
    agreement, so an 89% match against specimens that agree at 88.5% scores
    near the middle of the range. Reporting only the similarities makes that
    look like a defect.
    """
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:3])

    body = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": as_upload(signatures["genuine"][3])},
    ).json()

    comparison = body["comparison"]
    assert comparison["writer_normalised"] is True
    assert 0.0 < comparison["intra_reference_mean"] <= 1.0
    # raw is the margin, not the similarity — the arithmetic must reconcile.
    combined = 0.5 * comparison["max_similarity"] + 0.5 * comparison["mean_similarity"]
    assert comparison["raw"] == pytest.approx(
        combined - comparison["intra_reference_mean"], abs=1e-4
    )


def test_single_specimen_reports_no_baseline(client, auth, signatures):
    _create_customer(client, auth)
    _enrol(client, auth, "C001", signatures["genuine"][:1])

    body = client.post(
        "/api/verify",
        headers=auth,
        data={"customer_number": "C001", "is_full_page": "false"},
        files={"file": as_upload(signatures["genuine"][3])},
    ).json()

    assert body["comparison"]["writer_normalised"] is False
    assert body["comparison"]["intra_reference_mean"] == 0.0
    assert "score_not_writer_normalised" in body["warnings"]
