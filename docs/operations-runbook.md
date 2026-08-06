# Operations Runbook

## Before the first production deployment

Work through this list. Each item exists because skipping it causes a specific,
known failure.

### Secrets

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"          # SV_JWT_SECRET
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # SV_IMAGE_ENCRYPTION_KEY
```

`SV_IMAGE_ENCRYPTION_KEY` must be **stable and backed up**. It encrypts every
stored signature image. Lose it and every enrolled specimen becomes
unreadable; rotate it without re-encrypting and the same thing happens. Keep it
in a secret manager, not in `.env`, and not in git.

`SV_JWT_SECRET` must be at least 32 bytes — HS256 with a shorter key is weaker
than it appears. `/api/health` reports it if this is wrong.

### Environment

| Variable | Production value | Why |
|---|---|---|
| `SV_ENVIRONMENT` | `production` | Turns startup warnings into logged errors |
| `SV_DATABASE_URL` | `postgresql+psycopg://…` | SQLite is a single-writer file; production has concurrent operators |
| `SV_STORAGE_ENDPOINT` | your object store | Local files do not survive a container restart |
| `SV_REQUIRE_DEPLOYABLE_CHECKPOINT` | `true` | Refuses to serve a research-licensed (Track A) model |
| `SV_JWT_SECRET`, `SV_IMAGE_ENCRYPTION_KEY` | from the secret manager | See above |

`GET /api/health` lists every remaining misconfiguration. It should return an
empty `warnings` array in production.

### Model artifacts

Three files must be present and consistent with each other:

| File | Produced by | Consequence if missing |
|---|---|---|
| `artifacts/<arch>_track_b.pt` | `ml/embed/train.py` | Service starts degraded; verification returns 503 |
| `artifacts/cohort.npz` | `ml/eval/benchmark.py` | Scores are not comparable across customers; responses flagged `no_cohort_normalisation` |
| `artifacts/calibrator.json` | `ml/eval/benchmark.py` | Scores are placeholders; the UI shows "uncalibrated" and hides the number |

Gate the checkpoint before deploying:

```bash
python -m ml.embed.provenance artifacts/signet_track_b.pt --gate
```

Non-zero exit means the weight was trained on non-commercial data or from a
licence-encumbered initialisation. Do not deploy it. See `docs/licensing.md`.

### Re-enrolment after a model change

**Embeddings are model-specific.** Replacing the checkpoint invalidates every
stored `ReferenceSignature.embedding` and every cached `CustomerEnrolment`.
Scores computed against stale embeddings are meaningless but look completely
normal, which makes this the most dangerous operation in the system.

**Preprocessing changes carry exactly the same hazard**, because they change
the canvases the embeddings were computed from. Treat a preprocessing change as
a model change.

After any change to either:

```bash
# 1. Regenerate the cohort and calibrator for the new model
python -m ml.eval.benchmark --checkpoint artifacts/<new>.pt --split test

# 2. Point the service at the new checkpoint, then re-embed every specimen
python -m api.reenrol --check     # what is stale?
python -m api.reenrol --apply     # re-embed and commit

# 3. Confirm nothing is left behind — exits non-zero if anything is stale
python -m api.reenrol --check
```

Take the service out of rotation for step 2, and compare a sample of
before/after scores before returning it.

`CustomerEnrolment.model_version` records which model produced the cached
statistics. `--check` exits non-zero when any row disagrees with the loaded
model, so it can gate a deployment.

## Daily operation

### Health

`GET /api/health` — `status: ok` requires a loaded model. `degraded` means
verification is failing; check `error`.

### What to watch

| Signal | Where | Why it matters |
|---|---|---|
| Amber rate | `/api/audit/events` | Rising amber means the model is losing confidence — check for a scanner or lighting change before blaming the model |
| Disagreement rate | `/api/audit/agreement` | The pilot's key metric. A rise means the model and its users have diverged |
| `suspected_copy` count | `/api/audit/events` | Each one is a potential fraud attempt or a process problem (staff photocopying specimens) |
| `single_reference_lower_confidence` rate | verification warnings | Quantifies how much a re-enrolment programme would buy |
| Verification latency | `latency_ms` on each event | Target under 3s; the customer is at the desk |

### Common problems

**"Could not decrypt … the image encryption key has changed"**
`SV_IMAGE_ENCRYPTION_KEY` differs from the one used at enrolment. Restore the
original key. Images cannot be recovered without it.

**Every verification returns amber**
Usually the calibrator, not the model. Check `calibrated` on `/api/health`. An
uncalibrated service maps scores through a placeholder curve. If it *is*
calibrated, the model genuinely cannot separate the populations — check the
accuracy report and whether capture conditions changed.

**"No signature region found on this page"**
The heuristic detector declined rather than guessing. The employee can draw the
box manually in the UI, which always overrides detection. Frequent occurrences
mean the form layout changed, or a trained detector is overdue.

**Accuracy is poor and you do not know why**

Run the diagnostic before touching the model:

```bash
python -m ml.eval.diagnostics --checkpoint artifacts/<model>.pt
```

It reports whether the model survives the same signature being rescaled, and
how much of its score is explained by size alone. A model that scores well on
aggregate metrics can still be matching the wrong thing — that is exactly how
the first version of this system shipped with a defect where a genuine
signature written 20% smaller than the stored specimen scored below a skilled
forgery.

**Scores dropped across the board after a preprocessing change**

Preprocessing changes invalidate stored embeddings just as surely as a model
change does, and the failure looks identical: normal-looking scores that are
quietly wrong. Follow the same re-enrolment procedure as for a model change.

**Scores dropped across the board after a scanner change**
Expected. The model was trained on the old capture channel. Collect samples
through the new scanner and retrain; do not adjust thresholds to compensate,
which trades a measurable problem for an invisible one.

## Incident: suspected fraud

When `suspected_copy` fires, the captured signature is near-pixel-identical to
a stored specimen. A genuine signature never is.

1. Do not treat the high score as a strong match — it is the opposite.
2. Retrieve the event from `/api/audit/events` and pull the stored page image.
3. Escalate per your fraud process.
4. Keep the event; it is evidence and the audit trail is append-only.

## Data protection

- Signature images are biometric personal data. Encrypted at rest, served only
  to authenticated employees, and never cached (`Cache-Control: no-store`).
- `data/` and `artifacts/` must never be committed to git.
- Define and enforce a retention period for `VerificationEvent` page images.
  The scores and decisions are small and worth keeping; the page scans are
  large and the most sensitive thing in the system.
- Every verification and decision is attributed to a named employee. Removing
  an employee record breaks that attribution — deactivate (`is_active = false`)
  rather than delete.

## What this system does not do

It does not accept or reject signatures. It scores and explains; the employee
decides. Any future move toward automated accept/reject is a separate change
requiring risk and compliance sign-off, and should be based on the
disagreement data the pilot produces — not on the accuracy report alone.
