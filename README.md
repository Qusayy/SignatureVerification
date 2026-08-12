# Signature Verification

Offline handwritten signature verification, built to run entirely on your own
infrastructure.

An operator captures a signed document, the system locates and crops the
signature, compares it against the signer's stored specimens, and returns a
calibrated confidence score with a visual explanation.

**The system is advisory. It never accepts or rejects a signature.** A human
always makes the final call, and every decision is recorded.

---

## Why this exists

Most open signature verification code either (a) reports benchmark accuracy
that does not survive contact with real captures, or (b) is licensed such that
it cannot be deployed commercially. This project takes both problems seriously:

- **Honest metrics.** EER against *skilled* forgeries, TAR at fixed FAR, split
  by script, with explicit warnings when the test set is too small to support
  the number being quoted. No single blended "accuracy" figure.
- **Licence hygiene.** Every dataset and weight is tagged as research-only or
  commercially usable, and the training pipeline refuses to produce a
  production model from research-only inputs. See [`docs/licensing.md`](docs/licensing.md).
- **Diagnostics, not just scores.** An EER number tells you a model is bad; it
  does not tell you *what it latched onto*. `ml/eval/diagnostics.py` answers
  that, and caught a real defect in this repository (see below).

## Quick start

Runs end to end on a laptop with no infrastructure — SQLite and local encrypted
file storage — and with no real data.

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows
pip install -r requirements.txt

# 1. Secrets. Must be stable, or images written now cannot be read later.
python -c "from cryptography.fernet import Fernet; print('SV_IMAGE_ENCRYPTION_KEY=' + Fernet.generate_key().decode())" >> .env
python -c "import secrets; print('SV_JWT_SECRET=' + secrets.token_urlsafe(48))" >> .env

# 2. Synthetic corpus, so the stack runs before real data exists
python -m ml.data.synth --signers 120 --out data/synthetic

# 3. Manifest and signer-level splits (frozen and leakage-checked)
python -m ml.data.ingest --source synthetic --root data/synthetic
python -m ml.data.manifest split --test-frac 0.2 --val-frac 0.1
python -m ml.data.manifest verify --track b

# 4. Train, then evaluate. Evaluation fits the calibrator the API needs and
#    derives the band edges, so run it before starting the service.
python -m ml.embed.train --arch signet --epochs 30 --track b --lr 0.02 \
    --forgery-weight 0.5 --batches-per-epoch 52 --cache-dir data/cache
python -m ml.eval.benchmark --split test --by-script --cache-dir data/cache

# 5. Seed a demo operator and customers drawn from the *test* split —
#    signers the model has never seen.
python -m api.seed --customers 10 --references 1

# 6. Confirm everything lines up before starting the service
python -m api.doctor

# 7. Run it
python -m uvicorn api.main:app --port 8000            # terminal 1
npm --prefix web install && npm --prefix web run dev  # terminal 2 → :3000
```

Sign in as `demo` / `demo1234`. `data/demo_queries/<customer>/` holds
`genuine_*.png` and `forgery_*.png` for each seeded customer — drag them into
the verify screen to show both outcomes.

Containerised alternative to step 7: `docker compose up`.

### When something is broken

```bash
python -m api.doctor
```

Checks the secrets, the artifacts, the database schema, whether every stored
image still decrypts, and whether stored embeddings match the loaded model.
Every failure it reports comes with the command that fixes it.

Run it first whenever the interface returns a 500 with no obvious cause. The
faults it catches all look identical from the browser — a thumbnail that will
not load, a verification that will not run — and none of them are visible from
the error message alone.

**After retraining**, stored embeddings mean nothing under the new model, and
nothing about that failure is visible: scores keep arriving in the normal range
and are quietly wrong. Either re-embed in place or rebuild the demo:

```bash
python -m api.reenrol --check && python -m api.reenrol --apply   # keep the data
python -m api.seed --reset --customers 10 --references 1         # start over
```

## How it works

```
Captured document
   │
   ├─ Stage A: locate the signature region        ml/detector/
   │
   ├─ Preprocess: illumination, ruled-line removal, binarise,
   │              moment size-normalise, centre on canvas    ml/preprocess/
   │
   ├─ Stage B: embed (SigNet, CNN+ViT hybrid, or DINOv2/SigLIP + LoRA)   ml/embed/
   │
   ├─ Score: cosine to the stored specimen, isotonic
   │         calibration, copy-paste fraud check             ml/scoring/
   │
   └─ 0-100 confidence + band + difference overlay
```

| Path | Purpose |
|---|---|
| `ml/config.py` | Single source of truth for geometry, model, and scoring config |
| `ml/preprocess/` | Illumination, line removal, binarisation, deskew, size normalisation |
| `ml/data/` | Manifest, signer-level splits, leakage check, augmentation, synthetic corpus |
| `ml/detector/` | Stage A — locate the signature on a scanned document |
| `ml/embed/` | Stage B — SigNet baseline, CNN+ViT hybrid, DINOv2/SigLIP with LoRA |
| `ml/scoring/` | Comparison, calibration, band edges, explanations |
| `ml/eval/` | Metrics, benchmark harness, diagnostics, writer-count ablation |
| `api/` | FastAPI service: enrolment, verification, decision audit log |
| `web/` | React operator screen |
| `docs/` | Licensing policy, operations runbook, capture options |

## Documents

| Document | Read it when |
|---|---|
| [`docs/licensing.md`](docs/licensing.md) | **Before writing any code.** Which datasets and weights can legally back a production model |
| [`docs/demo-script.md`](docs/demo-script.md) | Demonstrating the system to the people who will use it |
| [`docs/operations-runbook.md`](docs/operations-runbook.md) | Deploying, or something is wrong in production |
| [`docs/capture-options.md`](docs/capture-options.md) | Choosing between paper capture and signature pads |
| `docs/accuracy-report.md` | Generated by `ml/eval/benchmark.py` — the only source of accuracy numbers |

## ⚠️ Before you ship anything

Read [`docs/licensing.md`](docs/licensing.md). Most publicly available
signature datasets and signature-specific pretrained weights are
**non-commercial**. This repository enforces a two-track split:

- **Track A** — research assets. Experiments and benchmarking only.
- **Track B** — data you own, plus permissively-licensed assets. The only track
  allowed to produce a shipped model.

Training refuses to write a Track B checkpoint from Track A data, and
`python -m ml.embed.provenance <checkpoint> --gate` exits non-zero on anything
that is not clear for deployment.

The important nuance: the rule is *not* "no pretrained weights". General vision
foundation models — DINOv2 and SigLIP (Apache-2.0), CLIP (MIT) — are
commercially usable and are supported first-class in `ml/embed/backbones.py`.
Only signature-*specific* pretrained weights are licence-poisoned.

## Diagnosing accuracy

```bash
python -m ml.eval.diagnostics --checkpoint artifacts/signet_track_b.pt
```

Reports whether the model survives the same signature being rescaled, and how
much of its score is explained by size alone.

This caught a real defect here. An early version preserved absolute signature
size deliberately, reasoning that size distinguishes writers. In practice the
same signature written 20% smaller scored **0.80** while a skilled forgery
scored **0.74** — a genuine signer writing slightly small was rejected, and no
aggregate metric revealed it. Moment-based size normalisation fixed it
(0.998 across a 0.6×–1.6× range) and the sweep is now a blocking test in
`ml/tests/test_scale_invariance.py`.

## Accuracy

A single "accuracy" number is meaningless for a verification system. The harness
reports EER against *skilled* forgeries, TAR at fixed FAR, and a per-script
breakdown, and refuses to emit a headline figure from a synthetic-only corpus.

**The deployed protocol is one stored specimen per customer.** Not a degraded
case — the design point, because the customer base is too large for a
specimen-collection programme. Everything in the scoring layer follows from it,
and the headline is measured at one specimen rather than at "all available",
which used to overstate the deployed system by 7.5 EER points.

| Specimens on file | EER | 95% CI |
|---|---|---|
| **1 (served)** | **27.08%** | 18.1–31.9 |
| 2 | 22.92% | 18.2–28.1 |
| all available | 20.14% | 16.7–24.3 |

### How a similarity becomes a score

Two steps, and nothing between them:

1. **Cosine to the nearest stored specimen.** Also what the interface shows as
   "Matched at 91.3%", so the chain from pixels to score is checkable by eye.
2. **An isotonic curve fitted on held-out signers, at the protocol that is
   served.** The calibrator records how many specimens it was fitted for and
   the service refuses a mismatch — a curve fitted on six specimens and applied
   to one produces plausible, wrong numbers, which is what every score before
   this rework was.

Two properties the curve is responsible for, each of which failed in production:

- **A continuum, not a staircase.** Isotonic produces flat blocks; keeping both
  ends of each made the shipped curve emit **eight** distinct scores. One knot
  per block, at its centroid, gives ~640.
- **A ceiling it has earned.** The top isotonic block reaches probability 1.0
  whenever the highest few comparisons happen to be genuine, which is why 99.5
  printed from a model with AUC 0.80. Laplace smoothing per block makes
  confidence a function of the evidence behind it.

### Bands come from operating points, not from a constant

The 0-100 score is P(genuine) under the *benchmark's* genuine/impostor mix
(~45% genuine). At a counter the forgery base rate is nearer 0.1%. So a fixed
edge at 75 has no operational meaning — change the forgeries per writer in the
validation split and the same signature lands in a different band.

FAR and FRR are conditional on class and do not move, so the edges are derived
from them on validation and stored in the calibrator:

- green admits at most **5%** of forgeries (`SCORING.green_max_far`)
- red rejects at most **5%** of genuine signatures (`SCORING.red_max_frr`)

Measured on the sealed test set at one specimen: **~24% of genuine traffic
clears green, ~60% lands amber, ~15% red.** Amber means the operator compares
manually, which is what they do for every signature today — so amber is not a
failure. Green is the only band that saves time and red the only one that raises
an alarm.

### Two numbers, always reported separately

Skilled-forgery EER (someone practising your signature) and random-impostor EER
(a stranger signing your name) differ by orders of magnitude — 27% versus 0.00%
on this corpus. Quoting the combined figure is the most common way signature
verification accuracy gets overstated.

Every EER carries a **writer-level bootstrap confidence interval**. An unpaired
difference below about 3 points is not evidence of anything.

### The service refuses rather than degrades

Artifacts are bound to weights and to a protocol. A calibrator that is missing,
from other weights, from an older schema, or fitted for a different specimen
count stops the service at startup with the command that fixes it. A customer
whose stored embeddings came from an older model gets a 409, not a score.

None of that ever reaches the counter as a withheld number: the operator has
already captured the signature and waited, and a dash is the least useful thing
a screen can show.

## Tests

```bash
pytest -m "not slow"         # full suite; `slow` downloads pretrained weights
npm --prefix web test
npm --prefix web run typecheck
```

API tests are genuinely end to end — real preprocessing, real model, real
scoring chain against a temporary database — because the wiring between those
pieces is where this kind of system actually breaks.

## Data protection

Signature images are biometric personal data. `data/` and `artifacts/` are
git-ignored and must never be committed. The API encrypts stored images at
rest, authenticates every request, sets `Cache-Control: no-store` on image
responses, and writes an append-only audit record for every verification and
every human decision.

## Licence

MIT — see [LICENSE](LICENSE). Note that this covers *this code*; the datasets
and pretrained weights it can consume have their own terms, which is the entire
subject of `docs/licensing.md`.
