# Demo Script

For showing the system to the people who will actually use it — the operators
who compare signatures by eye today. Roughly 15 minutes.

The goal is not to impress anyone with the model. It is to establish that the
system helps them and does not replace their judgement.

## Before you start

```bash
python -m uvicorn api.main:app --port 8000
npm --prefix web run dev
```

Check `GET /api/health` returns `model_loaded: true` and `calibrated: true`. If
`calibrated` is false the UI shows "uncalibrated" instead of a score — correct
behaviour, but a poor demo. Run `ml/eval/benchmark.py` first.

Have `data/demo_queries/` open in a file browser. Each customer folder holds
`genuine_*.png` and `forgery_*.png`.

## The opening line

> "This does not decide anything. It gives you a second opinion and a record of
> what you decided. You still make the call, exactly as you do now."

Say this first. Everyone in the room is wondering whether this is here to
replace them, and nothing else you show lands until that question is answered.

## 1. The current process, restated (1 min)

Ask someone to describe what they do today: look up the signer, find the
specimen, compare by eye. Agree that it works, and name the two things that are
genuinely hard:

- It is slow when there is a queue.
- Two colleagues can look at the same pair and reach different conclusions, and
  there is no record of why either decided as they did.

Those are the two problems this addresses. Not "people are bad at this".

## 2. A genuine signature (3 min)

Sign in as `demo`. Search `C1001`. The stored specimens appear immediately —
already a small win over pulling up the record manually.

Untick "This is a full form", upload `C1001/genuine_0.png`, click **Check
signature**.

Point at, in this order:

1. **The score and band** — one number, and a word.
2. **The difference overlay** — the important part. Blue is what was just
   written, orange is what is on file, grey is where they agree. *This is the
   digital version of holding two pages up to the light.*
3. **The per-specimen percentages** under the stored specimens: the system
   compared against all of them, not just one.
4. **"Advisory only — you decide"** under the gauge.

## 2a. Show the working (2 min)

Below the result, **How this result was reached** replays the verification one
step at a time — the capture, the region it found, the ink separated from the
paper, the size normalisation, the embedding, the comparison, the score. It
plays automatically; the filmstrip along the bottom jumps to any step.

This is the section that answers the question the room is actually holding:
*how do we know it isn't guessing?* Let it play once without narrating, then
stop on three panels:

1. **Signature located** — the box it drew. Everything outside it is discarded
   and never scored.
2. **Size and position normalised** — the same signature written larger or
   smaller lands in the same place. Worth stating plainly, because a customer
   signing bigger than usual is the obvious objection and this is the answer.
3. **Embedding** — the strip of colour. "This is what the system stores and
   compares. It is not a picture of the signature and it cannot be turned back
   into one."

Every panel is an image this verification genuinely produced, not an
illustration. If somebody asks whether it is a mock-up, that is the answer.

Two notes if asked:

- It adds roughly 50ms and a dozen small images per check. It can be turned off
  per request, and it is not on the path to the score.
- Those images are encrypted at rest and behind authentication, exactly like
  the signature itself.

## 3. A skilled forgery (3 min)

Upload `C1001/forgery_0.png` for the same customer.

Let the room look at the overlay before you say anything. On a forgery the blue
and orange separate visibly — that divergence is the thing to sell, more than
the number.

Be honest here. If the model is trained on synthetic data, these two cases may
score similarly. If so, say so plainly:

> "Right now the model is trained on synthetic signatures, so it is not yet
> good at this. The overlay is still useful today. The number becomes useful
> once we train it on our own signatures — which is the next phase, and it
> needs your help."

That admission buys more credibility than a rigged demo, and it sets up the ask
in section 6.

## 4. The decision, and the record (2 min)

Type a note, click **Accept** or **Reject**. Then open **History**.

- Every check is listed, with the score, the advice, and what was decided.
- Disagreements are highlighted.
- Nothing can be edited or deleted.

Frame this as protection, not surveillance:

> "If a signature is ever disputed, there is a record of exactly what was
> compared and what you decided. Today that conversation is one person's memory
> against another's."

## 5. The edge cases worth showing (3 min)

These build trust faster than the happy path.

**A signer with one specimen on file.** The system says so, in plain language,
and tells you confidence is lower. It does not pretend.

**A blank or failed scan.** Upload a blank image. The system refuses to score
it and asks for a rescan, rather than returning a confident-looking low number.

**A photocopy.** Upload one of the customer's own stored specimen images back as
the query. The system flags a suspected copy rather than reporting a perfect
match — because a genuine signature is never identical to the one on file.

**A full document.** Tick "This is a full form" and upload a page from
`data/synthetic/forms/`. The signature is located automatically; if it picks the
wrong area, the operator overrides it.

**The same signature, written larger.** Rescale one of the genuine images and
run it again. The score should barely move. This is worth showing because it is
the failure mode that nearly shipped — see `ml/eval/diagnostics.py`.

## 6. The ask (3 min)

This is the real purpose of the demo.

> "To make this accurate on our own signers — including non-Latin scripts — we
> need two things:
>
> 1. **Real signature samples**, captured on our own scanners, on our own forms.
> 2. **Practised imitations.** We need people to sit down and genuinely try to
>    copy a colleague's signature. That sounds strange, but a system that has
>    only ever seen careless fakes will not catch a careful one, and a careful
>    one is what actual fraud looks like.
>
> And as you use it, every accept and reject you record teaches it. The cases
> where you disagree with it are the most valuable of all."

## Questions you will get

**"Is this going to replace us?"**
No. It cannot accept or reject — that is built into the system, not a setting.
Automating any part of the decision would be a separate project requiring sign
off, and it would be based on the data from this pilot.

**"What if it is wrong?"**
Then you overrule it, and that disagreement is logged and reviewed.
Disagreements are the point, not a failure.

**"Does the signature data leave our systems?"**
No. Everything runs on our own servers. The service needs no internet access.

**"What about signers whose signature has changed over the years?"**
A real problem, and the system will flag those as inconclusive rather than
reject them. It is also an argument for refreshing old specimens, which the
pilot will quantify.

**"How accurate is it?"**
Give the real number from `docs/accuracy-report.md`, with its caveat, or say
"we do not know yet on our own signers, which is what the pilot measures". Do
not quote a vendor's marketing figure, and do not quote a benchmark number as
if it were ours.
