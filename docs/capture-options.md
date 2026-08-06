# Capture Options — paper vs signature pad

Requested as an investigation, not a decision. The purpose is to make the
accuracy ceiling of each route explicit *before* committing, because the
choice of capture channel bounds achievable accuracy more than any modelling
decision does.

## The short version

Offline verification — a photograph or scan of a signature on paper — is
working from the *outcome* of the signing. Online capture records the *act*:
pen pressure, velocity, stroke order, pen lifts. A skilled forger can reproduce
the shape of a signature given enough practice. Reproducing the motor programme
that produced it, while being watched by a 200 Hz sensor, is a different problem
entirely.

That difference is why online capture consistently outperforms offline, and no
amount of model work closes the gap.

## Measured accuracy

| Route | Published EER vs skilled forgeries | Source |
|---|---|---|
| Online (pad) | **3.33%** best system, easiest task | ICDAR SVC 2021 winner (DLVC-Lab) |
| Online (pad) | 6.04% – 7.41% on the harder tasks | same competition |
| Offline (paper), clean benchmark | 2 – 5% | SigNet / HTCSigNet on CEDAR, GPDS |
| Offline (paper), realistic real-world capture | expect worse; measure on your own data | — |

Two cautions on reading that table:

1. **The offline benchmark figures are not real-world figures.** They come from
   curated datasets: isolated signatures, clean white background, one scanner.
   Real captures involve printed rules through the strokes, stamps, phone
   photographs at an angle, and specimens signed years apart. This is why the
   project's own sealed test set is the only number that should be quoted.
2. **The online figures are competition figures too**, and the harder SVC tasks
   land at 6–7%. The honest claim is not "pads give you 1%" — it is that pads
   move the achievable range down by roughly a factor of two to three, and add
   a signal offline verification simply does not have.

A commonly repeated claim puts online capture at "1–3% EER". That is
optimistic: 3.3% is the best competition result and the harder tasks are
worse.

## Hardware cost

Mid-tier LCD pads with the sampling rate needed for dynamic verification
(≥500 Hz, 4–5" screen) run **€170–€320** per unit. Representative models used in
signature capture: Wacom STU-430 / STU-540, signotec Sigma, Topaz T-LBK462.

- Wireless or combo units: €240–€410.
- Full-page LCD models: €850+ — only needed if customers must sign a whole
  document on-screen, which is a different requirement.

Budgeting note: mid-tier units tend to win on three-year total cost, because
the cheapest pads generate rework and integration time that exceeds the saving.

To cost a rollout you need: the number of capture points, expected spares
ratio, and whether operator workstations have spare USB and an endpoint policy
that permits a new HID device.

## What pads do **not** solve

This is the part most easily missed, and it is decisive for sequencing:

- **Historical specimens stay on paper.** Every signature currently on file was
  captured on paper and has no dynamic data. A pad capture cannot be verified
  against a paper specimen using dynamic features — only the static shape is
  comparable. So the offline pipeline is still needed for every existing
  customer.
- **A dual-stack period is unavoidable**, and it is long: it ends only when
  every signer has been re-enrolled on a pad, which at any scale means years,
  and never completes for dormant records.
- **Not every interaction suits a pad.** Cheques, mailed instructions, and
  third-party documents arrive on paper regardless.

## Recommended shape of a decision

Rather than an organisation-wide either/or:

1. **Keep and improve the offline pipeline.** It is required for the existing
   specimen base no matter what else is decided, and it is already built.
2. **Pilot pads at a small number of capture points**, for new enrolments and
   for transactions above a value threshold. This is where the accuracy gain is
   worth the hardware, and it starts building a dynamic specimen base.
3. **Re-enrol opportunistically.** Every time an existing signer passes a
   piloted capture point, take a pad specimen alongside the paper one. The dynamic
   specimen base then grows without a dedicated campaign.
4. **Revisit a wider rollout** once the pilot has produced a measured EER on
   your own signers — the same discipline applied to the offline model.

## What this means for the accuracy target

A goal of ">=95% accuracy" is common. Stated properly as TAR at a low FAR
against *skilled* forgeries:

- Offline alone, on realistic captures, is unlikely to reach it.
- Online capture reaches materially better numbers but the published evidence
  puts the best systems at 3–7% EER, not near-zero.

If a hard guarantee is needed at the point of capture, the realistic
architecture is: **pads for new and high-value flows, offline as the advisory
assist everywhere else, and a human decision on top of both** — which is what
the system already implements.
