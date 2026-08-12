"""Comparing a query signature against a customer's stored specimens.

The live question is never "whose signature is this?" but "is this *this*
customer's signature?" — a 1:1 verification against a small reference set, not
a 1:N identification. Everything here is built for that.

Aggregating several references is not merely averaging. The two useful
statistics behave differently:

* ``max`` similarity — distance to the *nearest* specimen. Robust when a
  customer's stored specimens are inconsistent (signed years apart, different
  pens), because one good match is enough.
* ``mean`` similarity — agreement with the specimen set *as a whole*. Harder to
  fool, because a forgery that happens to resemble one specimen still has to
  resemble the others.

Using max alone rewards a lucky match; using mean alone punishes customers
whose specimen set is genuinely varied. The combination is what gets used, and
with a single stored reference the two collapse to the same number — which is
precisely why single-reference customers score worse and why the reference
count is a Phase 0 discovery question.

**What the similarity is measured against matters more than how it is pooled.**
A raw similarity of 0.82 says nothing on its own: for a customer who signs very
consistently it is poor, and for one whose own specimens only agree at 0.75 it
is excellent. Subtracting the mean pairwise similarity *among the customer's own
specimens* turns the score into "is this query as close to the specimens as the
specimens are to each other?", which is exactly the question a skilled forgery
is built to defeat. Measured on the sealed test set this is worth roughly 4 EER
points over the raw similarity, and ~15 over the cohort-normalised score that
preceded it — with no retraining. See :mod:`ml.scoring.znorm` for why the
cohort approach it replaces was a net loss here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.config import SCORING, ScoringConfig

__all__ = [
    "ComparisonScore",
    "compare_to_references",
    "intra_reference_mean",
    "l2_normalize",
]

# Weight on the nearest-reference term. The remainder goes to the mean.
DEFAULT_MAX_WEIGHT = 0.5


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Project embeddings onto the unit sphere, where cosine == dot product."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@dataclass
class ComparisonScore:
    """Raw (uncalibrated, unnormalised) comparison of a query to a reference set."""

    raw: float  # the combined score that downstream normalisation consumes
    max_similarity: float
    mean_similarity: float
    min_similarity: float
    per_reference: list[float]
    n_references: int
    # The baseline subtracted from the combined similarity to produce `raw`.
    intra_reference_mean: float = 0.0
    # Where that baseline came from:
    #   "own"        - this customer's specimens agree with each other this well
    #   "population" - only one specimen on file, so the corpus median is used
    #   "none"       - no baseline available; `raw` is a bare similarity and is
    #                  NOT on the scale the calibrator was fitted for
    baseline_source: str = "none"
    # The stored specimens do not resemble each other. A broken enrolment, not
    # a customer who is easy to match.
    specimens_disagree: bool = False
    # Whether the absolute floor actually decided `raw`, as opposed to merely
    # being available. Recorded because a diagnostic that reports which term
    # *would* bind rather than which one *did* is worse than none: it sent an
    # investigation in the wrong direction once already.
    floor_applied: bool = False

    @property
    def is_single_reference(self) -> bool:
        return self.n_references == 1

    @property
    def is_writer_normalised(self) -> bool:
        return self.baseline_source == "own"

    def to_dict(self) -> dict:
        return {
            "raw": round(self.raw, 5),
            "max_similarity": round(self.max_similarity, 5),
            "mean_similarity": round(self.mean_similarity, 5),
            "min_similarity": round(self.min_similarity, 5),
            "per_reference": [round(s, 5) for s in self.per_reference],
            "n_references": self.n_references,
            "single_reference": self.is_single_reference,
            "intra_reference_mean": round(self.intra_reference_mean, 5),
            "writer_normalised": self.is_writer_normalised,
            "baseline_source": self.baseline_source,
            "specimens_disagree": self.specimens_disagree,
        }


def intra_reference_mean(references: np.ndarray) -> float:
    """Mean pairwise cosine among a customer's own specimens.

    A measure of how consistently this person signs. Computed once at
    enrolment, since it changes only when the specimen set does.

    Returns 0.0 for a single specimen, where "how much do they agree with each
    other" is not a question that has an answer. Callers treat 0.0 as "not
    available" and fall back to the unnormalised similarity.
    """
    refs = l2_normalize(references)
    n = refs.shape[0]
    if n < 2:
        return 0.0
    gram = refs @ refs.T
    # Off-diagonal mean: the diagonal is every specimen compared to itself.
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def compare_to_references(
    query: np.ndarray,
    references: np.ndarray,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    writer_normalise: bool = True,
    reference_mean: float | None = None,
    population_reference_mean: float = 0.0,
    similarity_floor: float | None = None,
    cfg: ScoringConfig = SCORING,
) -> ComparisonScore:
    """Score one query embedding against a customer's reference embeddings.

    Args:
        query: embedding of shape (D,) or (1, D).
        references: embeddings of shape (N, D). At least one row.
        max_weight: weight on the nearest-reference term, in [0, 1].
        writer_normalise: express the score relative to how well the customer's
            own specimens agree with each other. See the module docstring — this
            is where most of the accuracy lives.
        reference_mean: precomputed :func:`intra_reference_mean`. Pass it in
            production; it only changes when the specimen set changes.
        population_reference_mean: the corpus-wide median specimen agreement,
            used when a customer has only one specimen on file and their own
            consistency therefore cannot be measured.
        similarity_floor: the similarity below which no genuine comparison was
            observed on validation. Measured per corpus, because the absolute
            cosine scale is a property of the trained model — a constant tuned
            on one corpus says nothing about another. Falls back to
            ``cfg.absolute_similarity_floor`` when not supplied.

    Returns cosine similarities in [-1, 1]; the combined ``raw`` is shifted by
    the baseline, so it can go slightly negative for a query less consistent
    with the specimens than they are with each other. That is meaningful, not a
    bug.

    **The single-specimen case is why ``population_reference_mean`` exists.**
    Subtracting nothing leaves ``raw`` as a bare similarity around 0.8-1.0,
    while the calibrator is fitted on margins around zero. Everything then
    clips to the top of the curve and *every* verification returns ~100,
    forgeries included. Falling back to the population median keeps
    single-specimen customers on the scale the calibrator was fitted for. It is
    an approximation — the customer's real consistency is unknown — and it is
    labelled ``baseline_source="population"`` so callers can say so.
    """
    q = l2_normalize(np.asarray(query).reshape(1, -1))[0]
    refs = l2_normalize(references)
    if refs.shape[0] == 0:
        raise ValueError("A customer with no stored reference signature cannot be verified")
    if refs.shape[1] != q.shape[0]:
        raise ValueError(f"Dimension mismatch: query {q.shape[0]} vs references {refs.shape[1]}")

    similarities = refs @ q
    max_sim = float(similarities.max())
    mean_sim = float(similarities.mean())

    combined = max_weight * max_sim + (1.0 - max_weight) * mean_sim
    raw = combined

    mu_ref = 0.0
    source = "none"
    specimens_disagree = False
    if writer_normalise:
        # Decided by how many specimens there are, not by the sign of the
        # value. A customer whose two specimens barely resemble each other has
        # an agreement near zero — or below it — and that is a real measurement
        # about a real customer, not a missing one.
        own = None
        if refs.shape[0] >= 2:
            own = reference_mean if reference_mean is not None else intra_reference_mean(refs)

        if own is not None and own >= cfg.min_specimen_agreement:
            mu_ref, source = own, "own"
        elif population_reference_mean:
            # Either one specimen, or specimens that do not resemble each
            # other. The second case is a broken enrolment, and using its
            # near-zero agreement as the bar is what let a 6.8% match score 88.
            specimens_disagree = own is not None
            mu_ref, source = population_reference_mean, "population"
        elif own is not None:
            mu_ref, source = own, "own"

        raw -= mu_ref

    # An absolute backstop, applied last and only ever downward. Whatever the
    # baseline says, a similarity this far below any plausible signature match
    # is not a match.
    #
    # Only when a baseline was actually subtracted. Without one `raw` is a bare
    # similarity on a different scale entirely, and shifting it would corrupt
    # the plain-similarity mode the benchmark uses for comparison.
    # Applied whenever normalisation is on at all — including, especially, when
    # no baseline could be established.
    #
    # This used to be skipped when `source == "none"`, which was exactly
    # backwards. A customer with one specimen and a calibrator carrying no
    # population baseline got no baseline *and* no floor, so `raw` came through
    # as a bare similarity: a 4.7% match scored 69/100. The case with the least
    # information is the case that most needs the backstop.
    #
    # `writer_normalise=False` is the one exemption, because that caller has
    # explicitly asked for the plain similarity — the benchmark uses it to
    # compare recipes.
    floor_applied = False
    if writer_normalise:
        floor = (
            similarity_floor
            if similarity_floor is not None and similarity_floor > 0.0
            else cfg.absolute_similarity_floor
        )
        capped = combined - floor
        floor_applied = capped < raw
        raw = min(raw, capped)

    return ComparisonScore(
        raw=raw,
        max_similarity=max_sim,
        mean_similarity=mean_sim,
        min_similarity=float(similarities.min()),
        per_reference=[float(s) for s in similarities],
        n_references=int(refs.shape[0]),
        intra_reference_mean=mu_ref,
        baseline_source=source,
        specimens_disagree=specimens_disagree,
        floor_applied=floor_applied,
    )
