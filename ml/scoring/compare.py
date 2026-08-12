"""Comparing a query signature against a customer's stored specimen.

The live question is never "whose signature is this?" but "is this *this*
customer's signature?" — a 1:1 verification against what is on file, not a 1:N
identification.

**One specimen is the design point.** The customer base is too large for a
specimen-collection programme, so a customer has one stored signature and will
continue to. This module is therefore small: L2-normalise, take the cosine
against each stored specimen, and return the nearest.

Two things that used to live here and no longer do, both worth knowing about
because they looked reasonable:

* **Writer-internal normalisation** — subtracting the mean pairwise similarity
  among a customer's own specimens, so the score asked "is this query as close
  to the specimens as they are to each other?". Worth 15.7 EER points at two or
  more specimens. At one specimen there are no pairs, so the "baseline" became a
  corpus-wide constant and the whole apparatus reduced to subtracting the same
  number from every score — an identity transform dressed as arithmetic.
* **An absolute similarity floor**, added to stop a broken enrolment scoring a
  nonsense match highly. It was compensating for the same missing baseline.
  With the calibration curve fitted on similarity directly, a low similarity
  maps to a low probability by construction and needs no separate guard.

Both survive in `ml.eval.benchmark` as alternative recipes, so if the
organisation ever runs a multi-specimen pilot the evidence is one benchmark run
away.

Pooling is `max`, not a blend of max and mean. That keeps the score's meaning
invariant to specimen count: with one specimen it is exactly the condition the
calibrator was fitted on, and an additional specimen can only raise a genuine
customer's score. A blended statistic shifts distribution with the count, so
the same signature would score differently for reasons that have nothing to do
with the signature.
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


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Project embeddings onto the unit sphere, where cosine == dot product."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@dataclass
class ComparisonScore:
    """A query compared against a customer's stored specimen(s)."""

    # The number the calibrator consumes: cosine to the nearest specimen, in
    # [-1, 1]. This is also the figure shown on screen as "Matched at 91.3%",
    # so the whole chain from pixels to score is checkable by eye.
    similarity: float
    max_similarity: float
    mean_similarity: float
    min_similarity: float
    per_reference: list[float]
    n_references: int

    @property
    def is_single_reference(self) -> bool:
        return self.n_references == 1

    def to_dict(self) -> dict:
        return {
            # Marks which scoring generation produced this row. Version 1 stored
            # `raw`, a writer-normalised margin; version 2 stores a similarity.
            # The audit trail is append-only, so both shapes coexist and a
            # reader needs to be able to tell them apart.
            "scoring_version": 2,
            "similarity": round(self.similarity, 5),
            "max_similarity": round(self.max_similarity, 5),
            "mean_similarity": round(self.mean_similarity, 5),
            "min_similarity": round(self.min_similarity, 5),
            "per_reference": [round(s, 5) for s in self.per_reference],
            "n_references": self.n_references,
            "single_reference": self.is_single_reference,
        }


def intra_reference_mean(references: np.ndarray) -> float:
    """Mean pairwise cosine among a customer's own specimens.

    **Diagnostic only.** Nothing on the scoring path reads this. It answers "do
    the specimens on file look like the same hand?", which detects a broken
    enrolment — wrong customer, mis-cropped scan, two different people — and is
    surfaced by `api.doctor` rather than at the counter, because the counter is
    not where an enrolment gets fixed.

    Returns 0.0 for a single specimen, where the question has no answer.
    """
    refs = l2_normalize(references)
    n = refs.shape[0]
    if n < 2:
        return 0.0
    gram = refs @ refs.T
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def compare_to_references(
    query: np.ndarray,
    references: np.ndarray,
    *,
    cfg: ScoringConfig = SCORING,
) -> ComparisonScore:
    """Score one query embedding against a customer's stored specimen(s).

    Args:
        query: embedding of shape (D,) or (1, D).
        references: embeddings of shape (N, D). At least one row; normally one.

    Returns cosine similarities in [-1, 1]. ``similarity`` is what the
    calibrator consumes; the rest is display metadata.
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

    similarity = max_sim if cfg.pooling == "max" else 0.5 * max_sim + 0.5 * mean_sim

    return ComparisonScore(
        similarity=similarity,
        max_similarity=max_sim,
        mean_similarity=mean_sim,
        min_similarity=float(similarities.min()),
        per_reference=[float(s) for s in similarities],
        n_references=int(refs.shape[0]),
    )
