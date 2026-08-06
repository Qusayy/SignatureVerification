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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ComparisonScore", "compare_to_references", "l2_normalize"]

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

    @property
    def is_single_reference(self) -> bool:
        return self.n_references == 1

    def to_dict(self) -> dict:
        return {
            "raw": round(self.raw, 5),
            "max_similarity": round(self.max_similarity, 5),
            "mean_similarity": round(self.mean_similarity, 5),
            "min_similarity": round(self.min_similarity, 5),
            "per_reference": [round(s, 5) for s in self.per_reference],
            "n_references": self.n_references,
            "single_reference": self.is_single_reference,
        }


def compare_to_references(
    query: np.ndarray,
    references: np.ndarray,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> ComparisonScore:
    """Score one query embedding against a customer's reference embeddings.

    Args:
        query: embedding of shape (D,) or (1, D).
        references: embeddings of shape (N, D). At least one row.
        max_weight: weight on the nearest-reference term, in [0, 1].

    Returns cosine similarities in [-1, 1]; higher means more similar.
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

    raw = max_weight * max_sim + (1.0 - max_weight) * mean_sim
    return ComparisonScore(
        raw=raw,
        max_similarity=max_sim,
        mean_similarity=mean_sim,
        min_similarity=float(similarities.min()),
        per_reference=[float(s) for s in similarities],
        n_references=int(refs.shape[0]),
    )
