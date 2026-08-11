"""Cohort score normalisation (S-norm).

The single largest practical accuracy gain available in this system, and the
one most consistently missing from published signature verification work.

The problem: a raw similarity of 0.72 means different things for different
people. Some signatures are elaborate and distinctive, so *nobody* scores 0.72
against them and 0.72 is a strong match. Others are a plain scrawl that half
the population scores 0.72 against, where the same number is worthless. A
single global threshold therefore over-rejects distinctive signers and
over-accepts generic ones.

The fix is to express every score relative to the impostor distribution it
should be compared against, using a fixed background cohort of unrelated
signers:

* **Z-norm** (enrolment side) — score the customer's stored references against
  cohort signatures. Gives (mu, sigma) per customer, computed once at
  enrolment, capturing "how generic is this customer's signature".
* **T-norm** (query side) — score the incoming query against cohort
  references. Computed per verification, capturing "how generic is what was
  just written".
* **S-norm** — the mean of the two z-scores. Symmetric, and more stable than
  either alone.

The output is a z-score: how many standard deviations above the impostor mean
this comparison sits. That is directly comparable across customers, which is
exactly what a single organisation-wide threshold needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.config import SCORING
from ml.scoring.compare import l2_normalize

__all__ = ["CohortStats", "CohortNormalizer"]


@dataclass
class CohortStats:
    """Impostor score statistics for one enrolled customer."""

    mean: float
    std: float

    def z(self, score: float) -> float:
        return (score - self.mean) / max(self.std, 1e-6)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class CohortNormalizer:
    """Holds the background cohort and applies S-norm.

    The cohort must contain signers who are **not** customers being verified
    and, critically, who are not in the model's training set either — otherwise
    the impostor distribution is measured against writers the model has
    memorised and comes out unrealistically tight.
    """

    def __init__(self, cohort_embeddings: np.ndarray, cohort_signers: list[str] | None = None):
        embeddings = l2_normalize(np.asarray(cohort_embeddings, dtype=np.float64))
        if embeddings.shape[0] < 10:
            raise ValueError(
                f"Cohort of {embeddings.shape[0]} is too small to estimate an impostor "
                "distribution; use at least ~100 unrelated signers."
            )
        self.embeddings = embeddings
        self.signers = cohort_signers or [f"cohort_{i}" for i in range(len(embeddings))]

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    # -- enrolment side ---------------------------------------------------

    def enrolment_stats(self, references: np.ndarray) -> CohortStats:
        """Z-norm statistics for a customer, computed once at enrolment.

        Scores every stored reference against every cohort member and
        summarises the resulting impostor distribution.
        """
        refs = l2_normalize(np.asarray(references, dtype=np.float64))
        scores = (self.embeddings @ refs.T).ravel()
        return CohortStats(mean=float(scores.mean()), std=float(scores.std()))

    # -- query side -------------------------------------------------------

    def query_stats(self, query: np.ndarray) -> CohortStats:
        """T-norm statistics for one incoming query signature."""
        q = l2_normalize(np.asarray(query).reshape(1, -1))[0]
        scores = self.embeddings @ q
        return CohortStats(mean=float(scores.mean()), std=float(scores.std()))

    # -- combination ------------------------------------------------------

    def snorm(
        self,
        raw_score: float,
        query: np.ndarray,
        references: np.ndarray | None = None,
        enrolment: CohortStats | None = None,
    ) -> float:
        """Return the S-normalised score for one comparison.

        Args:
            raw_score: the combined similarity from
                :func:`ml.scoring.compare.compare_to_references`.
            query: the query embedding, for the T-norm half.
            references: the customer's reference embeddings. Only needed if
                ``enrolment`` is not supplied.
            enrolment: precomputed Z-norm statistics. Pass this in production —
                recomputing it against the whole cohort on every verification
                is wasted work, since it only changes when the customer's
                specimens change.
        """
        if enrolment is None:
            if references is None:
                raise ValueError("Provide either precomputed enrolment stats or the references")
            enrolment = self.enrolment_stats(references)

        z_enrol = enrolment.z(raw_score)
        z_query = self.query_stats(query).z(raw_score)
        return float((z_enrol + z_query) / 2.0)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | str, *, weights_id: str = "") -> Path:
        """Persist the cohort, stamped with the weights that produced it.

        The stamp is not bookkeeping. Cohort vectors are embeddings, so they
        are only meaningful under the model that produced them; pairing them
        with different weights yields z-scores that look entirely normal and
        mean nothing. That exact mistake shipped — see
        :meth:`ml.scoring.verifier.Verifier.from_artifacts`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np.float32),
            signers=np.array(self.signers),
            weights_id=np.array(weights_id),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> CohortNormalizer:
        payload = np.load(Path(path), allow_pickle=False)
        normalizer = cls(payload["embeddings"], list(payload["signers"]))
        # Absent on cohorts written before the stamp existed.
        normalizer.weights_id = str(payload["weights_id"]) if "weights_id" in payload else ""
        return normalizer

    @classmethod
    def from_embeddings_by_signer(
        cls,
        embeddings_by_signer: dict[str, np.ndarray],
        *,
        size: int = SCORING.cohort_size,
        seed: int = 1337,
    ) -> CohortNormalizer:
        """Build a cohort by taking one representative vector per signer.

        One vector per signer, not one per image: several samples from the same
        writer would weight that writer disproportionately and narrow the
        estimated impostor spread.
        """
        rng = np.random.default_rng(seed)
        signers = sorted(embeddings_by_signer)
        if len(signers) > size:
            chosen = rng.choice(len(signers), size=size, replace=False)
            signers = [signers[i] for i in sorted(chosen)]

        vectors = []
        for signer in signers:
            samples = l2_normalize(np.asarray(embeddings_by_signer[signer], dtype=np.float64))
            # The signer's centroid is a more stable representative than any
            # single sample.
            centroid = samples.mean(axis=0)
            vectors.append(centroid)
        return cls(np.vstack(vectors), signers)

    def describe(self) -> dict:
        return {"cohort_size": self.size, "embedding_dim": int(self.embeddings.shape[1])}


def write_cohort_manifest(normalizer: CohortNormalizer, path: Path | str) -> Path:
    """Record which signers form the cohort, for audit and reproducibility."""
    path = Path(path)
    path.write_text(json.dumps({"signers": normalizer.signers, **normalizer.describe()}, indent=2))
    return path
