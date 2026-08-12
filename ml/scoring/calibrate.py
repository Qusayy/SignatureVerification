"""Turning a normalised score into a number an employee can act on.

An S-normalised score is a z-score: statistically meaningful, and useless on a
operator screen. "2.4 standard deviations above the impostor mean" is not
something to ask a operator to reason about under time pressure.

Calibration maps that z-score to a probability that the signature is genuine,
fitted with isotonic regression on held-out data. Isotonic is used rather than
a sigmoid (Platt scaling) because it makes no assumption about the shape of the
score distribution, only that higher scores are never less likely to be
genuine — which is exactly the guarantee needed and nothing more.

The fitted curve is stored as plain breakpoints rather than a pickled
scikit-learn object. A pickle ties the production service to the exact library
version that produced it, and an organisation's deployment lifecycle outlives any given
scikit-learn release.

**The output is a probability, not a decision.** Bands are advisory guidance
for the employee, who always makes the final call.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from ml.config import SCORING, ScoringConfig

__all__ = ["Band", "ScoreCalibrator", "band_for"]


class Band(StrEnum):
    """Advisory guidance shown alongside the score. Never a decision."""

    GREEN = "green"  # consistent with the stored specimens
    AMBER = "amber"  # inconclusive — inspect carefully
    RED = "red"  # inconsistent with the stored specimens

    @property
    def guidance(self) -> str:
        return {
            Band.GREEN: "Consistent with the stored specimen(s).",
            Band.AMBER: "Inconclusive. Compare manually before deciding.",
            Band.RED: "Not consistent with the stored specimen(s).",
        }[self]


def band_for(score_0_100: float, cfg: ScoringConfig = SCORING) -> Band:
    if score_0_100 >= cfg.green_min:
        return Band.GREEN
    if score_0_100 <= cfg.red_max:
        return Band.RED
    return Band.AMBER


@dataclass
class ScoreCalibrator:
    """Monotone piecewise-linear map from normalised score to P(genuine).

    Built by :meth:`fit` from isotonic regression, then evaluated with plain
    interpolation so inference carries no scikit-learn dependency.
    """

    x: np.ndarray  # ascending normalised scores
    y: np.ndarray  # non-decreasing probabilities in [0, 1]
    n_fit_genuine: int = 0
    n_fit_impostor: int = 0
    fitted_on: str = ""
    # The weights whose score distribution this curve was fitted to. A curve
    # applied to a different model maps the wrong z-scores onto confidence,
    # which is how two thirds of skilled forgeries came to print 99.5.
    weights_id: str = ""
    # Corpus median of per-customer specimen agreement. Lives here because it
    # is part of the score *scale* this curve was fitted for: a customer with
    # one specimen has no measurable consistency of their own, and without this
    # substitute their score is a bare similarity of ~0.9 fed into a curve whose
    # domain ends near 0.04 — so it clips to the ceiling and every single
    # specimen customer scores ~100, forgeries included.
    population_reference_mean: float = 0.0
    # A similarity below which no genuine comparison was ever observed on the
    # validation split. Derived from the corpus rather than fixed, because the
    # absolute cosine scale is a property of the trained model: a constant
    # tuned on one corpus is meaningless on another and fails silently. 0.0
    # means "not measured", and the configured default is used instead.
    genuine_similarity_floor: float = 0.0

    # -- fitting ----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        genuine_scores: np.ndarray | list[float],
        impostor_scores: np.ndarray | list[float],
        *,
        fitted_on: str = "",
        weights_id: str = "",
        population_reference_mean: float = 0.0,
        genuine_similarity_floor: float = 0.0,
        clip: tuple[float, float] = (0.005, 0.995),
        min_samples: int = 10,
    ) -> ScoreCalibrator:
        """Fit the calibration curve on held-out comparison scores.

        Fit this on the **validation** split, never on the sealed test set: a
        calibrator fitted on test data reports a confidence that is
        indistinguishable from having peeked.

        Args:
            clip: probabilities are clipped away from exactly 0 and 1. A
                verification system that reports absolute certainty is
                misleading regardless of how clean the score was.
        """
        from sklearn.isotonic import IsotonicRegression

        g = np.asarray(genuine_scores, dtype=float).ravel()
        i = np.asarray(impostor_scores, dtype=float).ravel()
        if len(g) < min_samples or len(i) < min_samples:
            raise ValueError(
                f"Too few scores to calibrate ({len(g)} genuine, {len(i)} impostor); "
                f"need at least {min_samples} of each and realistically hundreds."
            )
        if len(g) < 200 or len(i) < 200:
            # Not fatal — the stack has to run on a small corpus — but say it
            # plainly. The shipped curve was fitted on 72/96 and collapsed to a
            # five-step staircase that saturated at 0.995, so every score above
            # a modest threshold printed as 99.5 regardless of how good it was.
            warnings.warn(
                f"Calibrating on {len(g)} genuine / {len(i)} impostor comparisons. Below "
                "roughly 200 of each the isotonic fit degenerates into a few coarse steps "
                "and the 0-100 score stops discriminating near the top of its range. "
                "Enlarge the validation split before quoting calibrated scores.",
                stacklevel=2,
            )

        scores = np.concatenate([g, i])
        labels = np.concatenate([np.ones(len(g)), np.zeros(len(i))])
        order = np.argsort(scores)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        probabilities = iso.fit_transform(scores[order], labels[order])

        # Collapse to the breakpoints where the isotonic step function changes,
        # keeping the curve compact and exactly reproducible.
        xs, ys = scores[order], probabilities
        keep = np.ones(len(xs), dtype=bool)
        keep[1:-1] = (np.diff(ys)[:-1] != 0) | (np.diff(ys)[1:] != 0)
        xs, ys = xs[keep], ys[keep]

        # Deduplicate equal x values, which np.interp cannot handle.
        unique_x, index = np.unique(xs, return_index=True)
        unique_y = np.maximum.accumulate(ys[index])

        return cls(
            x=unique_x,
            y=np.clip(unique_y, *clip),
            n_fit_genuine=len(g),
            n_fit_impostor=len(i),
            fitted_on=fitted_on,
            weights_id=weights_id,
            population_reference_mean=population_reference_mean,
            genuine_similarity_floor=genuine_similarity_floor,
        )

    # -- application ------------------------------------------------------

    def probability(self, score: float | np.ndarray) -> float | np.ndarray:
        """Map a normalised score to P(genuine)."""
        result = np.interp(np.asarray(score, dtype=float), self.x, self.y)
        return float(result) if np.isscalar(score) or result.ndim == 0 else result

    def score_0_100(self, score: float) -> float:
        """Map a normalised score to the 0-100 confidence shown to employees."""
        return round(float(self.probability(score)) * 100.0, 1)

    def band(self, score: float, cfg: ScoringConfig = SCORING) -> Band:
        return band_for(self.score_0_100(score), cfg)

    def threshold_for_probability(self, probability: float) -> float:
        """Invert the curve: the normalised score achieving a given P(genuine)."""
        return float(np.interp(probability, self.y, self.x))

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "x": self.x.tolist(),
                    "y": self.y.tolist(),
                    "n_fit_genuine": self.n_fit_genuine,
                    "n_fit_impostor": self.n_fit_impostor,
                    "fitted_on": self.fitted_on,
                    "weights_id": self.weights_id,
                    "population_reference_mean": self.population_reference_mean,
                    "genuine_similarity_floor": self.genuine_similarity_floor,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> ScoreCalibrator:
        payload = json.loads(Path(path).read_text())
        return cls(
            x=np.asarray(payload["x"], dtype=float),
            y=np.asarray(payload["y"], dtype=float),
            n_fit_genuine=payload.get("n_fit_genuine", 0),
            n_fit_impostor=payload.get("n_fit_impostor", 0),
            fitted_on=payload.get("fitted_on", ""),
            weights_id=payload.get("weights_id", ""),
            population_reference_mean=payload.get("population_reference_mean", 0.0),
            genuine_similarity_floor=payload.get("genuine_similarity_floor", 0.0),
        )

    @classmethod
    def identity(cls) -> ScoreCalibrator:
        """An uncalibrated fallback mapping z-scores in [-2, 6] onto [0, 1].

        Used only so the stack runs before a calibrator has been fitted. Any
        score produced through this path must be labelled uncalibrated in the
        UI — it is a placeholder, not a confidence.
        """
        x = np.linspace(-2.0, 6.0, 33)
        y = np.clip((x + 2.0) / 8.0, 0.005, 0.995)
        return cls(x=x, y=y, fitted_on="identity-placeholder")

    @property
    def is_placeholder(self) -> bool:
        return self.fitted_on == "identity-placeholder"
