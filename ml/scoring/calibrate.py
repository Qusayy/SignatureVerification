"""Turning a similarity into a number an employee can act on.

The model emits a cosine similarity between the captured signature and the
customer's stored specimen. That number is not a confidence: 0.91 is excellent
for one model and mediocre for another, and it says nothing about how often a
signature scoring 0.91 turns out to be genuine.

Calibration answers that question. Isotonic regression, fitted on held-out
comparisons, maps similarity to P(genuine). Isotonic rather than a sigmoid
because it assumes nothing about the shape of the distribution — only that a
higher similarity is never less likely to be genuine, which is exactly the
guarantee needed and nothing more.

Three properties this file is responsible for, each learned the hard way:

**The curve must be fitted on the protocol that is served.** A curve fitted on
six stored specimens, applied to a customer with one, is a value from one
measurement pushed through a scale built for another. Every score the system
showed before this was that. ``protocol_references`` records the protocol and
the service refuses a mismatch.

**The curve must emit a continuum, not a staircase.** Isotonic produces flat
blocks; keeping both ends of each block makes ``np.interp`` reproduce the steps
literally, and the shipped curve had **eight** distinct outputs. One knot per
block, placed at the block's centroid, interpolates between them instead — same
fit, ~800 distinct scores rather than 8.

**The curve must not claim certainty it has not earned.** The top isotonic block
reaches probability 1.0 whenever the highest few comparisons happen to be
genuine, which is why 99.5 printed from a model with AUC 0.80. Laplace
smoothing per block caps the ceiling at what the evidence supports.

**The output is a probability, not a decision.** Bands are advisory guidance for
the employee, who always makes the final call.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from ml.config import SCORING, ScoringConfig

__all__ = ["Band", "ScoreCalibrator", "CalibratorSchemaError", "band_for"]

# Bumped when the meaning of `x` changes. Version 1 was fitted on
# writer-normalised margins; version 2 is fitted on raw cosine similarity. A
# version-1 curve applied to a similarity produces confident nonsense, so it is
# refused rather than migrated.
SCHEMA_VERSION = 2

# Quantiles of the fitted populations, stored so the interface can put a score
# in context: "34% of genuine signatures reach this level".
QUANTILE_POINTS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class CalibratorSchemaError(RuntimeError):
    """A calibrator artifact this code cannot safely use."""


class Band(StrEnum):
    """Advisory guidance shown alongside the score. Never a decision."""

    GREEN = "green"  # consistent with the stored specimen
    AMBER = "amber"  # inconclusive — inspect carefully
    RED = "red"  # inconsistent with the stored specimen

    @property
    def guidance(self) -> str:
        return {
            Band.GREEN: "Consistent with the stored specimen.",
            Band.AMBER: "Inconclusive. Compare manually before deciding.",
            Band.RED: "Not consistent with the stored specimen.",
        }[self]


def band_for(score_0_100: float, green_min: float, red_max: float) -> Band:
    if score_0_100 >= green_min:
        return Band.GREEN
    if score_0_100 <= red_max:
        return Band.RED
    return Band.AMBER


def _isotonic_knots(
    scores: np.ndarray, labels: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression and reduce it to one knot per block.

    Returns ascending ``x`` with non-decreasing ``y``.

    Two departures from the obvious implementation, both of which matter more
    than they look:

    * **One knot per flat block, at the block's centroid.** Keeping both ends of
      a block makes the piecewise-linear reconstruction step rather than ramp,
      so the curve emits only as many values as it has blocks. Isotonic on 500
      genuine / 600 impostor comparisons produces roughly 18 blocks — eight
      distinct scores after rounding. Interpolating between block centroids
      instead yields ~800, at slightly *better* log loss.
    * **Laplace smoothing inside each block**, ``(k + alpha) / (n + 2*alpha)``.
      The raw isotonic solution assigns probability 1.0 to any top block that
      happens to contain only genuine comparisons, however few. That is how a
      model with AUC 0.80 came to print 99.5.
    """
    from sklearn.isotonic import IsotonicRegression

    order = np.argsort(scores, kind="mergesort")
    xs, ys = scores[order], labels[order]

    fitted = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit_transform(xs, ys)

    # Block boundaries: positions where the fitted value changes.
    edges = np.flatnonzero(np.diff(fitted)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(fitted)]])

    knots_x, knots_y = [], []
    for start, end in zip(starts, ends, strict=True):
        members = ys[start:end]
        n = len(members)
        smoothed = (members.sum() + alpha) / (n + 2.0 * alpha)
        # The centroid of the block's x values, so neighbouring blocks are
        # joined by a ramp through the mass of each rather than at its edge.
        knots_x.append(float(xs[start:end].mean()))
        knots_y.append(float(smoothed))

    x = np.asarray(knots_x, dtype=float)
    y = np.maximum.accumulate(np.asarray(knots_y, dtype=float))

    # np.interp requires strictly ascending x; identical block centroids can
    # occur when many comparisons share a score.
    unique_x, index = np.unique(x, return_index=True)
    return unique_x, np.maximum.accumulate(y[index])


@dataclass
class ScoreCalibrator:
    """Monotone piecewise-linear map from similarity to P(genuine).

    Built by :meth:`fit`, then evaluated with plain interpolation so inference
    carries no scikit-learn dependency. Stored as breakpoints rather than a
    pickle: a pickle ties the service to the exact library version that produced
    it, and a deployment lifecycle outlives any scikit-learn release.
    """

    x: np.ndarray  # ascending similarities
    y: np.ndarray  # non-decreasing probabilities in [0, 1]

    # --- Provenance: what this curve is for ------------------------------
    # How many stored specimens the fitting protocol used. The service refuses
    # a curve whose protocol differs from the one it serves.
    protocol_references: int = 1
    # What `x` is. Recorded so a future change of input is a loud failure.
    score_input: str = "similarity"
    schema_version: int = SCHEMA_VERSION
    weights_id: str = ""
    fitted_on: str = ""
    n_fit_genuine: int = 0
    n_fit_impostor: int = 0
    thin_fit: bool = False

    # --- Band edges, derived from operating points -----------------------
    # On the 0-100 scale. Set by `derive_band_edges` from validation, because a
    # fixed edge on a probability has no operational meaning: the probability is
    # conditioned on the benchmark's genuine/impostor mix, not on the base rate
    # at a counter. FAR and FRR are conditional on class, so they are invariant
    # to that mix and can be held to a target.
    green_min: float = 0.0
    red_max: float = 0.0
    operating_points: dict = field(default_factory=dict)

    # --- Context for the interface ---------------------------------------
    genuine_quantiles: dict = field(default_factory=dict)
    impostor_quantiles: dict = field(default_factory=dict)
    # Similarity above which a match is too perfect to be a fresh signature.
    # Expressed on the similarity, not on the calibrated score: the score's
    # ceiling moves with the fit, and a fraud control that depends on reaching
    # 99/100 silently stops firing when the ceiling drops.
    duplicate_similarity_min: float = 1.0

    # -- fitting ----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        genuine_scores: np.ndarray | list[float],
        impostor_scores: np.ndarray | list[float],
        *,
        protocol_references: int,
        fitted_on: str = "",
        weights_id: str = "",
        alpha: float = 1.0,
        min_samples: int = 50,
        thin_threshold: int = 200,
    ) -> ScoreCalibrator:
        """Fit the calibration curve on held-out comparison similarities.

        Fit on the **validation** split, never on the sealed test set: a
        calibrator fitted on test data reports a confidence indistinguishable
        from having peeked.

        Args:
            protocol_references: specimens per customer in the fitting
                protocol. Must equal what production serves.
            alpha: Laplace smoothing inside each isotonic block. Caps the
                ceiling at what the evidence supports.
        """
        g = np.asarray(genuine_scores, dtype=float).ravel()
        i = np.asarray(impostor_scores, dtype=float).ravel()
        if len(g) < min_samples or len(i) < min_samples:
            raise ValueError(
                f"Too few scores to calibrate ({len(g)} genuine, {len(i)} impostor); "
                f"need at least {min_samples} of each and realistically hundreds. "
                "The fix is more validation *writers*, not more comparisons per "
                "writer, and never borrowing from train — the model memorised "
                "those writers, so the curve would be optimistic."
            )

        thin = len(g) < thin_threshold or len(i) < thin_threshold
        if thin:
            warnings.warn(
                f"Calibrating on {len(g)} genuine / {len(i)} impostor comparisons, below "
                f"{thin_threshold} of each. The curve will be coarse and its ceiling "
                "conservative. Widen the validation split before quoting scores.",
                stacklevel=2,
            )

        x, y = _isotonic_knots(
            np.concatenate([g, i]),
            np.concatenate([np.ones(len(g)), np.zeros(len(i))]),
            alpha=alpha,
        )

        quantiles = {f"{q:g}": round(float(np.quantile(g, q)), 5) for q in QUANTILE_POINTS}
        impostor_quantiles = {
            f"{q:g}": round(float(np.quantile(i, q)), 5) for q in QUANTILE_POINTS
        }

        return cls(
            x=x,
            y=y,
            protocol_references=protocol_references,
            fitted_on=fitted_on,
            weights_id=weights_id,
            n_fit_genuine=len(g),
            n_fit_impostor=len(i),
            thin_fit=thin,
            genuine_quantiles=quantiles,
            impostor_quantiles=impostor_quantiles,
            # A genuine signature is never reproduced exactly, so a similarity
            # above essentially every genuine comparison indicates a copy.
            #
            # Capped below 1.0 deliberately. A cosine cannot exceed 1, so a
            # threshold at or above it can never be met and the photocopy check
            # would be permanently disabled — the same failure as the old
            # score-based threshold of 99, arrived at from the other direction.
            duplicate_similarity_min=round(min(float(np.quantile(g, 0.999)), 0.999), 5),
        )

    def derive_band_edges(
        self,
        genuine_scores: np.ndarray | list[float],
        impostor_scores: np.ndarray | list[float],
        *,
        green_max_far: float = SCORING.green_max_far,
        red_max_frr: float = SCORING.red_max_frr,
    ) -> None:
        """Set the band edges from operating points on the fitting data.

        Green is the tightest score at which no more than ``green_max_far`` of
        forgeries are accepted; red the loosest at which no more than
        ``red_max_frr`` of genuine signatures are rejected. Both are conditional
        on class, so unlike a fixed threshold on the probability they do not
        move when the corpus composition does.
        """
        g = np.sort(np.asarray([self.score_0_100(s) for s in genuine_scores], dtype=float))
        i = np.sort(np.asarray([self.score_0_100(s) for s in impostor_scores], dtype=float))

        # Green: lowest score whose FAR is within target.
        candidates = np.unique(np.concatenate([g, i]))
        green = float(candidates[-1])
        for threshold in candidates:
            if float((i >= threshold).mean()) <= green_max_far:
                green = float(threshold)
                break

        # Red: highest score whose FRR is within target.
        red = float(candidates[0])
        for threshold in candidates[::-1]:
            if float((g <= threshold).mean()) <= red_max_frr:
                red = float(threshold)
                break

        # A degenerate fit can invert them; keep the ordering sane rather than
        # emitting a band scheme where every score is simultaneously green and
        # red.
        if red >= green:
            red = max(0.0, green - 1.0)

        self.green_min = round(green, 1)
        self.red_max = round(red, 1)
        self.operating_points = {
            "green_max_far": green_max_far,
            "red_max_frr": red_max_frr,
            "achieved_far_at_green": round(float((i >= green).mean()), 5),
            "achieved_frr_at_red": round(float((g <= red).mean()), 5),
            "genuine_share_green": round(float((g >= green).mean()), 5),
            "genuine_share_red": round(float((g <= red).mean()), 5),
        }

    # -- application ------------------------------------------------------

    def probability(self, similarity: float | np.ndarray) -> float | np.ndarray:
        """Map a similarity to P(genuine)."""
        result = np.interp(np.asarray(similarity, dtype=float), self.x, self.y)
        return float(result) if np.isscalar(similarity) or result.ndim == 0 else result

    def score_0_100(self, similarity: float) -> float:
        """The 0-100 confidence shown to employees."""
        return round(float(self.probability(similarity)) * 100.0, 1)

    def band(self, similarity: float) -> Band:
        """Band from this curve's own derived edges, not from global config."""
        green, red = self.effective_edges()
        return band_for(self.score_0_100(similarity), green, red)

    def effective_edges(self) -> tuple[float, float]:
        """Derived edges when present, otherwise the configured fallback."""
        if self.green_min > 0.0:
            return self.green_min, self.red_max
        return SCORING.green_min_fallback, SCORING.red_max_fallback

    def similarity_for_score(self, score_0_100: float) -> float:
        """Invert the curve: the similarity achieving a given score."""
        return float(np.interp(score_0_100 / 100.0, self.y, self.x))

    def share_reaching(self, similarity: float, population: str = "genuine") -> float | None:
        """Share of a stored population at or above this similarity.

        Powers the one sentence that turns a bare number into a decision:
        "34% of genuine signatures and 6% of practised forgeries reach this
        level." Interpolated from the stored quantiles; None when unavailable.
        """
        table = self.genuine_quantiles if population == "genuine" else self.impostor_quantiles
        if not table:
            return None
        qs = np.array([float(k) for k in table], dtype=float)
        vs = np.array([table[k] for k in table], dtype=float)
        order = np.argsort(vs)
        # np.interp gives the quantile at this similarity; the share at or above
        # it is the complement.
        below = float(np.interp(similarity, vs[order], qs[order]))
        return round(max(0.0, min(1.0, 1.0 - below)), 4)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "score_input": self.score_input,
                    "protocol_references": self.protocol_references,
                    "x": [round(v, 6) for v in self.x.tolist()],
                    "y": [round(v, 6) for v in self.y.tolist()],
                    "green_min": self.green_min,
                    "red_max": self.red_max,
                    "operating_points": self.operating_points,
                    "genuine_quantiles": self.genuine_quantiles,
                    "impostor_quantiles": self.impostor_quantiles,
                    "duplicate_similarity_min": self.duplicate_similarity_min,
                    "n_fit_genuine": self.n_fit_genuine,
                    "n_fit_impostor": self.n_fit_impostor,
                    "thin_fit": self.thin_fit,
                    "fitted_on": self.fitted_on,
                    "weights_id": self.weights_id,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> ScoreCalibrator:
        payload = json.loads(Path(path).read_text())
        version = payload.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            raise CalibratorSchemaError(
                f"{Path(path).name} is schema version {version}; this code needs "
                f"{SCHEMA_VERSION}. Version 1 was fitted on writer-normalised margins "
                "and applying it to a similarity produces confident nonsense, so it "
                "cannot be migrated. Regenerate it:\n"
                "    python -m ml.eval.benchmark --checkpoint <checkpoint> --split test"
            )

        return cls(
            x=np.asarray(payload["x"], dtype=float),
            y=np.asarray(payload["y"], dtype=float),
            protocol_references=payload.get("protocol_references", 1),
            score_input=payload.get("score_input", "similarity"),
            schema_version=version,
            weights_id=payload.get("weights_id", ""),
            fitted_on=payload.get("fitted_on", ""),
            n_fit_genuine=payload.get("n_fit_genuine", 0),
            n_fit_impostor=payload.get("n_fit_impostor", 0),
            thin_fit=payload.get("thin_fit", False),
            green_min=payload.get("green_min", 0.0),
            red_max=payload.get("red_max", 0.0),
            operating_points=payload.get("operating_points", {}),
            genuine_quantiles=payload.get("genuine_quantiles", {}),
            impostor_quantiles=payload.get("impostor_quantiles", {}),
            duplicate_similarity_min=payload.get("duplicate_similarity_min", 1.0),
        )

    @property
    def distinct_scores(self) -> int:
        """How many different numbers this curve can emit, as displayed.

        A curve with a handful cannot support a decision, however good the
        model behind it.
        """
        probe = np.linspace(float(self.x.min()), float(self.x.max()), 2000)
        return len({round(float(v) * 100.0, 1) for v in self.probability(probe)})
