"""Biometric verification metrics.

Terminology, because "accuracy" is ambiguous here and the ambiguity has cost
projects real money:

* **FAR** (false acceptance rate) — the share of *impostor* comparisons the
  system accepts. The fraud-risk number.
* **FRR** (false rejection rate) — the share of *genuine* comparisons it
  rejects. The customer-friction number.
* **EER** — the threshold where FAR equals FRR. Threshold-independent, so it is
  the right number for comparing two models. It is *not* the right number for
  choosing an operating point.
* **TAR@FAR** — the true acceptance rate when the threshold is fixed to hold
  FAR at a chosen level. This is the number to commit to in an organisation: risk picks
  the tolerable FAR, and TAR is what the business gets in return.

Scores throughout are **similarity** scores: higher means more likely genuine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VerificationMetrics", "compute_metrics", "equal_error_rate", "tar_at_far", "det_curve"]


@dataclass
class VerificationMetrics:
    """Metrics for one population of genuine and impostor comparison scores."""

    n_genuine: int
    n_impostor: int
    eer: float
    eer_threshold: float
    auc: float
    tar_at_far: dict[float, float] = field(default_factory=dict)
    threshold_at_far: dict[float, float] = field(default_factory=dict)
    # Accuracy at the EER threshold. Reported only because stakeholders ask
    # for "accuracy"; EER and TAR@FAR are the metrics that should drive
    # decisions.
    accuracy_at_eer: float = 0.0
    # FAR targets finer than the impostor set can resolve. Measured FAR moves
    # in steps of 1/n_impostor, so a "TAR at FAR = 0.1%" claim needs at least
    # 1000 impostor comparisons to mean anything, and realistically ~10x that
    # for a stable estimate. Reporting the number without this caveat is how
    # unsupportable accuracy claims reach a procurement document.
    unresolvable_far_targets: list[float] = field(default_factory=list)

    @property
    def far_resolution(self) -> float:
        return 1.0 / self.n_impostor if self.n_impostor else float("inf")

    def to_dict(self) -> dict:
        return {
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "eer": round(self.eer, 5),
            "eer_threshold": round(self.eer_threshold, 5),
            "auc": round(self.auc, 5),
            "accuracy_at_eer": round(self.accuracy_at_eer, 5),
            "tar_at_far": {f"{k:g}": round(v, 5) for k, v in self.tar_at_far.items()},
            "threshold_at_far": {f"{k:g}": round(v, 5) for k, v in self.threshold_at_far.items()},
            "far_resolution": round(self.far_resolution, 6),
            "unresolvable_far_targets": [f"{t:g}" for t in self.unresolvable_far_targets],
        }


def _rates(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep every threshold and return (thresholds, FAR, FRR).

    Accept when ``score >= threshold``.
    """
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    # Extend past both ends so the sweep reaches FAR=0 and FRR=0.
    step = np.diff(thresholds).min() if len(thresholds) > 1 else 1e-6
    thresholds = np.concatenate([[thresholds[0] - step], thresholds, [thresholds[-1] + step]])

    # Vectorised over thresholds via sorted-array searches rather than an
    # O(T*N) broadcast, which matters once the test set has millions of pairs.
    g_sorted = np.sort(genuine)
    i_sorted = np.sort(impostor)
    frr = np.searchsorted(g_sorted, thresholds, side="left") / max(len(g_sorted), 1)
    far = 1.0 - np.searchsorted(i_sorted, thresholds, side="left") / max(len(i_sorted), 1)
    return thresholds, far, frr


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold at the EER)."""
    thresholds, far, frr = _rates(genuine, impostor)
    diff = far - frr
    crossing = np.argmin(np.abs(diff))

    # Interpolate between the two thresholds bracketing the crossing, so the
    # EER is not quantised to the nearest observed score.
    if 0 < crossing < len(thresholds) - 1:
        lo, hi = crossing - 1, crossing + 1
        window = slice(lo, hi + 1)
        d = diff[window]
        sign_change = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if len(sign_change):
            i = lo + sign_change[0]
            d0, d1 = diff[i], diff[i + 1]
            t = 0.0 if d1 == d0 else d0 / (d0 - d1)
            eer = float(far[i] + t * (far[i + 1] - far[i]))
            thr = float(thresholds[i] + t * (thresholds[i + 1] - thresholds[i]))
            return max(eer, 0.0), thr

    return float((far[crossing] + frr[crossing]) / 2.0), float(thresholds[crossing])


def tar_at_far(
    genuine: np.ndarray, impostor: np.ndarray, target_far: float
) -> tuple[float, float]:
    """Return (TAR, threshold) at the tightest threshold holding FAR <= target.

    Beware of resolution: measured FAR only takes values that are multiples of
    ``1 / len(impostor)``, so asking for a target below that does not fail — it
    silently returns the FAR = 0 operating point instead. Check
    :attr:`VerificationMetrics.unresolvable_far_targets` before quoting a
    number, or use an impostor set at least 10x larger than 1/target.
    """
    thresholds, far, frr = _rates(genuine, impostor)
    feasible = np.where(far <= target_far)[0]
    if len(feasible) == 0:
        return 0.0, float(thresholds[-1])
    # far decreases as threshold rises; take the lowest such threshold to keep
    # TAR as high as the constraint allows.
    i = int(feasible[0])
    return float(1.0 - frr[i]), float(thresholds[i])


def det_curve(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (FAR, FRR) arrays for plotting a DET curve."""
    _, far, frr = _rates(genuine, impostor)
    return far, frr


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Area under the ROC curve, computed via the rank-sum identity."""
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan")
    scores = np.concatenate([genuine, impostor])
    ranks = np.argsort(np.argsort(scores)) + 1  # average-tie handling below
    # Correct for ties by averaging ranks of equal scores.
    order = np.argsort(scores)
    sorted_scores = scores[order]
    avg_ranks = np.empty(len(scores), dtype=float)
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            avg_ranks[order[start:end]] = (start + end + 1) / 2.0
            start = end
    ranks = avg_ranks

    n_g, n_i = len(genuine), len(impostor)
    rank_sum = ranks[:n_g].sum()
    return float((rank_sum - n_g * (n_g + 1) / 2.0) / (n_g * n_i))


def compute_metrics(
    genuine: np.ndarray | list[float],
    impostor: np.ndarray | list[float],
    far_targets: tuple[float, ...] = (0.10, 0.05, 0.01, 0.001),
) -> VerificationMetrics:
    """Compute the full metric set for one comparison population."""
    g = np.asarray(genuine, dtype=float).ravel()
    i = np.asarray(impostor, dtype=float).ravel()
    if len(g) == 0 or len(i) == 0:
        raise ValueError(
            f"Need both genuine ({len(g)}) and impostor ({len(i)}) scores to compute metrics"
        )

    eer, eer_threshold = equal_error_rate(g, i)
    tars: dict[float, float] = {}
    thresholds: dict[float, float] = {}
    for target in far_targets:
        tar, thr = tar_at_far(g, i, target)
        tars[target] = tar
        thresholds[target] = thr

    accepted_genuine = float((g >= eer_threshold).mean())
    rejected_impostor = float((i < eer_threshold).mean())
    accuracy = (accepted_genuine * len(g) + rejected_impostor * len(i)) / (len(g) + len(i))

    resolution = 1.0 / len(i)
    unresolvable = sorted(t for t in far_targets if t < resolution)

    return VerificationMetrics(
        n_genuine=len(g),
        n_impostor=len(i),
        eer=eer,
        eer_threshold=eer_threshold,
        auc=roc_auc(g, i),
        tar_at_far=tars,
        threshold_at_far=thresholds,
        accuracy_at_eer=accuracy,
        unresolvable_far_targets=unresolvable,
    )
