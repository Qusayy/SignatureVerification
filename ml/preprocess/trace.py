"""Recording what the pipeline did, stage by stage, for display.

A verification score is a single number produced by a dozen transformations. If
the only thing on screen is the number, the operator has to take it on trust,
and the first question anyone asks — *why?* — has no answer.

This module collects the intermediate image at each step, plus the handful of
measurements that explain it, so the interface can replay the whole chain.

Design constraints:

* **Nothing here may change the result.** Tracing is strictly observational: a
  disabled trace and an enabled one must produce byte-identical output from
  every function that accepts one. Every call site is therefore
  ``trace.add(...)`` with no return value used.
* **Off by default.** The training loop runs this pipeline millions of times
  and must not pay for image copies it will never look at.
* **Images are shrunk on capture, not on display.** A full-page scan is several
  megabytes; twelve of them per verification is not something to push through
  an encrypted store and down a browser connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

__all__ = ["Stage", "PipelineTrace", "vector_strip"]

# Widest edge of a captured stage image. Large enough that stroke detail is
# still visible on a laptop screen, small enough that a full trace is a few
# hundred kilobytes rather than tens of megabytes.
MAX_STAGE_EDGE = 560


@dataclass
class Stage:
    """One step of the pipeline, ready to render."""

    key: str
    title: str
    caption: str
    kind: str = "image"  # image | vector | compare | score
    image: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Some stages are best shown with black ink on white (photographic) and
    # some are masks the model sees ink-high on black. The UI needs to know
    # which, or half the filmstrip looks like a photographic negative.
    invert_for_display: bool = False


def _shrink(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_STAGE_EDGE:
        return image.copy()
    scale = MAX_STAGE_EDGE / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


class PipelineTrace:
    """Collects stages. A disabled trace is an inert sink."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stages: list[Stage] = []

    def __bool__(self) -> bool:
        return self.enabled

    def add(
        self,
        key: str,
        title: str,
        caption: str,
        *,
        image: np.ndarray | None = None,
        kind: str = "image",
        invert_for_display: bool = False,
        **metrics: Any,
    ) -> None:
        if not self.enabled:
            return
        captured = None
        if image is not None and image.size:
            captured = _shrink(image)
        self.stages.append(
            Stage(
                key=key,
                title=title,
                caption=caption,
                kind=kind,
                image=captured,
                metrics={k: v for k, v in metrics.items() if v is not None},
                invert_for_display=invert_for_display,
            )
        )

    def __len__(self) -> int:
        return len(self.stages)


def vector_strip(vector: np.ndarray, *, height: int = 96, width: int = 512) -> np.ndarray:
    """Render an embedding as a diverging colour strip.

    The embedding is the one stage with nothing to photograph — it is a point
    on a unit sphere in a few hundred dimensions. Drawing it as a strip is not
    an analogy; each column *is* one dimension, blue for negative and amber for
    positive, and two signatures by the same hand produce visibly similar
    strips. That is the whole claim of the system, made visible in one image.
    """
    values = np.asarray(vector, dtype=np.float32).ravel()
    if values.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    limit = float(np.abs(values).max()) or 1.0
    normalised = np.clip(values / limit, -1.0, 1.0)

    # One column per dimension, then stretched to the requested width so the
    # strip stays readable whatever the embedding dimensionality.
    row = np.zeros((1, values.size, 3), dtype=np.uint8)
    positive = normalised > 0
    magnitude = np.abs(normalised)

    # BGR. Amber for positive, blue for negative, near-black at zero.
    row[0, positive, 0] = (40 * magnitude[positive]).astype(np.uint8)
    row[0, positive, 1] = (170 * magnitude[positive]).astype(np.uint8)
    row[0, positive, 2] = (255 * magnitude[positive]).astype(np.uint8)
    row[0, ~positive, 0] = (255 * magnitude[~positive]).astype(np.uint8)
    row[0, ~positive, 1] = (150 * magnitude[~positive]).astype(np.uint8)
    row[0, ~positive, 2] = (60 * magnitude[~positive]).astype(np.uint8)

    return cv2.resize(row, (width, height), interpolation=cv2.INTER_NEAREST)
