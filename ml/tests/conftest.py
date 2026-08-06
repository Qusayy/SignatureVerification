"""Shared helpers for preprocessing tests.

Comparing two binarised signatures needs care. Strokes are only a few pixels
wide, so plain binary IoU collapses toward zero under a one- or two-pixel
shift even when the shapes are identical. Every comparison here therefore
measures either *geometry* (ink spread, extent) or *tolerant* overlap.
"""

from __future__ import annotations

import cv2
import numpy as np

from ml.preprocess.pipeline import ink_rms_radius

__all__ = ["canvas_geometry", "tolerant_overlap"]


def canvas_geometry(canvas: np.ndarray) -> tuple[float, int, int]:
    """(RMS ink radius, ink width, ink height) of a preprocessed canvas."""
    ys, xs = np.nonzero(canvas)
    if len(xs) == 0:
        return (0.0, 0, 0)
    return (
        ink_rms_radius(canvas),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def tolerant_overlap(a: np.ndarray, b: np.ndarray, tolerance: int = 9) -> float:
    """Fraction of ink in each image lying within ``tolerance`` px of the other.

    Answers "do the strokes land in the same place", which is the property that
    matters, rather than "are the pixels identical", which they never are.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance))
    a_ink, b_ink = a > 0, b > 0
    total = np.count_nonzero(a_ink) + np.count_nonzero(b_ink)
    if total == 0:
        return 0.0
    a_near = cv2.dilate(a_ink.astype(np.uint8), kernel) > 0
    b_near = cv2.dilate(b_ink.astype(np.uint8), kernel) > 0
    return (np.count_nonzero(a_ink & b_near) + np.count_nonzero(b_ink & a_near)) / total
