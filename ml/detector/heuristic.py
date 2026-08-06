"""Heuristic signature locator — Stage A without a trained model.

Purpose: the demo, and the cold start before the organisation has annotated any forms.
A learned detector (``ml/detector/train.py``) will beat this once there is
training data; until then this is what makes the whole pipeline runnable.

The idea is that handwriting is *structurally* different from everything else
on a printed form, in ways that survive without any learning:

* **Printed rules** are long, straight, and thin. Removed outright.
* **Printed text** sits in tight horizontal bands with a regular baseline and a
  high fill ratio — glyphs pack densely into their bounding box.
* **A signature** is sparse inside its bounding box (mostly whitespace between
  strokes), has strokes running in many directions, and is usually the largest
  such structure in the lower part of the page.

Fill ratio does most of the work: a printed text block typically fills 25-45%
of its box, a signature 5-15%.

Deliberately unlicensed by anything: no AGPL detector, no third-party weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ml.preprocess.pipeline import (
    normalize_illumination,
    remove_form_lines,
    sauvola_binarize,
    to_grayscale,
)

__all__ = ["Detection", "detect_signature", "detect_candidates"]


@dataclass
class Detection:
    """A located signature region."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float  # heuristic score in [0, 1], not a probability
    method: str = "heuristic"

    def crop(self, image: np.ndarray, pad_ratio: float = 0.06) -> np.ndarray:
        x, y, w, h = self.bbox
        px, py = int(w * pad_ratio), int(h * pad_ratio)
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(image.shape[1], x + w + px)
        y1 = min(image.shape[0], y + h + py)
        return image[y0:y1, x0:x1]

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "confidence": round(self.confidence, 4),
            "method": self.method,
        }


def _candidate_score(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    area: int,
    page_shape: tuple[int, int],
) -> float:
    """Score how signature-like a connected component is. Higher is better."""
    x, y, w, h = bbox
    page_h, page_w = page_shape

    if w < page_w * 0.06 or h < page_h * 0.015:
        return 0.0  # too small to be a signature on this page
    if w > page_w * 0.95 and h > page_h * 0.8:
        return 0.0  # the whole page — background leaked through

    fill = area / max(w * h, 1)
    aspect = w / max(h, 1)

    # Sparse inside its box. Printed text blocks are much denser.
    fill_score = float(np.clip((0.32 - fill) / 0.25, 0.0, 1.0))

    # Signatures are wider than tall, but not a thin ribbon like a text line.
    aspect_score = float(np.exp(-((aspect - 3.0) ** 2) / (2 * 2.2**2)))

    # Stroke direction diversity: printed text is dominated by horizontal and
    # vertical strokes; handwriting is not.
    gx = cv2.Sobel(component.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(component.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    strong = magnitude > magnitude.max() * 0.2 if magnitude.max() > 0 else np.zeros_like(magnitude, bool)
    if strong.sum() < 20:
        return 0.0
    angles = np.arctan2(gy[strong], gx[strong]) % np.pi
    histogram, _ = np.histogram(angles, bins=12, range=(0, np.pi))
    distribution = histogram / histogram.sum()
    entropy = float(-(distribution * np.log(distribution + 1e-12)).sum() / np.log(12))

    # Signature boxes sit low on a form far more often than not.
    vertical_score = float(np.clip((y + h / 2) / page_h, 0.0, 1.0)) ** 0.5

    # Size matters, but saturates: the signature need only be big enough.
    size_score = float(np.clip((w * h) / (page_w * page_h * 0.04), 0.0, 1.0))

    return float(
        0.30 * fill_score
        + 0.20 * aspect_score
        + 0.25 * entropy
        + 0.10 * vertical_score
        + 0.15 * size_score
    )


def detect_candidates(page: np.ndarray, *, max_candidates: int = 5) -> list[Detection]:
    """Return ranked candidate signature regions on a scanned form."""
    gray = to_grayscale(page)
    flattened = normalize_illumination(gray)
    binary = sauvola_binarize(flattened)
    binary = remove_form_lines(binary)

    page_h, page_w = binary.shape

    # Join strokes and letters that belong to one signature, without bridging
    # to neighbouring form fields. The kernel is wide and short because
    # handwriting connects horizontally.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, page_w // 45), max(3, page_h // 180))
    )
    grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    candidates: list[Detection] = []
    for i in range(1, count):
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        # Score the *original* ink inside the box, not the dilated blob, so
        # fill ratio reflects real stroke density.
        region = binary[y : y + h, x : x + w]
        ink_area = int(np.count_nonzero(region))
        score = _candidate_score(region, (int(x), int(y), int(w), int(h)), ink_area, (page_h, page_w))
        if score > 0:
            candidates.append(Detection((int(x), int(y), int(w), int(h)), score))

    candidates.sort(key=lambda d: d.confidence, reverse=True)
    return candidates[:max_candidates]


def detect_signature(page: np.ndarray) -> Detection | None:
    """Locate the most signature-like region on a form, or None if unsure.

    Returning None matters: a wrong crop scored confidently is worse than
    asking the employee to draw the box themselves, which the UI supports.
    """
    candidates = detect_candidates(page)
    if not candidates:
        return None
    best = candidates[0]
    return best if best.confidence >= 0.35 else None
