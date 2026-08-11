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
    # Fraction of the image's handwriting-like ink that falls inside the box.
    # Well below 1.0 means the detector is throwing ink away. On a printed form
    # that is correct and expected — the discarded ink is the form. On an image
    # that is *already* a cropped signature it means part of the signature was
    # cut off, which produces a confident score computed from half a signature.
    ink_captured: float = 1.0

    @property
    def is_partial(self) -> bool:
        """True when enough ink was excluded to be worth telling somebody."""
        return self.ink_captured < 0.9

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
            "ink_captured": round(self.ink_captured, 4),
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


def _ink_mask(page: np.ndarray) -> np.ndarray:
    gray = to_grayscale(page)
    flattened = normalize_illumination(gray)
    return remove_form_lines(sauvola_binarize(flattened))


def _has_printed_structure(page: np.ndarray, binary: np.ndarray) -> bool:
    """Does this image contain a printed form, or only handwriting?

    The distinction decides how the detector should behave, and the two cases
    want opposite things:

    * On a **form**, most of the ink is not the signature. Cropping tightly is
      the entire job, and merging distant blobs would swallow the form.
    * On an image that is **already a cropped signature**, there is nothing to
      exclude. Any cropping can only remove part of the signature, and the
      resulting fragment still scores plausibly — which is how a wrong crop
      goes unnoticed.

    Two signals, either sufficient: printed rules were removed during masking,
    or some blob is dense enough to be printed text rather than handwriting.
    """
    raw = sauvola_binarize(normalize_illumination(to_grayscale(page)))
    raw_ink = int(np.count_nonzero(raw))
    if raw_ink and (raw_ink - int(np.count_nonzero(binary))) > raw_ink * 0.05:
        return True  # printed rules were stripped, so there was a form

    page_h, page_w = binary.shape
    for x, y, w, h in _components(binary):
        # Skip specks only. The height floor has to stay low: a line of printed
        # text is a wide, *thin* component, and a floor set as a fraction of
        # page height skips exactly the thing being looked for.
        if w < max(20, page_w * 0.03) or h < max(6, page_h * 0.008):
            continue
        region = binary[y : y + h, x : x + w]
        fill = float(np.count_nonzero(region)) / max(w * h, 1)
        # Glyphs pack densely into their box; handwriting leaves whitespace.
        if fill > 0.35:
            return True
    return False


def _components(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of every ink blob, grouped by proximity along a line."""
    page_h, page_w = binary.shape
    # Join strokes and letters that belong to one signature, without bridging
    # to neighbouring form fields. The kernel is wide and short because
    # handwriting connects horizontally.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, page_w // 45), max(3, page_h // 180))
    )
    grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, _labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    return [
        (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        for i in range(1, count)
    ]


def _gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    """Horizontal and vertical separation between two boxes; 0 where they overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return dx, dy


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def _absorb_neighbours(
    seed: tuple[int, int, int, int],
    parts: list[tuple[int, int, int, int]],
    binary: np.ndarray,
) -> tuple[int, int, int, int]:
    """Grow the seed box to include disconnected pieces of the same signature.

    A signature is very often not one connected component. A detached flourish,
    a dotted 'i', initials written apart from the surname, and most Arabic
    hands all break into several blobs. Taking only the highest-scoring blob
    therefore crops part of the signature away — and the result is not an
    error, it is a confident score computed from a fragment.

    Pieces are absorbed when they sit within roughly one signature-height of
    the current box, which is how far apart parts of one signature fall, and
    much closer than the next field on a form. Dense blocks are refused
    outright: those are printed text, and bridging into them would swallow the
    form.
    """
    page_h, page_w = binary.shape
    box = seed

    for _ in range(len(parts) + 1):
        grown = box
        for part in parts:
            if part == box or _gap(box, part) == (0, 0) and _union(box, part) == box:
                continue

            scale = max(box[3], part[3])
            dx, dy = _gap(box, part)
            if dx > scale * 1.1 or dy > scale * 0.8:
                continue

            px, py, pw, ph = part
            region = binary[py : py + ph, px : px + pw]
            fill = float(np.count_nonzero(region)) / max(pw * ph, 1)
            # Printed text packs densely into its box; handwriting does not.
            if fill > 0.35 and pw > page_w * 0.12:
                continue

            merged = _union(grown, part)
            # Never let absorption run away into the whole page.
            if merged[2] > page_w * 0.97 and merged[3] > page_h * 0.85:
                continue
            grown = merged

        if grown == box:
            break
        box = grown

    return box


def detect_candidates(page: np.ndarray, *, max_candidates: int = 5) -> list[Detection]:
    """Return ranked candidate signature regions on a scanned form."""
    binary = _ink_mask(page)

    page_h, page_w = binary.shape

    candidates: list[Detection] = []
    for x, y, w, h in _components(binary):
        # Score the *original* ink inside the box, not the dilated blob, so
        # fill ratio reflects real stroke density.
        region = binary[y : y + h, x : x + w]
        ink_area = int(np.count_nonzero(region))
        score = _candidate_score(region, (x, y, w, h), ink_area, (page_h, page_w))
        if score > 0:
            candidates.append(Detection((x, y, w, h), score))

    candidates.sort(key=lambda d: d.confidence, reverse=True)
    return candidates[:max_candidates]


def _handwriting_mask(binary: np.ndarray) -> np.ndarray:
    """The ink that is plausibly handwriting, with printed text blanked out."""
    page_h, page_w = binary.shape
    mask = binary.copy()
    for x, y, w, h in _components(binary):
        if w < max(20, page_w * 0.03) or h < max(6, page_h * 0.008):
            continue
        region = binary[y : y + h, x : x + w]
        if float(np.count_nonzero(region)) / max(w * h, 1) > 0.35:
            mask[y : y + h, x : x + w] = 0
    return mask


def _ink_fraction_inside(binary: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """How much of the *handwriting* the box keeps.

    Measured against handwriting rather than all ink on purpose. On a printed
    form the detector correctly discards the overwhelming majority of the ink,
    so a fraction computed over everything reads as catastrophic on every
    single form and the warning it feeds becomes noise nobody looks at. What
    actually matters is whether handwriting was left outside the box.
    """
    handwriting = _handwriting_mask(binary)
    total = int(np.count_nonzero(handwriting))
    if total == 0:
        return 1.0
    x, y, w, h = box
    inside = int(np.count_nonzero(handwriting[max(0, y) : y + h, max(0, x) : x + w]))
    return inside / total


def detect_signature(page: np.ndarray) -> Detection | None:
    """Locate the signature on an image, or None if unsure.

    Returning None matters: a wrong crop scored confidently is worse than
    asking the employee to draw the box themselves, which the UI supports.

    The returned box is the *whole* signature, not the largest piece of it.
    Selecting a single connected component was a real defect: an image
    containing nothing but a signature would come back cropped to one blob of
    it, and the fragment then scored perfectly normally against the stored
    specimens — high enough to pass, because part of a signature does resemble
    the whole of it. Nothing in the output revealed that most of the ink had
    been thrown away.
    """
    binary = _ink_mask(page)
    parts = _components(binary)
    if not parts:
        return None
    page_h, page_w = binary.shape

    # No form, no cropping. Everything here is signature, so take all of it.
    if not _has_printed_structure(page, binary):
        box = parts[0]
        for part in parts[1:]:
            box = _union(box, part)
        return Detection(box, 1.0, method="whole-image", ink_captured=1.0)

    candidates = detect_candidates(page)
    if not candidates or candidates[0].confidence < 0.35:
        return None

    best = candidates[0]
    box = _absorb_neighbours(best.bbox, parts, binary)
    return Detection(box, best.confidence, ink_captured=_ink_fraction_inside(binary, box))
