"""Signature preprocessing.

Turns a raw scan or phone photograph of a signature into the fixed-geometry,
background-free binary image the embedding model expects.

Two design decisions worth understanding before changing anything here:

1. **Page skew is corrected; signature slant is not.**
   A photographed form may be rotated by a few degrees, and that rotation is
   noise. The writer's own baseline slant, however, is one of the more stable
   discriminative features of a signature. Naively deskewing a *cropped*
   signature with ``minAreaRect`` destroys it and measurably hurts accuracy.
   So skew is estimated from the printed structure of the **whole page** via
   :func:`estimate_page_skew` and applied before cropping. The crop path never
   rotates.

2. **Ink is centred on a large canvas, then resized.**
   Resizing straight to the network input distorts wide signatures and
   destroys the relative stroke scale. Centring on a fixed canvas first keeps
   aspect ratio and stroke thickness comparable across writers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ml.config import PREPROCESS, PreprocessConfig

__all__ = [
    "PreprocessResult",
    "preprocess_signature",
    "estimate_page_skew",
    "deskew_page",
    "normalize_illumination",
    "remove_form_lines",
    "sauvola_binarize",
    "ink_bounding_box",
    "center_on_canvas",
    "to_model_input",
    "BlankSignatureError",
]


class BlankSignatureError(ValueError):
    """Raised when a crop contains too little ink to be scored.

    Scoring noise produces a confident-looking number from nothing, which is
    worse than refusing. The API surfaces this to the employee as "no signature
    detected — rescan" rather than as a low score.
    """


@dataclass
class PreprocessResult:
    """Output of the preprocessing pipeline plus diagnostics for the UI."""

    # Canvas-sized binary image. Background 0, ink 255 (inverted relative to
    # the input, matching the SigNet convention).
    image: np.ndarray
    ink_fraction: float
    source_bbox: tuple[int, int, int, int]  # (x, y, w, h) of ink in the input crop
    warnings: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.ink_fraction >= PREPROCESS.min_ink_fraction


# --------------------------------------------------------------------------
# Grayscale / illumination
# --------------------------------------------------------------------------


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Accept BGR, BGRA, or already-gray input and return uint8 grayscale."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def normalize_illumination(gray: np.ndarray, kernel_frac: float = 0.10) -> np.ndarray:
    """Flatten uneven lighting from phone photographs and thick scanner lids.

    Estimates the page background with a large morphological closing (which
    swallows thin dark strokes but keeps slow illumination gradients), then
    divides it out. Without this step, a shadow across one half of the page
    reads as ink after thresholding.
    """
    h, w = gray.shape
    k = max(15, int(min(h, w) * kernel_frac) | 1)  # odd, at least 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=k / 6.0)
    background = np.maximum(background, 1)

    flattened = (gray.astype(np.float32) / background.astype(np.float32)) * 255.0
    return np.clip(flattened, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Page skew (applied to the full page, before cropping)
# --------------------------------------------------------------------------


def estimate_page_skew(
    page_gray: np.ndarray,
    max_angle: float = 15.0,
    *,
    min_segments: int = 6,
    max_dispersion: float = 2.0,
) -> float:
    """Estimate page skew from the printed structure of a form.

    Uses the long straight edges a printed form provides (rules, table borders,
    field boxes). Returns **the rotation to apply in order to correct the
    page**, in degrees, ready to hand to :func:`cv2.getRotationMatrix2D`.
    Concretely: a page rotated counter-clockwise by +4 degrees has printed
    rules measuring about -4 degrees, and -4 is exactly the correction needed,
    so the measured median is returned unchanged.

    Returns 0.0 whenever the evidence is weak. Two guards matter:

    * at least ``min_segments`` qualifying segments, and
    * those segments must *agree*: printed rules on a skewed page all share one
      angle, whereas the strokes of a signature scatter. Without the
      dispersion check a bare signature yields a confident-looking but
      meaningless angle, and the page gets rotated for no reason.
    """
    edges = cv2.Canny(page_gray, 50, 150, apertureSize=3)
    min_len = int(min(page_gray.shape) * 0.35)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=100,
        minLineLength=max(min_len, 60),
        maxLineGap=8,
    )
    if lines is None or len(lines) == 0:
        return 0.0

    # OpenCV 4.x returns (N, 1, 4); 5.x returns (N, 4). Normalise both.
    segments = np.asarray(lines).reshape(-1, 4)

    angles: list[float] = []
    for x1, y1, x2, y2 in segments:
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        # Fold near-vertical lines onto the horizontal frame of reference.
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) <= max_angle:
            angles.append(angle)

    if len(angles) < min_segments:
        return 0.0

    arr = np.asarray(angles)
    median = float(np.median(arr))
    dispersion = float(np.median(np.abs(arr - median)))
    if dispersion > max_dispersion:
        return 0.0  # segments disagree — no reliable page structure

    return median


def deskew_page(image: np.ndarray, angle: float | None = None) -> tuple[np.ndarray, float]:
    """Rotate a full page to remove scan/photo skew.

    Args:
        angle: correction to apply, in the sense returned by
            :func:`estimate_page_skew`. Estimated from the image when omitted.

    Returns the corrected image and the angle applied. A no-op for angles below
    0.2 degrees, where rotation costs interpolation blur and buys nothing.
    """
    gray = to_grayscale(image)
    if angle is None:
        angle = estimate_page_skew(gray)
    if abs(angle) < 0.2:
        return image, 0.0

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    border = int(np.median(gray[gray > np.percentile(gray, 60)])) if gray.size else 255
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border, border, border) if image.ndim == 3 else border,
    )
    return rotated, float(angle)


# --------------------------------------------------------------------------
# Printed line removal
# --------------------------------------------------------------------------


def remove_form_lines(binary: np.ndarray, cfg: PreprocessConfig = PREPROCESS) -> np.ndarray:
    """Remove printed ruled lines and form borders from a binary ink mask.

    Only structures that run for a large fraction of the image width or height
    *and* are thin are treated as printed lines. A signature stroke rarely
    satisfies both. Removed pixels are then dilated back into the strokes that
    cross them, so a signature written over a line does not end up severed.

    Input and output: uint8, ink 255 on background 0.
    """
    h, w = binary.shape
    min_h = max(12, int(w * cfg.line_removal_ratio))
    min_v = max(12, int(h * cfg.line_removal_ratio))

    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_h, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v))
    )
    lines = cv2.bitwise_or(horizontal, vertical)
    if not lines.any():
        return binary

    # Thin structures only: a thick band that survives the opening is more
    # likely a stamp or a heavy stroke than a printed rule.
    thin = cv2.morphologyEx(
        lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    lines = cv2.subtract(lines, cv2.subtract(thin, lines))

    cleaned = cv2.subtract(binary, lines)

    # Reconnect strokes that the removal cut in two.
    bridge = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return cv2.bitwise_and(bridge, cv2.bitwise_or(cleaned, cv2.bitwise_not(lines)))


# --------------------------------------------------------------------------
# Binarization
# --------------------------------------------------------------------------


def sauvola_binarize(gray: np.ndarray, cfg: PreprocessConfig = PREPROCESS) -> np.ndarray:
    """Local adaptive threshold (Sauvola) returning ink 255 on background 0.

    Sauvola beats a global Otsu threshold on scanned paper because paper tone
    varies across the sheet. Implemented directly with box filters rather than
    pulled from scikit-image: it is a handful of lines, avoids a heavy import
    in the inference path, and keeps the exact formula visible.

        T(x, y) = m(x, y) * (1 + k * (s(x, y) / R - 1)),  R = 128 for 8-bit
    """
    window = cfg.sauvola_window | 1  # force odd
    img = gray.astype(np.float32)

    mean = cv2.boxFilter(img, ddepth=-1, ksize=(window, window), normalize=True)
    mean_sq = cv2.boxFilter(img * img, ddepth=-1, ksize=(window, window), normalize=True)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)

    threshold = mean * (1.0 + cfg.sauvola_k * (std / 128.0 - 1.0))
    binary = (img < threshold).astype(np.uint8) * 255

    # Otsu fallback: on a nearly uniform crop Sauvola can latch onto paper
    # texture and return noise across the whole field.
    if binary.mean() > 200:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def denoise(binary: np.ndarray, min_component_frac: float = 0.0004) -> np.ndarray:
    """Drop specks: connected components far too small to be part of a stroke."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    min_area = max(4, int(binary.size * min_component_frac))
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    if not keep.any():  # everything is small — keep the largest rather than blank it
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep[1 + int(np.argmax(areas))] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Geometry normalization
# --------------------------------------------------------------------------


def ink_bounding_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Tight (x, y, w, h) bounding box of ink pixels."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return (0, 0, 0, 0)
    return cv2.boundingRect(coords)


def ink_rms_radius(binary: np.ndarray) -> float:
    """RMS distance of ink pixels from their centroid.

    A robust measure of how large a signature is. Unlike the bounding box, one
    long flourish barely moves it, because the flourish contributes few pixels
    relative to the body of the signature.
    """
    moments = cv2.moments(binary, binaryImage=True)
    if moments["m00"] <= 0:
        return 0.0
    # Normalised second-order central moments are the variances about the
    # centroid; their sum is the squared RMS radius.
    var_x = moments["mu20"] / moments["m00"]
    var_y = moments["mu02"] / moments["m00"]
    return float(np.sqrt(max(var_x + var_y, 0.0)))


def center_on_canvas(
    binary: np.ndarray, cfg: PreprocessConfig = PREPROCESS
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Normalise a signature's size, then centre it on the fixed canvas.

    Two properties, and the distinction between them is the whole point:

    * **Absolute size is removed.** The signature is rescaled so the RMS spread
      of its ink about the centroid is a fixed fraction of the canvas. A
      customer who signs larger or smaller than usual therefore lands in the
      same place. An earlier version deliberately preserved absolute size, on
      the theory that size distinguishes writers; in practice it made the model
      reject genuine signatures written 20% smaller than the stored specimen
      while accepting forgeries that happened to match its size. See
      :mod:`ml.eval.diagnostics`.
    * **Aspect ratio is preserved.** Both axes are scaled by the same factor,
      so a writer whose signature is genuinely long and thin still looks long
      and thin. Proportion is discriminative; overall size is mostly pen and
      mood.

    Centring uses the centre of mass rather than the bounding-box centre, for
    the same robustness reason as the RMS radius.
    """
    canvas_h, canvas_w = cfg.canvas_size
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    x, y, w, h = ink_bounding_box(binary)
    if w == 0 or h == 0:
        return canvas, (0, 0, 0, 0)

    pad_x = int(w * cfg.ink_pad_ratio)
    pad_y = int(h * cfg.ink_pad_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(binary.shape[1], x + w + pad_x)
    y1 = min(binary.shape[0], y + h + pad_y)
    cropped = binary[y0:y1, x0:x1]

    scale = 1.0
    if cfg.normalize_size:
        rms = ink_rms_radius(cropped)
        if rms > 1e-3:
            scale = (cfg.target_ink_rms_ratio * canvas_h) / rms
            scale = min(scale, cfg.max_upscale)

    # Whatever the moments ask for, the result must fit the canvas. Leave a
    # small margin so strokes are never flush against the edge.
    ch, cw = cropped.shape
    fit = min((canvas_h * 0.98) / (ch * scale), (canvas_w * 0.98) / (cw * scale), 1.0)
    scale *= fit

    if abs(scale - 1.0) > 1e-3:
        cropped = cv2.resize(
            cropped,
            (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        cropped = (cropped > 96).astype(np.uint8) * 255
        ch, cw = cropped.shape

    moments = cv2.moments(cropped, binaryImage=True)
    if moments["m00"] > 0:
        com_x = moments["m10"] / moments["m00"]
        com_y = moments["m01"] / moments["m00"]
    else:
        com_x, com_y = cw / 2.0, ch / 2.0

    offset_x = int(round(canvas_w / 2.0 - com_x))
    offset_y = int(round(canvas_h / 2.0 - com_y))
    offset_x = max(0, min(offset_x, canvas_w - cw))
    offset_y = max(0, min(offset_y, canvas_h - ch))

    canvas[offset_y : offset_y + ch, offset_x : offset_x + cw] = cropped
    return canvas, (x0, y0, x1 - x0, y1 - y0)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def preprocess_signature(
    image: np.ndarray,
    cfg: PreprocessConfig = PREPROCESS,
    *,
    strip_form_lines: bool = True,
    strict: bool = True,
) -> PreprocessResult:
    """Full pipeline for a cropped signature region.

    Args:
        image: BGR, BGRA, or grayscale crop containing a signature.
        cfg: geometry and threshold configuration.
        strip_form_lines: remove printed rules crossing the signature. Disable
            for already-clean specimen scans where there is nothing to remove.
        strict: raise :class:`BlankSignatureError` when the crop is effectively
            empty. Set False during bulk ingest to collect diagnostics instead.

    Note this function does **not** rotate. Page skew must be corrected on the
    full page with :func:`deskew_page` before cropping — see module docstring.
    """
    warnings: list[str] = []

    gray = to_grayscale(image)
    flattened = normalize_illumination(gray)
    binary = sauvola_binarize(flattened, cfg)

    if strip_form_lines:
        binary = remove_form_lines(binary, cfg)
    binary = denoise(binary)

    ink_fraction = float(np.count_nonzero(binary)) / float(binary.size)
    if ink_fraction < cfg.min_ink_fraction:
        if strict:
            raise BlankSignatureError(
                f"Crop contains {ink_fraction:.4%} ink, below the {cfg.min_ink_fraction:.4%} "
                "minimum. Rescan rather than score this."
            )
        warnings.append("blank_or_near_blank")
    if ink_fraction > 0.45:
        warnings.append("very_dark_crop_possible_background_leak")

    canvas, bbox = center_on_canvas(binary, cfg)
    return PreprocessResult(
        image=canvas, ink_fraction=ink_fraction, source_bbox=bbox, warnings=warnings
    )


def to_model_input(
    result: PreprocessResult | np.ndarray,
    cfg: PreprocessConfig = PREPROCESS,
    *,
    train: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert a canvas image to the network input tensor layout.

    Returns float32 of shape (1, crop_h, crop_w) scaled to [0, 1], ink high.
    Random crop during training, centre crop at inference — the SigNet recipe.
    """
    canvas = result.image if isinstance(result, PreprocessResult) else result
    resized = cv2.resize(
        canvas, (cfg.input_size[1], cfg.input_size[0]), interpolation=cv2.INTER_AREA
    )

    crop_h, crop_w = cfg.crop_size
    max_y = resized.shape[0] - crop_h
    max_x = resized.shape[1] - crop_w
    if train:
        rng = rng or np.random.default_rng()
        y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    else:
        y0 = max_y // 2
        x0 = max_x // 2
    cropped = resized[y0 : y0 + crop_h, x0 : x0 + crop_w]

    return (cropped.astype(np.float32) / 255.0)[np.newaxis, ...]
