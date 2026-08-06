"""Capture-condition augmentation.

Augmentation here simulates the **capture channel**, not the signature. It is
applied to the raw image *before* preprocessing, so the model trains on exactly
what the live pipeline will hand it: blurred scans, phone photographs at an
angle, ink that bled, a rule running through the strokes.

What is deliberately *not* augmented: anything that changes the identity of the
signature itself. Large rotations, heavy shear, and elastic warps make a
genuine signature look like a forgery, and training through them teaches the
model to ignore the very cues it needs. Rotation stays within the few degrees a
scanner introduces, and there is no elastic deformation at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["AugmentConfig", "augment_capture", "apply_ruled_line", "apply_stamp"]


@dataclass(frozen=True)
class AugmentConfig:
    """Probabilities and magnitudes for each simulated capture effect."""

    p_rotate: float = 0.6
    max_rotate_deg: float = 2.5  # scanner/feeder skew only, never signature slant

    # Uniform scale is now normalised away by preprocessing, so this augments
    # the *capture* channel (scanner DPI, camera distance) rather than teaching
    # scale invariance. The range is wide because it costs nothing and proves
    # the normalisation holds across everything the model sees in training.
    p_scale: float = 0.7
    scale_range: tuple[float, float] = (0.6, 1.6)

    # Aspect jitter is what actually survives normalisation, so it is the
    # augmentation that matters now. Kept small: stretch a signature too far
    # and it stops being that person's signature.
    p_aspect: float = 0.5
    max_aspect_jitter: float = 0.08

    # Slight shear from paper lying askew under the pen. Small, for the same
    # reason rotation is small: slant is the writer's, not noise.
    p_shear: float = 0.35
    max_shear_deg: float = 3.0

    p_pen_width: float = 0.5  # a different pen on the day
    p_blur: float = 0.4
    max_blur_sigma: float = 1.4

    p_perspective: float = 0.3  # phone photographed at an angle
    max_perspective: float = 0.025

    p_illumination: float = 0.4
    p_noise: float = 0.4
    max_noise_sigma: float = 6.0

    p_jpeg: float = 0.3
    jpeg_quality_range: tuple[int, int] = (35, 85)

    p_ruled_line: float = 0.35
    p_stamp: float = 0.12
    p_ink_bleed: float = 0.2


DEFAULT_AUGMENT = AugmentConfig()


def apply_ruled_line(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a printed rule across the signature, as on a real form."""
    out = image.copy()
    h, w = out.shape[:2]
    y = int(rng.uniform(0.45, 0.85) * h)
    tone = int(rng.integers(110, 185))
    thickness = int(rng.integers(1, 3))
    cv2.line(out, (0, y), (w, y), tone, thickness)
    if rng.random() < 0.3:  # a second rule below
        cv2.line(out, (0, min(h - 1, y + int(rng.integers(20, 60)))), (w, min(h - 1, y + int(rng.integers(20, 60)))), tone, 1)
    return out


def apply_stamp(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay a semi-transparent round stamp partially covering the signature."""
    out = image.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    centre = (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h))
    radius = int(min(h, w) * rng.uniform(0.25, 0.45))
    tone = int(rng.integers(70, 140))
    cv2.circle(overlay, centre, radius, tone, int(rng.integers(2, 5)))
    cv2.circle(overlay, centre, int(radius * 0.75), tone, 1)
    alpha = float(rng.uniform(0.3, 0.6))
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def _perspective(image: np.ndarray, rng: np.random.Generator, magnitude: float) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = magnitude * min(h, w)
    dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    border = int(np.percentile(image, 90))
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def augment_capture(
    image: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig = DEFAULT_AUGMENT,
) -> np.ndarray:
    """Simulate one plausible capture of ``image``.

    Args:
        image: raw grayscale signature, dark ink on a light background.

    Returns a new image in the same convention. Feed the result through
    :func:`ml.preprocess.pipeline.preprocess_signature` exactly as the live
    pipeline would.
    """
    out = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = out.copy()
    background = int(np.percentile(out, 90))

    if rng.random() < cfg.p_scale:
        s = float(rng.uniform(*cfg.scale_range))
        out = cv2.resize(
            out,
            (max(8, int(out.shape[1] * s)), max(8, int(out.shape[0] * s))),
            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
        )

    if rng.random() < cfg.p_aspect:
        # Stretch one axis only. Unlike uniform scale this survives moment
        # normalisation, so it is the augmentation that teaches the model
        # tolerance to a signature written a little wide or a little cramped.
        jitter = 1.0 + float(rng.uniform(-cfg.max_aspect_jitter, cfg.max_aspect_jitter))
        if rng.random() < 0.5:
            out = cv2.resize(out, (max(8, int(out.shape[1] * jitter)), out.shape[0]))
        else:
            out = cv2.resize(out, (out.shape[1], max(8, int(out.shape[0] * jitter))))

    if rng.random() < cfg.p_shear:
        shear = math.tan(math.radians(float(rng.uniform(-cfg.max_shear_deg, cfg.max_shear_deg))))
        h, w = out.shape
        matrix = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        out = cv2.warpAffine(
            out, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=background,
        )

    if rng.random() < cfg.p_pen_width:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Ink is dark, so eroding the image thickens the stroke.
        out = cv2.erode(out, k) if rng.random() < 0.5 else cv2.dilate(out, k)

    if rng.random() < cfg.p_ink_bleed:
        # Bleed on absorbent paper: blur then re-darken, so strokes spread
        # without the whole image going grey.
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=float(rng.uniform(0.8, 2.0)))
        out = np.minimum(out, blurred)

    if rng.random() < cfg.p_ruled_line:
        out = apply_ruled_line(out, rng)

    if rng.random() < cfg.p_stamp:
        out = apply_stamp(out, rng)

    if rng.random() < cfg.p_rotate:
        angle = float(rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg))
        h, w = out.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(
            out, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=background,
        )

    if rng.random() < cfg.p_perspective:
        out = _perspective(out, rng, float(rng.uniform(0.005, cfg.max_perspective)))

    if rng.random() < cfg.p_illumination:
        h, w = out.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gradient = 1.0 - float(rng.uniform(0.05, 0.30)) * (
            (xx / w) * float(rng.uniform(-1, 1)) + (yy / h) * float(rng.uniform(-1, 1))
        )
        out = np.clip(out.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    if rng.random() < cfg.p_blur:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(rng.uniform(0.4, cfg.max_blur_sigma)))

    if rng.random() < cfg.p_noise:
        sigma = float(rng.uniform(1.5, cfg.max_noise_sigma))
        noise = rng.normal(0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < cfg.p_jpeg:
        out = _jpeg(out, int(rng.integers(*cfg.jpeg_quality_range)))

    return out
