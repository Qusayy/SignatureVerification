"""Explainability for the employee screen.

A bare number will not be trusted by a operator and will not be signed off by a
risk function. Three artefacts ship alongside every score:

* **Difference overlay** — query and specimen drawn in two colours over each
  other, so divergence is visible at a glance. This is the one employees
  actually use; it is the digital version of what they already do by eye.
* **Attention heatmap** — Grad-CAM against the *similarity to the specimen*,
  answering "which strokes drove this score" rather than the meaningless
  "which strokes look like a signature".
* **Plain-language reason** — one sentence, no jargon, no false precision.

None of these justify the score after the fact; they are computed from the same
forward pass that produced it.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.preprocess.pipeline import to_model_input
from ml.scoring.calibrate import Band
from ml.scoring.verifier import VerificationResult

__all__ = ["difference_overlay", "attention_heatmap", "reason_text", "side_by_side"]

# BGR, chosen to stay distinguishable for the most common colour vision
# deficiencies: blue/orange rather than the obvious red/green.
QUERY_COLOUR = (200, 120, 0)  # blue
REFERENCE_COLOUR = (0, 140, 235)  # orange
AGREEMENT_COLOUR = (70, 70, 70)  # dark grey where the two coincide


def _align(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Refine the alignment of the reference onto the query by translation only.

    Preprocessing already centres both on their centre of mass, so what remains
    is a small residual shift. Only translation is corrected: allowing rotation
    or scale here would make a forgery look better aligned than it is, which is
    the opposite of what the overlay is for.
    """
    if query.shape != reference.shape:
        reference = cv2.resize(reference, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_NEAREST)

    a = query.astype(np.float32) / 255.0
    b = reference.astype(np.float32) / 255.0
    if a.std() < 1e-6 or b.std() < 1e-6:
        return reference

    (dx, dy), _ = cv2.phaseCorrelate(b, a)
    # Ignore implausibly large shifts — those mean the correlation failed, not
    # that the signature moved half a canvas.
    limit = 0.1 * min(query.shape)
    if abs(dx) > limit or abs(dy) > limit:
        return reference

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(reference, matrix, (query.shape[1], query.shape[0]), flags=cv2.INTER_NEAREST)


def difference_overlay(
    query_canvas: np.ndarray,
    reference_canvas: np.ndarray,
    *,
    align: bool = True,
    thickness: int = 2,
    trim: bool = True,
) -> np.ndarray:
    """Render query and specimen over each other in contrasting colours.

    Returns a BGR image on a white background: query-only strokes in blue,
    specimen-only strokes in orange, agreement in grey.

    Args:
        trim: crop the result to the strokes. The working canvas is mostly
            empty space by design, and displaying it untrimmed shrinks the
            signature to a fraction of the panel — which is the one thing the
            employee actually needs to look at.
    """
    reference = _align(query_canvas, reference_canvas) if align else reference_canvas
    if reference.shape != query_canvas.shape:
        reference = cv2.resize(
            reference, (query_canvas.shape[1], query_canvas.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Dilate both so thin strokes remain visible when the canvas is scaled down
    # for display, and so "agreement" tolerates a pixel or two of jitter.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1,) * 2)
    q = cv2.dilate((query_canvas > 0).astype(np.uint8), kernel) > 0
    r = cv2.dilate((reference > 0).astype(np.uint8), kernel) > 0

    canvas = np.full((*query_canvas.shape, 3), 255, dtype=np.uint8)
    canvas[q & ~r] = QUERY_COLOUR
    canvas[r & ~q] = REFERENCE_COLOUR
    canvas[q & r] = AGREEMENT_COLOUR

    if trim:
        ink = (q | r).astype(np.uint8)
        coords = cv2.findNonZero(ink)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad_x, pad_y = int(w * 0.05) + 8, int(h * 0.12) + 8
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1 = min(canvas.shape[1], x + w + pad_x)
            y1 = min(canvas.shape[0], y + h + pad_y)
            canvas = canvas[y0:y1, x0:x1]

    return canvas


def side_by_side(query_canvas: np.ndarray, reference_canvas: np.ndarray, *, gap: int = 12) -> np.ndarray:
    """Stack query above specimen with a separator, both on white."""
    h, w = query_canvas.shape
    reference = cv2.resize(reference_canvas, (w, h), interpolation=cv2.INTER_NEAREST)
    separator = np.full((gap, w), 220, dtype=np.uint8)
    # Invert so ink reads as dark on white, the way a person expects.
    stacked = np.vstack([255 - query_canvas, separator, 255 - reference])
    return cv2.cvtColor(stacked, cv2.COLOR_GRAY2BGR)


def _last_conv(model: nn.Module) -> nn.Module:
    conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv = module
    if conv is None:
        raise ValueError("Model has no convolutional layer to attach Grad-CAM to")
    return conv


def attention_heatmap(
    model: nn.Module,
    query_canvas: np.ndarray,
    reference_embedding: np.ndarray,
    *,
    device: str = "cpu",
    alpha: float = 0.55,
) -> np.ndarray:
    """Grad-CAM against similarity to the specimen.

    The target is the cosine similarity between the query embedding and the
    stored specimen embedding, so the map answers "which strokes made this look
    like (or unlike) the customer's signature".

    Returns a BGR image: the signature with a heat overlay.
    """
    model = model.to(device)
    was_training = model.training
    model.eval()

    target_layer = _last_conv(model)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        tensor = torch.from_numpy(to_model_input(query_canvas)).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        embedding = F.normalize(model(tensor), p=2, dim=1)
        reference = F.normalize(
            torch.from_numpy(np.asarray(reference_embedding, dtype=np.float32)).view(1, -1), p=2, dim=1
        ).to(device)
        similarity = (embedding * reference).sum()

        model.zero_grad(set_to_none=True)
        similarity.backward()

        acts = activations["value"].detach()[0]  # (C, H, W)
        grads = gradients["value"].detach()[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=0)).cpu().numpy()
    finally:
        handle.remove()
        if was_training:
            model.train()

    if cam.max() <= 0:
        cam = np.zeros_like(cam)
    else:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (query_canvas.shape[1], query_canvas.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(255 - query_canvas, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heat, alpha, base, 1 - alpha, 0)


def reason_text(result: VerificationResult) -> str:
    """One plain sentence explaining the score. No jargon, no false precision."""
    n = result.comparison.n_references
    specimens = "the stored specimen" if n == 1 else f"all {n} stored specimens"

    if result.suspected_copy:
        return (
            "This appears to be a copy of the stored specimen rather than a freshly written "
            "signature — a genuine signature is never an exact match. Treat as suspicious."
        )

    spread = result.comparison.max_similarity - result.comparison.min_similarity
    parts: list[str] = []

    if result.band is Band.GREEN:
        parts.append(f"Stroke shape and proportions are consistent with {specimens}.")
    elif result.band is Band.RED:
        parts.append(f"Stroke shape and proportions differ clearly from {specimens}.")
    else:
        parts.append(f"Partial agreement with {specimens} — some features match, others do not.")

    if n > 1 and spread > 0.15:
        parts.append(
            "The stored specimens are themselves inconsistent, which widens the margin of doubt."
        )
    if n == 1:
        parts.append("Only one specimen is on file, so confidence is lower than usual.")
    if not result.calibrated:
        parts.append("Score is uncalibrated and should be treated as indicative only.")
    if "no_cohort_normalisation" in result.warnings:
        parts.append("Cohort normalisation is unavailable, so this score is not comparable across customers.")

    return " ".join(parts)
