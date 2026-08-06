"""Synthetic signature corpus generator.

Purpose: let the entire stack — preprocessing, training, scoring, evaluation,
API, and the employee demo — run end to end *before* the organisation's real data
collection has finished. It is scaffolding for plumbing and demos.

**This corpus is not a substitute for real data.** Accuracy measured on
synthetic signatures says nothing about production performance, and the evaluation
harness refuses to emit a headline accuracy report from a synthetic-only test
split. Real genuine samples and, above all, real *skilled* forgeries are the
Phase 1 deliverable.

Output is Track B for licensing purposes: it is generated here, so nothing
restricts its commercial use.

What it models, and why those choices:

* A signer is a fixed set of spline control points, stroke widths, slant, and
  size — the writer's "motor programme".
* A **genuine** sample perturbs that programme slightly: small per-point
  jitter plus a mild global affine wobble. Real writers are consistent in
  shape but never identical.
* A **skilled forgery** reproduces the *shape* closely but not the
  *production*. Forgers draw slowly and deliberately, which shows up as
  (a) high-frequency tremor along the path, (b) flattened curvature extremes
  because fluent ballistic strokes are hard to copy, and (c) a low-frequency
  spatial warp from copying by eye. Those are the cues a verifier must learn,
  so they are exactly what is simulated.
* A **random forgery** is simply another signer's signature.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["SignerStyle", "make_signer", "render_signature", "render_on_form", "generate_corpus"]


# --------------------------------------------------------------------------
# Signer definition
# --------------------------------------------------------------------------


@dataclass
class SignerStyle:
    """The fixed 'motor programme' of one synthetic signer."""

    signer_id: str
    script: str  # "latin" | "arabic"
    strokes: list[list[list[float]]]  # per stroke: list of [x, y] control points, unit box
    dots: list[list[float]]  # diacritic / i-dot positions, unit box
    stroke_width: float
    slant_deg: float
    width_ratio: float  # aspect: signature width relative to height
    ink_intensity: int  # 0-255 darkness of the pen


def _spline(points: np.ndarray, samples: int = 600) -> np.ndarray:
    """Smooth a control polygon into a fluent path with a Catmull-Rom spline.

    Catmull-Rom passes through its control points, so the signer's programme is
    preserved exactly while the drawn line stays pen-like rather than angular.
    """
    if len(points) < 2:
        return points
    padded = np.vstack([points[0], points, points[-1]])
    out = []
    segments = len(padded) - 3
    per_seg = max(2, samples // max(segments, 1))
    for i in range(segments):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    out.append(padded[-2][None, :])
    return np.vstack(out)


# Cursive primitives, expressed as control points in a local frame where x
# advances left to right and y is relative to the baseline (negative = above).
# Real signatures are built from a small vocabulary of ballistic pen movements
# rather than from a smooth waveform, so the generator composes them the same
# way. Notably, ``loop`` and ``knot`` cross back over themselves — those
# self-intersections are what make a rendered sample read as handwriting.

def _primitive(kind: str, w: float, h: float) -> list[tuple[float, float]]:
    if kind == "arch":
        return [(0.0, 0.0), (0.20 * w, -0.62 * h), (0.55 * w, -0.70 * h), (0.85 * w, -0.25 * h), (w, 0.0)]
    if kind == "loop":
        return [
            (0.0, 0.0), (0.12 * w, -0.60 * h), (0.30 * w, -1.05 * h), (0.55 * w, -0.95 * h),
            (0.60 * w, -0.45 * h), (0.36 * w, -0.20 * h), (0.42 * w, 0.10 * h), (w, 0.0),
        ]
    if kind == "valley":
        return [(0.0, 0.0), (0.25 * w, 0.45 * h), (0.65 * w, 0.50 * h), (w, 0.0)]
    if kind == "spike":
        return [(0.0, 0.0), (0.30 * w, -0.55 * h), (0.50 * w, -1.15 * h), (0.70 * w, -0.50 * h), (w, 0.0)]
    if kind == "descender":
        return [
            (0.0, 0.0), (0.22 * w, 0.75 * h), (0.45 * w, 1.15 * h), (0.72 * w, 0.85 * h),
            (0.60 * w, 0.25 * h), (w, 0.0),
        ]
    if kind == "knot":  # tight double crossing, common in flourished signatures
        return [
            (0.0, 0.0), (0.20 * w, -0.55 * h), (0.48 * w, -0.20 * h), (0.22 * w, -0.05 * h),
            (0.30 * w, -0.50 * h), (0.62 * w, -0.62 * h), (w, -0.10 * h),
        ]
    # "run" — a flat connecting ligature between letter bodies
    return [(0.0, 0.0), (0.5 * w, -0.10 * h), (w, 0.0)]


_LATIN_UNITS = ("arch", "loop", "valley", "spike", "descender", "knot", "run")
_ARABIC_UNITS = ("valley", "run", "arch", "descender")


def _compose(
    units: tuple[str, ...],
    rng: np.random.Generator,
    *,
    n_units: int,
    baseline_drift: float,
) -> np.ndarray:
    """Chain primitives along a baseline into one continuous pen path."""
    pts: list[tuple[float, float]] = []
    x = 0.0
    y = 0.0
    for _ in range(n_units):
        kind = str(rng.choice(units))
        w = float(rng.uniform(0.6, 1.4))
        h = float(rng.uniform(0.6, 1.5))
        local = _primitive(kind, w, h)
        # Baseline wanders slightly upward or downward across the signature,
        # as a real hand does.
        y += float(rng.normal(0, baseline_drift))
        for lx, ly in local:
            pts.append((x + lx, y + ly))
        x += w * float(rng.uniform(0.72, 0.95))  # units overlap when joined up
    return np.asarray(pts, dtype=np.float64)


def _to_unit_box(pts: np.ndarray) -> np.ndarray:
    """Rescale a path into the [0, 1] x [0, 1] box, preserving aspect."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return (pts - lo) / span


def make_signer(signer_id: str, script: str, rng: np.random.Generator) -> SignerStyle:
    """Build a stable, distinctive style for one synthetic signer."""
    strokes: list[np.ndarray] = []

    if script == "arabic":
        # Right-to-left: a long, mostly horizontal connecting baseline with
        # bowls dipping below it, then separated diacritic dots. Rendered by
        # composing left-to-right and mirroring, so the ligature shapes keep
        # their characteristic asymmetry.
        body = _compose(_ARABIC_UNITS, rng, n_units=int(rng.integers(5, 9)), baseline_drift=0.05)
        body[:, 0] = body[:, 0].max() - body[:, 0]  # mirror to right-to-left
        strokes.append(body)
        # Standalone vertical ascenders (alif / lam shapes).
        for _ in range(int(rng.integers(1, 4))):
            ax = float(rng.uniform(0.15, 0.85)) * body[:, 0].max()
            strokes.append(np.array([[ax, 0.15], [ax + rng.uniform(-0.05, 0.05), -0.95]]))
        n_dots = int(rng.integers(2, 6))
        width_ratio = float(rng.uniform(2.6, 4.0))
    else:
        # Latin cursive: a capital flourish, the body, and often an underline.
        strokes.append(
            _compose(("loop", "knot", "spike"), rng, n_units=int(rng.integers(1, 3)), baseline_drift=0.02)
        )
        strokes.append(
            _compose(_LATIN_UNITS, rng, n_units=int(rng.integers(5, 10)), baseline_drift=0.06)
        )
        n_dots = int(rng.integers(0, 3))
        width_ratio = float(rng.uniform(2.2, 3.6))

    # Normalise the whole signature jointly so the strokes keep their relative
    # position and scale, then optionally add a trailing underline flourish.
    combined = np.vstack(strokes)
    lo = combined.min(axis=0)
    span = np.maximum(combined.max(axis=0) - lo, 1e-6)
    strokes = [(s - lo) / span for s in strokes]

    if rng.random() < 0.45:
        y0 = float(rng.uniform(0.92, 1.05))
        strokes.append(
            np.array([[0.02, y0], [0.35, y0 - 0.05], [0.72, y0 + 0.03], [1.0, y0 - 0.06]])
        )

    dots = [
        [float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.05, 0.35))] for _ in range(n_dots)
    ]

    return SignerStyle(
        signer_id=signer_id,
        script=script,
        strokes=[np.asarray(s, dtype=float).tolist() for s in strokes],
        dots=dots,
        stroke_width=float(rng.uniform(2.2, 4.5)),
        slant_deg=float(rng.uniform(-18, 8)),
        width_ratio=width_ratio,
        ink_intensity=int(rng.integers(20, 70)),
    )


# --------------------------------------------------------------------------
# Sample synthesis
# --------------------------------------------------------------------------


def _perturb(
    strokes: list[np.ndarray],
    rng: np.random.Generator,
    *,
    kind: str,
    skill: float = 0.7,
) -> list[np.ndarray]:
    """Apply genuine variation or forgery distortion to a signer's strokes.

    Args:
        skill: forger competence in [0, 1]. Higher means a closer copy and a
            harder verification problem. Scale the corpus difficulty with this
            rather than hand-editing the constants below.
    """
    out = []
    # A skilled forger deviates less; an unskilled one deviates a lot.
    deviation = 1.0 - 0.75 * float(np.clip(skill, 0.0, 1.0))
    for pts in strokes:
        pts = np.asarray(pts, dtype=np.float64).copy()

        if kind == "genuine":
            # Small, unstructured wobble — the writer's own repeatability.
            pts += rng.normal(0, 0.012, pts.shape)
        else:
            # Skilled forgery. Three compounding effects, all real-world cues:
            # 1. low-frequency warp from copying by eye
            n = len(pts)
            t = np.linspace(0, 1, n)
            warp = 0.055 * deviation
            warp_x = warp * np.sin(rng.uniform(0, 2 * math.pi) + rng.uniform(1.5, 3.0) * math.pi * t)
            warp_y = warp * np.sin(rng.uniform(0, 2 * math.pi) + rng.uniform(1.5, 3.0) * math.pi * t)
            pts += np.stack([warp_x, warp_y], axis=1)
            # 2. flattened curvature: fluent ballistic extremes are hard to copy
            centre = pts.mean(axis=0)
            pts = centre + (pts - centre) * (1.0 - rng.uniform(0.03, 0.11) * deviation)
            # 3. larger residual jitter than a genuine repetition
            pts += rng.normal(0, 0.026 * deviation, pts.shape)
        out.append(pts)
    return out


def _apply_affine(pts: np.ndarray, slant_deg: float, rng: np.random.Generator, kind: str) -> np.ndarray:
    """Global slant plus per-sample rotation/scale wobble, in unit coordinates."""
    jitter = 1.2 if kind == "genuine" else 3.0
    rot = math.radians(rng.normal(0, jitter))
    scale = 1.0 + rng.normal(0, 0.02 if kind == "genuine" else 0.05)
    shear = math.tan(math.radians(slant_deg + rng.normal(0, jitter)))

    centre = np.array([0.5, 0.5])
    p = pts - centre
    p = np.stack([p[:, 0] + shear * (0.5 - p[:, 1]), p[:, 1]], axis=1)
    r = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    p = p @ r.T * scale
    return p + centre


def render_signature(
    style: SignerStyle,
    rng: np.random.Generator,
    *,
    kind: str = "genuine",
    skill: float = 0.7,
    height: int = 220,
    background: int = 255,
) -> np.ndarray:
    """Render one sample of a signer's signature as a grayscale image.

    Args:
        kind: ``"genuine"`` or ``"forgery"`` (skilled).
        skill: forger competence in [0, 1]; ignored for genuine samples.
    """
    width = int(height * style.width_ratio)
    pad = int(height * 0.18)
    canvas = np.full((height + 2 * pad, width + 2 * pad), background, dtype=np.uint8)

    deviation = 1.0 - 0.75 * float(np.clip(skill, 0.0, 1.0))
    strokes = _perturb([np.asarray(s) for s in style.strokes], rng, kind=kind, skill=skill)
    base_width = style.stroke_width * (
        1.0 + rng.normal(0, 0.08 if kind == "genuine" else 0.20 * deviation)
    )
    intensity = int(np.clip(style.ink_intensity + rng.normal(0, 12), 0, 140))

    for pts in strokes:
        pts = _apply_affine(pts, style.slant_deg, rng, kind)
        path = _spline(pts)

        if kind == "forgery":
            # Tremor: the hallmark of a slowly drawn copy. High-frequency,
            # low-amplitude noise superimposed along the drawn path.
            n = len(path)
            freq = rng.uniform(40, 90)
            amp = rng.uniform(0.0012, 0.0035) * deviation
            phase = rng.uniform(0, 2 * math.pi)
            tremor = amp * np.sin(freq * np.linspace(0, 2 * math.pi, n) + phase)
            normals = np.gradient(path, axis=0)
            norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
            normals = np.stack([-normals[:, 1], normals[:, 0]], axis=1) / norms
            path = path + normals * tremor[:, None]

        px = np.clip(path[:, 0] * width + pad, 0, canvas.shape[1] - 1)
        py = np.clip(path[:, 1] * height + pad, 0, canvas.shape[0] - 1)
        poly = np.stack([px, py], axis=1).astype(np.int32)

        # Vary thickness along the stroke to imitate pen pressure.
        segments = max(6, len(poly) // 25)
        for i in range(segments):
            a = i * len(poly) // segments
            b = min(len(poly), (i + 1) * len(poly) // segments + 1)
            if b - a < 2:
                continue
            w = max(1, int(round(base_width * (1.0 + 0.25 * math.sin(i * 1.7)))))
            cv2.polylines(canvas, [poly[a:b]], False, int(intensity), w, cv2.LINE_AA)

    for dx, dy in style.dots:
        dot_sigma = 0.015 if kind == "genuine" else 0.045 * deviation
        jx = dx + rng.normal(0, dot_sigma)
        jy = dy + rng.normal(0, dot_sigma)
        cx = int(np.clip(jx * width + pad, 0, canvas.shape[1] - 1))
        cy = int(np.clip(jy * height + pad, 0, canvas.shape[0] - 1))
        cv2.circle(canvas, (cx, cy), max(1, int(base_width * 0.7)), int(intensity), -1, cv2.LINE_AA)

    return canvas


# --------------------------------------------------------------------------
# Form rendering (for Stage A detector training and realistic demo input)
# --------------------------------------------------------------------------


def render_on_form(
    signature: np.ndarray,
    rng: np.random.Generator,
    *,
    page_size: tuple[int, int] = (1400, 1000),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paste a signature onto a printed-form background.

    Returns the page image and the ground-truth signature bounding box
    (x, y, w, h), which the Stage A detector trains against.

    The form deliberately includes the things that break naive pipelines: ruled
    lines running through the signature, a box border, printed text blocks, a
    round stamp, and uneven illumination.
    """
    ph, pw = page_size
    page = np.full((ph, pw), 252, dtype=np.uint8)

    # Printed text blocks (grey bars stand in for form field labels).
    for _ in range(int(rng.integers(6, 12))):
        y = int(rng.integers(40, ph - 200))
        x = int(rng.integers(40, pw // 2))
        w = int(rng.integers(120, pw // 2))
        cv2.rectangle(page, (x, y), (x + w, y + int(rng.integers(6, 12))), 150, -1)

    # Horizontal rules.
    for _ in range(int(rng.integers(3, 7))):
        y = int(rng.integers(60, ph - 60))
        cv2.line(page, (40, y), (pw - 40, y), int(rng.integers(120, 180)), 1)

    # Signature box near the bottom of the page.
    box_h = int(rng.integers(160, 240))
    box_w = int(rng.integers(int(pw * 0.45), int(pw * 0.75)))
    box_x = int(rng.integers(40, max(41, pw - box_w - 40)))
    box_y = int(rng.integers(int(ph * 0.55), max(int(ph * 0.56), ph - box_h - 60)))
    cv2.rectangle(page, (box_x, box_y), (box_x + box_w, box_y + box_h), 140, 1)
    # A signature line inside the box — the classic case of ink over a rule.
    cv2.line(
        page,
        (box_x + 20, box_y + box_h - 40),
        (box_x + box_w - 20, box_y + box_h - 40),
        130,
        1,
    )

    # Fit the signature into the box.
    sh, sw = signature.shape
    scale = min((box_w - 40) / sw, (box_h - 30) / sh)
    if scale < 1.0:
        signature = cv2.resize(signature, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = signature.shape
    ox = box_x + int(rng.integers(15, max(16, box_w - sw - 10)))
    oy = box_y + int(rng.integers(10, max(11, box_h - sh - 10)))

    region = page[oy : oy + sh, ox : ox + sw]
    page[oy : oy + sh, ox : ox + sw] = np.minimum(region, signature)

    # Round stamp overlapping the signature area, partially transparent.
    if rng.random() < 0.35:
        cx = ox + int(rng.integers(0, max(1, sw)))
        cy = oy + int(rng.integers(0, max(1, sh)))
        overlay = page.copy()
        cv2.circle(overlay, (cx, cy), int(rng.integers(50, 90)), 90, 3)
        page = cv2.addWeighted(overlay, 0.45, page, 0.55, 0)

    # Uneven illumination across the page.
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    gradient = 1.0 - 0.18 * (
        (xx / pw) * rng.uniform(-1, 1) + (yy / ph) * rng.uniform(-1, 1)
    )
    page = np.clip(page.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    page = np.clip(page.astype(np.int16) + rng.normal(0, 3, page.shape).astype(np.int16), 0, 255).astype(np.uint8)

    return page, (ox, oy, sw, sh)


# --------------------------------------------------------------------------
# Corpus generation
# --------------------------------------------------------------------------


def generate_corpus(
    out_dir: Path,
    *,
    n_signers: int = 120,
    genuine_per_signer: int = 12,
    forgeries_per_signer: int = 8,
    forms_per_signer: int = 2,
    arabic_fraction: float = 0.5,
    forger_skill: float = 0.7,
    seed: int = 1337,
) -> dict:
    """Generate the corpus on disk and return a summary dictionary."""
    out_dir = Path(out_dir)
    (out_dir / "signers").mkdir(parents=True, exist_ok=True)
    (out_dir / "forms").mkdir(parents=True, exist_ok=True)

    master = np.random.default_rng(seed)
    manifest: list[dict] = []
    styles: dict[str, dict] = {}

    for i in range(n_signers):
        signer_id = f"S{i:04d}"
        script = "arabic" if master.random() < arabic_fraction else "latin"
        # Deterministic per-signer stream so a signer is reproducible in
        # isolation, which makes debugging a single signer possible.
        signer_rng = np.random.default_rng(seed + 100_000 + i)
        style = make_signer(signer_id, script, signer_rng)
        styles[signer_id] = asdict(style)

        gen_dir = out_dir / "signers" / signer_id / "genuine"
        forg_dir = out_dir / "signers" / signer_id / "forgery"
        gen_dir.mkdir(parents=True, exist_ok=True)
        forg_dir.mkdir(parents=True, exist_ok=True)

        for j in range(genuine_per_signer):
            rng = np.random.default_rng(seed + 200_000 + i * 1000 + j)
            img = render_signature(style, rng, kind="genuine")
            path = gen_dir / f"{signer_id}_g{j:02d}.png"
            cv2.imwrite(str(path), img)
            manifest.append(
                {
                    "path": str(path.relative_to(out_dir)).replace("\\", "/"),
                    "signer_id": signer_id,
                    "label": "genuine",
                    "script": script,
                    "source": "synthetic",
                }
            )

        for j in range(forgeries_per_signer):
            rng = np.random.default_rng(seed + 300_000 + i * 1000 + j)
            # Vary forger competence within the cohort: a corpus where every
            # forgery is equally good produces a misleadingly clean ROC curve.
            sample_skill = float(np.clip(forger_skill + rng.normal(0, 0.12), 0.0, 1.0))
            img = render_signature(style, rng, kind="forgery", skill=sample_skill)
            path = forg_dir / f"{signer_id}_f{j:02d}.png"
            cv2.imwrite(str(path), img)
            manifest.append(
                {
                    "path": str(path.relative_to(out_dir)).replace("\\", "/"),
                    "signer_id": signer_id,
                    "label": "skilled_forgery",
                    "script": script,
                    "source": "synthetic",
                }
            )

        for j in range(forms_per_signer):
            rng = np.random.default_rng(seed + 400_000 + i * 1000 + j)
            sig = render_signature(style, rng, kind="genuine")
            page, bbox = render_on_form(sig, rng)
            page_path = out_dir / "forms" / f"{signer_id}_form{j:02d}.png"
            cv2.imwrite(str(page_path), page)
            (out_dir / "forms" / f"{signer_id}_form{j:02d}.json").write_text(
                json.dumps({"signer_id": signer_id, "bbox": list(bbox), "script": script}, indent=2)
            )

    (out_dir / "manifest_raw.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "styles.json").write_text(json.dumps(styles, indent=2))

    summary = {
        "signers": n_signers,
        "images": len(manifest),
        "genuine": n_signers * genuine_per_signer,
        "skilled_forgery": n_signers * forgeries_per_signer,
        "forms": n_signers * forms_per_signer,
        "arabic_fraction": arabic_fraction,
        "forger_skill": forger_skill,
        "seed": seed,
        "licence_track": "track_b_production",
        "warning": "Synthetic. Never report headline accuracy from this corpus.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--signers", type=int, default=120)
    parser.add_argument("--genuine", type=int, default=12)
    parser.add_argument("--forgeries", type=int, default=8)
    parser.add_argument("--forms", type=int, default=2)
    parser.add_argument("--arabic-fraction", type=float, default=0.5)
    parser.add_argument(
        "--forger-skill",
        type=float,
        default=0.7,
        help="Forger competence 0-1. Higher = closer copies = harder problem.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    summary = generate_corpus(
        args.out,
        n_signers=args.signers,
        genuine_per_signer=args.genuine,
        forgeries_per_signer=args.forgeries,
        forms_per_signer=args.forms,
        arabic_fraction=args.arabic_fraction,
        forger_skill=args.forger_skill,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {args.out.resolve()}")


if __name__ == "__main__":
    main()
