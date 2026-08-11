"""Diagnostics that answer *why* a model scores the way it does.

An EER number tells you the model is bad. It does not tell you what it latched
onto. These probes do, and they exist because the first model shipped here
looked like it was matching signature *size* rather than *shape* — which turned
out to be substantially true, and was invisible in every aggregate metric.

Run after any change to preprocessing, augmentation, or the backbone::

    python -m ml.eval.diagnostics --checkpoint artifacts/signet_track_b.pt

The scale sweep is also enforced as a blocking test in
``ml/tests/test_scale_invariance.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.config import ARTIFACT_ROOT
from ml.data.manifest import DEFAULT_MANIFEST_PATH, Manifest, Record
from ml.eval.metrics import compute_metrics
from ml.preprocess.pipeline import preprocess_signature, to_model_input

__all__ = [
    "ScaleSweep",
    "scale_sweep",
    "size_correlation",
    "load_embedder",
    "DEFAULT_SCALES",
]

# The range a real customer plausibly varies across between visits. A model
# that cannot hold a signature's identity across this range will reject
# genuine customers for writing slightly larger than the specimen on file.
DEFAULT_SCALES: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.4, 1.6)


def load_embedder(checkpoint: Path | str):
    """Return a function mapping a list of canvases to L2-normalised vectors."""
    from ml.embed.models import load_checkpoint

    model, _payload = load_checkpoint(checkpoint)
    model.eval()

    def embed(canvases: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([to_model_input(c) for c in canvases])
        with torch.no_grad():
            out = model(torch.from_numpy(batch))
        return torch.nn.functional.normalize(out, p=2, dim=1).numpy().astype(np.float64)

    return embed


@dataclass
class ScaleSweep:
    """How similarity to the original survives rescaling the same signature."""

    similarities: dict[float, float] = field(default_factory=dict)
    genuine_repeat: float | None = None
    skilled_forgery: float | None = None

    @property
    def worst(self) -> float:
        return min(self.similarities.values()) if self.similarities else float("nan")

    @property
    def passes(self) -> bool:
        """Every rescaled copy must stay clearly above the forgery band.

        The bar is deliberately absolute rather than relative to the forgery
        score: a model can satisfy a relative bar simply by scoring everything
        alike, which is what a useless model does.
        """
        return self.worst >= 0.95

    def to_dict(self) -> dict:
        return {
            "similarities": {f"{k:g}x": round(v, 4) for k, v in self.similarities.items()},
            "worst": round(self.worst, 4),
            "genuine_repeat": round(self.genuine_repeat, 4) if self.genuine_repeat else None,
            "skilled_forgery": round(self.skilled_forgery, 4) if self.skilled_forgery else None,
            "passes": self.passes,
        }


def scale_sweep(
    embed,
    raw_signature: np.ndarray,
    *,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> ScaleSweep:
    """Rescale one signature and measure similarity to the unscaled original.

    Shape is identical throughout; only size changes. A shape-driven model
    should barely notice.
    """
    base = embed([preprocess_signature(raw_signature).image])[0]
    result = ScaleSweep()

    h, w = raw_signature.shape[:2]
    for factor in scales:
        resized = cv2.resize(
            raw_signature,
            (max(8, int(w * factor)), max(8, int(h * factor))),
            interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR,
        )
        canvas = preprocess_signature(resized, strict=False).image
        result.similarities[factor] = float(base @ embed([canvas])[0])

    return result


def _canvas_size_features(canvas: np.ndarray) -> tuple[int, int, int]:
    ys, xs = np.nonzero(canvas)
    if len(xs) == 0:
        return (0, 1, 1)
    return (
        int(np.count_nonzero(canvas)),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def size_correlation(embed, manifest: Manifest, records: list[Record]) -> dict:
    """Does model similarity track shape, or merely size?

    Reports correlations against pure size features and, more usefully, the EER
    a size-only 'model' would achieve. If the trained model is not far better
    than that baseline, it has learned very little about shape.
    """
    canvases: dict[str, np.ndarray] = {}
    features: dict[str, tuple[int, int, int]] = {}
    for record in records:
        image = cv2.imread(str(manifest.resolve(record)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        canvas = preprocess_signature(image, strict=False).image
        canvases[record.image_path] = canvas
        features[record.image_path] = _canvas_size_features(canvas)

    paths = list(canvases)
    vectors = embed([canvases[p] for p in paths])
    index = {p: i for i, p in enumerate(paths)}

    by_signer: dict[str, dict[str, list[Record]]] = defaultdict(
        lambda: {"genuine": [], "skilled_forgery": []}
    )
    for record in records:
        if record.image_path in canvases:
            by_signer[record.signer_id][
                "genuine" if record.label == "genuine" else "skilled_forgery"
            ].append(record)

    sims, ink_ratio, area_ratio, labels = [], [], [], []
    for bucket in by_signer.values():
        genuine = bucket["genuine"]
        if len(genuine) < 2:
            continue
        anchor = genuine[0]
        a = vectors[index[anchor.image_path]]
        ia, wa, ha = features[anchor.image_path]
        pairs = [(g, 1) for g in genuine[1:]] + [(f, 0) for f in bucket["skilled_forgery"]]
        for other, is_genuine in pairs:
            b = vectors[index[other.image_path]]
            ib, wb, hb = features[other.image_path]
            sims.append(float(a @ b))
            ink_ratio.append(min(ia, ib) / max(ia, ib, 1))
            area_ratio.append(min(wa * ha, wb * hb) / max(wa * ha, wb * hb, 1))
            labels.append(is_genuine)

    sims_a = np.asarray(sims)
    labels_a = np.asarray(labels)
    if len(sims_a) < 20 or labels_a.sum() == 0 or (1 - labels_a).sum() == 0:
        return {"error": "not enough comparisons to correlate"}

    model_metrics = compute_metrics(sims_a[labels_a == 1], sims_a[labels_a == 0])
    ink_a = np.asarray(ink_ratio)
    size_metrics = compute_metrics(ink_a[labels_a == 1], ink_a[labels_a == 0])

    return {
        "comparisons": len(sims_a),
        "corr_with_ink_similarity": round(float(np.corrcoef(sims_a, ink_a)[0, 1]), 3),
        "corr_with_bbox_area_similarity": round(
            float(np.corrcoef(sims_a, np.asarray(area_ratio))[0, 1]), 3
        ),
        "corr_with_genuine_label": round(float(np.corrcoef(sims_a, labels_a)[0, 1]), 3),
        "eer_size_only_baseline": round(size_metrics.eer, 4),
        "eer_trained_model": round(model_metrics.eer, 4),
        "model_beats_size_baseline_by": round(size_metrics.eer - model_metrics.eer, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose what a model is matching on")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_ROOT / "signet_track_b.pt")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-correlation", action="store_true", help="Scale sweep only (fast)")
    args = parser.parse_args()

    embed = load_embedder(args.checkpoint)
    manifest = Manifest.load(args.manifest)
    records = manifest.by_split(args.split)  # type: ignore[arg-type]

    sample = next(r for r in records if r.label == "genuine")
    raw = cv2.imread(str(manifest.resolve(sample)), cv2.IMREAD_GRAYSCALE)
    sweep = scale_sweep(embed, raw)

    # Reference points, so the sweep numbers mean something.
    peers = [r for r in records if r.signer_id == sample.signer_id]
    repeat = next((r for r in peers if r.label == "genuine" and r is not sample), None)
    forgery = next((r for r in peers if r.label != "genuine"), None)
    base = embed([preprocess_signature(raw).image])[0]
    for record, attribute in ((repeat, "genuine_repeat"), (forgery, "skilled_forgery")):
        if record is None:
            continue
        image = cv2.imread(str(manifest.resolve(record)), cv2.IMREAD_GRAYSCALE)
        canvas = preprocess_signature(image, strict=False).image
        setattr(sweep, attribute, float(base @ embed([canvas])[0]))

    report = {"scale_sweep": sweep.to_dict()}
    if not args.skip_correlation:
        report["size_vs_shape"] = size_correlation(embed, manifest, records)

    print(json.dumps(report, indent=2))
    if not sweep.passes:
        print(
            f"\nFAIL: worst rescaled similarity {sweep.worst:.3f} is below 0.95. "
            "The model is size-sensitive; a customer signing larger or smaller than "
            "their stored specimen will be rejected."
        )


if __name__ == "__main__":
    main()
