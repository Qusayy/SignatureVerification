"""Measure Stage A detection quality against ground-truth boxes.

Reports IoU distribution and recall at several IoU thresholds. For this stage
recall matters far more than precision: a slightly loose crop still verifies
fine, because preprocessing crops to the ink anyway. A *missed* signature stops
the employee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ml.detector.heuristic import detect_signature

__all__ = ["iou", "evaluate_directory"]


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def evaluate_directory(forms_dir: Path, limit: int | None = None) -> dict:
    """Evaluate every ``*_form*.png`` with a matching ``.json`` ground truth."""
    forms_dir = Path(forms_dir)
    pages = sorted(forms_dir.glob("*.png"))
    if limit:
        pages = pages[:limit]
    if not pages:
        raise FileNotFoundError(f"No form images in {forms_dir}")

    ious: list[float] = []
    misses = 0
    for page_path in pages:
        truth_path = page_path.with_suffix(".json")
        if not truth_path.exists():
            continue
        truth = tuple(json.loads(truth_path.read_text())["bbox"])
        page = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)

        detection = detect_signature(page)
        if detection is None:
            misses += 1
            ious.append(0.0)
            continue
        ious.append(iou(detection.bbox, truth))  # type: ignore[arg-type]

    values = np.asarray(ious)
    return {
        "pages": len(values),
        "detection_misses": misses,
        "mean_iou": round(float(values.mean()), 4),
        "median_iou": round(float(np.median(values)), 4),
        "recall_at_iou_0.5": round(float((values >= 0.5).mean()), 4),
        "recall_at_iou_0.3": round(float((values >= 0.3).mean()), 4),
        "recall_at_iou_0.1": round(float((values >= 0.1).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage A detector")
    parser.add_argument("--forms", type=Path, default=Path("data/synthetic/forms"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_directory(args.forms, args.limit), indent=2))


if __name__ == "__main__":
    main()
