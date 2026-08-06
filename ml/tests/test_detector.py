"""Tests for the Stage A signature detector."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from ml.data.synth import make_signer, render_on_form, render_signature
from ml.detector.evaluate import evaluate_directory, iou
from ml.detector.heuristic import Detection, detect_candidates, detect_signature


def _page(seed: int = 3):
    rng = np.random.default_rng(seed)
    style = make_signer(f"D{seed}", "latin", rng)
    signature = render_signature(style, rng, kind="genuine")
    return render_on_form(signature, rng)


# --------------------------------------------------------------------------
# Heuristic detection
# --------------------------------------------------------------------------


def test_detects_the_signature_on_a_form():
    page, truth = _page()
    detection = detect_signature(page)
    assert detection is not None
    assert detection.method == "heuristic"
    # A loose box is fine — preprocessing re-crops to the ink. A miss is not.
    assert iou(detection.bbox, truth) > 0.1


def test_declines_rather_than_guessing_on_a_blank_page():
    """A wrong crop scored confidently is worse than asking for a manual box."""
    blank = np.full((1400, 1000), 250, dtype=np.uint8)
    assert detect_signature(blank) is None


def test_candidates_are_ranked_by_confidence():
    page, _ = _page(5)
    candidates = detect_candidates(page)
    assert candidates
    scores = [c.confidence for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_crop_includes_padding_and_stays_in_bounds():
    page, _ = _page(7)
    detection = detect_signature(page)
    assert detection is not None
    crop = detection.crop(page)
    assert crop.size > 0
    assert crop.shape[0] <= page.shape[0] and crop.shape[1] <= page.shape[1]


def test_crop_at_the_page_edge_does_not_overflow():
    page = np.full((400, 600), 250, dtype=np.uint8)
    detection = Detection(bbox=(560, 360, 80, 80), confidence=1.0)
    crop = detection.crop(page)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_detection_serialises_for_the_api():
    payload = Detection((10, 20, 300, 90), 0.72).to_dict()
    assert payload["bbox"] == {"x": 10, "y": 20, "width": 300, "height": 90}
    assert payload["confidence"] == 0.72


# --------------------------------------------------------------------------
# IoU helper
# --------------------------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_of_half_overlap():
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


# --------------------------------------------------------------------------
# Directory evaluation
# --------------------------------------------------------------------------


def test_evaluate_directory_reports_recall(tmp_path):
    for i in range(4):
        page, bbox = _page(20 + i)
        cv2.imwrite(str(tmp_path / f"p{i}.png"), page)
        (tmp_path / f"p{i}.json").write_text(json.dumps({"bbox": list(bbox)}))

    report = evaluate_directory(tmp_path)
    assert report["pages"] == 4
    assert report["detection_misses"] == 0
    assert report["recall_at_iou_0.1"] == 1.0


def test_evaluate_directory_errors_on_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_directory(tmp_path)


# --------------------------------------------------------------------------
# Licence policy
# --------------------------------------------------------------------------


def test_pretrained_detector_is_blocked_on_the_production_track():
    from ml.detector.train import build_detector

    with pytest.raises(ValueError, match="Track B"):
        build_detector(pretrained=True, track="b")


def test_untrained_detector_never_pulls_imagenet_backbone_weights(monkeypatch):
    """Regression guard for a subtle licence leak.

    ``fasterrcnn_resnet50_fpn`` has a separate ``weights_backbone`` default of
    ImageNet, so ``weights=None`` alone still downloads pretrained weights into
    a model that is supposed to have a clean origin. Nothing surfaces the
    contamination except watching the network, so it is asserted here.
    """
    import torchvision.models.detection as detection

    captured: dict = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before constructing the model")

    monkeypatch.setattr(detection, "fasterrcnn_resnet50_fpn", spy)

    from ml.detector.train import build_detector

    with pytest.raises(RuntimeError, match="stop before"):
        build_detector(pretrained=False, track="b")

    assert "weights_backbone" in captured, "weights_backbone must be passed explicitly"
    assert captured["weights_backbone"] is None
    assert captured["weights"] is None
