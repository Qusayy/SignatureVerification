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


# --------------------------------------------------------------------------
# Whole-signature capture
#
# The defect these guard: `detect_signature` returned a single connected
# component. An image containing nothing but a signature therefore came back
# cropped to the largest blob of it, and the fragment scored perfectly
# normally against the stored specimens — part of a signature does resemble the
# whole of it. Nothing in the output showed that most of the ink was discarded.
# --------------------------------------------------------------------------


def _bare_signature(gap: int = 260) -> np.ndarray:
    """A signature in two clearly separate pieces, on blank paper.

    Detached parts are the common case, not an edge case: initials written
    apart from a surname, a separate flourish, Arabic diacritics.
    """
    image = np.full((260, 900), 255, np.uint8)
    cv2.ellipse(image, (200, 130), (150, 60), 12, 0, 300, 0, 5)
    cv2.ellipse(image, (200 + gap + 160, 120), (110, 45), -8, 20, 320, 0, 5)
    cv2.circle(image, (770, 70), 4, 0, -1)
    return image


def _form_with_signature(extra_mark: bool = False) -> np.ndarray:
    """Printed text block, a rule, and a signature below it."""
    page = np.full((900, 700), 255, np.uint8)
    for row in range(80, 460, 34):
        for col in range(60, 620, 13):
            cv2.rectangle(page, (col, row), (col + 9, row + 11), 0, -1)
    cv2.line(page, (60, 700), (640, 700), 0, 2)
    cv2.ellipse(page, (300, 660), (170, 42), 8, 0, 300, 0, 4)
    if extra_mark:
        cv2.ellipse(page, (600, 800), (60, 30), 0, 0, 300, 0, 4)
    return page


def test_a_bare_signature_is_not_cropped_to_one_piece():
    image = _bare_signature()
    detection = detect_signature(image)

    assert detection is not None
    x, y, w, h = detection.bbox
    # Both pieces are inside the box: the second ends near x=770.
    assert x < 100 and x + w > 760
    assert detection.ink_captured == 1.0
    assert not detection.is_partial


def test_a_bare_signature_reports_the_whole_image_method():
    """Nothing to exclude means nothing should be excluded."""
    detection = detect_signature(_bare_signature())
    assert detection is not None
    assert detection.method == "whole-image"


def test_a_form_is_still_cropped_tightly():
    """The fix must not turn every form into a full-page crop."""
    detection = detect_signature(_form_with_signature())

    assert detection is not None
    assert detection.method == "heuristic"
    x, y, w, h = detection.bbox
    # The signature sits at y=660; the printed text ends at y=471.
    assert y > 500, "the crop swallowed the printed text block"
    assert h < 400


def test_a_form_does_not_warn_about_its_own_printed_ink():
    """`ink_captured` is measured against handwriting, not all ink.

    Measured against everything, a correct crop on a form discards ~95% of the
    ink and would warn every single time, which trains people to ignore it.
    """
    detection = detect_signature(_form_with_signature())
    assert detection is not None
    assert detection.ink_captured == 1.0
    assert not detection.is_partial


def test_a_detached_mark_on_a_form_is_absorbed():
    detection = detect_signature(_form_with_signature(extra_mark=True))
    assert detection is not None
    x, y, w, h = detection.bbox
    assert y + h > 790, "the second mark at y=800 was left outside the box"


def test_printed_structure_is_recognised():
    from ml.detector.heuristic import _has_printed_structure, _ink_mask

    form = _form_with_signature()
    bare = _bare_signature()

    assert _has_printed_structure(form, _ink_mask(form))
    assert not _has_printed_structure(bare, _ink_mask(bare))


def test_ink_captured_is_reported_in_the_payload():
    payload = Detection((10, 20, 300, 90), 0.72, ink_captured=0.61).to_dict()
    assert payload["ink_captured"] == 0.61


@pytest.mark.parametrize("captured,expected", [(1.0, False), (0.95, False), (0.6, True)])
def test_is_partial_threshold(captured, expected):
    assert Detection((0, 0, 10, 10), 0.5, ink_captured=captured).is_partial is expected


def test_blank_image_still_declines():
    assert detect_signature(np.full((300, 800), 255, np.uint8)) is None
