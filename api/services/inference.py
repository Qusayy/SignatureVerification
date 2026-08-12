"""Model loading and the verification pipeline used by the API.

Loads once at startup and keeps the model warm; a operator waiting three seconds
is acceptable, waiting for a cold model load is not.

If no checkpoint is present the service still starts, reports
``model_loaded: false`` on ``/api/health``, and fails verification requests with
a clear 503. Refusing to boot would make the stack impossible to stand up
incrementally, and silently serving a random-weight model would be far worse
than an explicit error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from api.settings import get_settings
from ml.detector.heuristic import Detection, detect_signature
from ml.preprocess.pipeline import deskew_page, estimate_page_skew, to_grayscale
from ml.preprocess.trace import PipelineTrace, Stage
from ml.scoring.compare import intra_reference_mean
from ml.scoring.explain import difference_overlay, reason_text
from ml.scoring.verifier import EnrolmentBundle, VerificationResult, Verifier
from ml.scoring.znorm import CohortStats

logger = logging.getLogger(__name__)

__all__ = ["InferenceService", "get_service", "ModelNotLoaded", "PipelineOutput"]


class ModelNotLoaded(RuntimeError):
    """Raised when verification is attempted without a usable model."""


@dataclass
class PipelineOutput:
    """Everything one verification produces, ready to persist and return."""

    result: VerificationResult
    detection: Detection | None
    crop: np.ndarray
    overlay: np.ndarray
    reason: str
    stages: list[Stage] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _annotate_detection(page: np.ndarray, detection: Detection) -> np.ndarray:
    """Draw the detected region on a copy of the page, for the trace."""
    canvas = page if page.ndim == 3 else cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    canvas = canvas.copy()
    x, y, w, h = detection.bbox
    thickness = max(2, int(min(canvas.shape[:2]) * 0.004))
    # BGR: amber, matching the accent the interface uses for the active stage.
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 170, 255), thickness)
    return canvas


class InferenceService:
    """Owns the loaded model and runs the full capture-to-score pipeline."""

    def __init__(self) -> None:
        self.verifier: Verifier | None = None
        self.load_error: str | None = None
        self.checkpoint_path: Path | None = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        settings = get_settings()
        self.checkpoint_path = settings.checkpoint_path

        if not settings.checkpoint_path.exists():
            self.load_error = (
                f"No checkpoint at {settings.checkpoint_path}. Train one with "
                "`python -m ml.embed.train`."
            )
            logger.warning(self.load_error)
            return

        if settings.require_deployable_checkpoint:
            from ml.embed.provenance import assert_deployable

            # Deliberately not caught: if the operator asked for the licence
            # gate, a research-licensed model must stop the service.
            assert_deployable(settings.checkpoint_path)

        try:
            self.verifier = Verifier.from_artifacts(
                settings.checkpoint_path,
                cohort_path=settings.cohort_path,
                calibrator_path=settings.calibrator_path,
                device=None if settings.device == "auto" else settings.device,
            )
            self.load_error = None
            logger.info("Loaded model %s", self.verifier.model_version)
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
            self.load_error = f"Failed to load model: {exc}"
            logger.exception("Model load failed")

    @property
    def is_ready(self) -> bool:
        return self.verifier is not None

    def status(self) -> dict:
        settings = get_settings()
        verifier = self.verifier
        return {
            "model_loaded": self.is_ready,
            "model_version": verifier.model_version if verifier else None,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            # Whether it is *applied*, not merely loaded. The cohort file is
            # still produced and still verified against the weights, but
            # SCORING.cohort_normalise decides whether scoring uses it.
            "cohort_normalisation": bool(
                verifier and verifier.cohort is not None and verifier.cfg.cohort_normalise
            ),
            "writer_normalisation": bool(verifier and verifier.cfg.writer_normalise),
            "calibrated": bool(verifier and not verifier.calibrator.is_placeholder),
            "advisory_only": settings.advisory_only,
            "error": self.load_error,
        }

    def _require(self) -> Verifier:
        if self.verifier is None:
            raise ModelNotLoaded(self.load_error or "Model is not loaded")
        return self.verifier

    # -- enrolment --------------------------------------------------------

    def embed_reference(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess and embed one specimen. Returns (canvas, embedding)."""
        verifier = self._require()
        canvas = verifier.preprocess(image).image
        embedding = verifier.embed_canvases([canvas])[0]
        return canvas, embedding

    def enrolment_stats(self, embeddings: np.ndarray) -> CohortStats | None:
        verifier = self._require()
        if verifier.cohort is None:
            return None
        return verifier.cohort.enrolment_stats(embeddings)

    def enrolment_state(self, embeddings: np.ndarray) -> tuple[CohortStats | None, float]:
        """Everything cached per customer: cohort statistics and specimen agreement.

        Returned together because both are computed from the same embeddings
        and both must be refreshed whenever the specimen set changes. The
        cohort half is None when cohort normalisation is off — which is now the
        default — but the specimen agreement is always available, so the
        enrolment row is written either way. It did not used to be, and a
        customer without one silently fell back to an unnormalised score.
        """
        return self.enrolment_stats(embeddings), intra_reference_mean(embeddings)

    # -- verification -----------------------------------------------------

    def locate_signature(
        self,
        page: np.ndarray,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        trace: PipelineTrace | None = None,
    ) -> tuple[np.ndarray, Detection | None]:
        """Find the signature on an uploaded page and return the crop.

        Args:
            bbox: an employee-supplied region. When given, detection is skipped
                entirely — a human correcting the crop must always win over the
                detector.
            trace: optional recorder for the capture, deskew and detection
                stages.
        """
        if bbox is not None:
            x, y, w, h = bbox
            crop = page[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                raise ValueError("The supplied crop region is empty")
            detection = Detection(bbox, confidence=1.0, method="employee")
            if trace:
                trace.add(
                    "detect",
                    "Region supplied by operator",
                    "You drew this box, so automatic detection was skipped entirely. A "
                    "human correcting the crop always overrides the detector.",
                    image=_annotate_detection(page, detection),
                    method="operator",
                )
            return crop, detection

        # Correct page rotation before locating anything. Skew estimation needs
        # the printed structure of the whole form, which the crop no longer has.
        angle = estimate_page_skew(to_grayscale(page))
        deskewed, applied = deskew_page(page, angle)
        if trace:
            trace.add(
                "deskew",
                "Page straightened",
                "Skew is measured from the printed rules of the form, never from the "
                "handwriting — a writer's own slant is one of the more stable things "
                "that distinguishes them, so it is preserved, not corrected."
                if abs(applied) >= 0.2
                else "The printed rules on this page are already square, so no rotation "
                "was applied. Skew is only ever measured from printed structure, never "
                "from the handwriting.",
                image=deskewed,
                measured_deg=round(float(angle), 2),
                applied_deg=round(float(applied), 2),
            )

        detection = detect_signature(deskewed)
        if detection is None:
            raise ValueError(
                "No signature region found on this page. Draw the box manually or rescan."
            )
        if trace:
            trace.add(
                "detect",
                "Signature located",
                "Candidate regions are scored on ink density, stroke connectivity and "
                "aspect ratio, then detached pieces of the same signature — a separate "
                "flourish, initials written apart, Arabic diacritics — are merged back "
                "in. Printed text is refused: it packs far more densely into its box "
                "than handwriting does."
                + (
                    f" {(1 - detection.ink_captured):.0%} of the ink on this image falls "
                    "outside the box. On a form that is the form itself; on an image "
                    "that is only a signature it means part of it was cut off."
                    if detection.is_partial
                    else ""
                ),
                image=_annotate_detection(deskewed, detection),
                confidence=round(float(detection.confidence), 3),
                ink_captured=round(float(detection.ink_captured), 3),
                method=detection.method,
            )
        return detection.crop(deskewed), detection

    def verify(
        self,
        page_or_crop: np.ndarray,
        enrolment: EnrolmentBundle,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        is_full_page: bool = True,
        explain: bool = False,
    ) -> PipelineOutput:
        """Run detection, preprocessing, scoring, and explanation.

        Args:
            explain: capture every intermediate stage for display. Adds image
                copies and a dozen PNG encodes to the request, so it is opt-in.

        Raises:
            ModelNotLoaded: no usable model.
            BlankSignatureError: the crop holds too little ink to score.
            ValueError: no signature could be located on the page.
        """
        verifier = self._require()
        trace = PipelineTrace(enabled=explain)

        if trace:
            trace.add(
                "capture",
                "Captured image",
                "Exactly what arrived from the scanner or camera. Nothing has been "
                "altered yet.",
                image=page_or_crop,
                pixels=f"{page_or_crop.shape[1]}x{page_or_crop.shape[0]}",
            )

        if is_full_page:
            crop, detection = self.locate_signature(page_or_crop, bbox=bbox, trace=trace)
        else:
            crop, detection = page_or_crop, None

        if trace and is_full_page:
            trace.add(
                "crop",
                "Region extracted",
                "Everything outside the signature is discarded. The rest of the form is "
                "never embedded, never stored as a specimen and never scored.",
                image=crop,
                pixels=f"{crop.shape[1]}x{crop.shape[0]}",
            )

        result = verifier.verify(crop, enrolment, trace=trace)

        if detection is not None and detection.is_partial:
            # Say so on the result, not only in the replay. A score computed
            # from part of a signature looks exactly like a score computed from
            # all of it, and the part that was dropped is invisible by then.
            result.warnings.append("ink_outside_detected_region")

        # Overlay against the *closest* specimen, not simply the first one.
        # When a customer's stored specimens differ from each other — signed
        # years apart, different pens — overlaying an arbitrary one shows the
        # employee a mismatch that the score did not actually penalise.
        overlay = difference_overlay(
            result.query_canvas, self._closest_canvas(result, enrolment)
        )

        if trace:
            trace.add(
                "overlay",
                "Difference overlay",
                "The captured signature laid over the closest specimen on file. Agreement "
                "is grey; ink only in the capture is one colour and ink only in the "
                "specimen the other. This is where a forger's hesitation shows.",
                image=overlay,
                compared_against="closest specimen on file",
            )

        return PipelineOutput(
            result=result,
            detection=detection,
            crop=crop,
            overlay=overlay,
            reason=reason_text(result),
            stages=trace.stages,
            diagnostics=verifier.score_diagnostics(result),
        )

    @staticmethod
    def _closest_canvas(result: VerificationResult, enrolment: EnrolmentBundle) -> np.ndarray:
        """The stored specimen that best matched, for the overlay.

        Falls back to the query itself when no specimen canvas could be loaded,
        which yields an overlay showing perfect agreement — visibly wrong
        rather than subtly misleading, and the warning in the response explains
        why.
        """
        if not enrolment.canvases:
            return result.query_canvas
        similarities = result.comparison.per_reference
        # canvases can be shorter than embeddings if an image failed to load.
        usable = min(len(similarities), len(enrolment.canvases))
        if usable == 0:
            return result.query_canvas
        best = max(range(usable), key=lambda i: similarities[i])
        return enrolment.canvases[best]


_service: InferenceService | None = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
        _service.load()
    return _service


def reset_service() -> None:
    """Drop the cached service. Used by tests."""
    global _service
    _service = None


# Re-exported so routers can catch it without importing from ml directly.
__all__.append("BlankSignatureError")
