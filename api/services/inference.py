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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from api.settings import get_settings
from ml.detector.heuristic import Detection, detect_signature
from ml.preprocess.pipeline import deskew_page
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
            "cohort_normalisation": bool(verifier and verifier.cohort is not None),
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

    # -- verification -----------------------------------------------------

    def locate_signature(
        self, page: np.ndarray, *, bbox: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray, Detection | None]:
        """Find the signature on an uploaded page and return the crop.

        Args:
            bbox: an employee-supplied region. When given, detection is skipped
                entirely — a human correcting the crop must always win over the
                detector.
        """
        if bbox is not None:
            x, y, w, h = bbox
            crop = page[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                raise ValueError("The supplied crop region is empty")
            return crop, Detection(bbox, confidence=1.0, method="employee")

        # Correct page rotation before locating anything. Skew estimation needs
        # the printed structure of the whole form, which the crop no longer has.
        deskewed, _angle = deskew_page(page)
        detection = detect_signature(deskewed)
        if detection is None:
            raise ValueError(
                "No signature region found on this page. Draw the box manually or rescan."
            )
        return detection.crop(deskewed), detection

    def verify(
        self,
        page_or_crop: np.ndarray,
        enrolment: EnrolmentBundle,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        is_full_page: bool = True,
    ) -> PipelineOutput:
        """Run detection, preprocessing, scoring, and explanation.

        Raises:
            ModelNotLoaded: no usable model.
            BlankSignatureError: the crop holds too little ink to score.
            ValueError: no signature could be located on the page.
        """
        verifier = self._require()

        if is_full_page:
            crop, detection = self.locate_signature(page_or_crop, bbox=bbox)
        else:
            crop, detection = page_or_crop, None

        result = verifier.verify(crop, enrolment)

        # Overlay against the *closest* specimen, not simply the first one.
        # When a customer's stored specimens differ from each other — signed
        # years apart, different pens — overlaying an arbitrary one shows the
        # employee a mismatch that the score did not actually penalise.
        overlay = difference_overlay(
            result.query_canvas, self._closest_canvas(result, enrolment)
        )

        return PipelineOutput(
            result=result,
            detection=detection,
            crop=crop,
            overlay=overlay,
            reason=reason_text(result),
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
