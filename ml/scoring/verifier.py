"""End-to-end verification: image in, advisory score out.

This is the single entry point the API calls. It owns the whole chain —
preprocess, embed, compare against the customer's specimens, cohort-normalise,
calibrate, band — so that the training pipeline and the live service can never
drift apart in how they treat an image.

Everything it returns is advisory. There is no accept/reject anywhere in this
module by design; the employee decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ml.config import ARTIFACT_ROOT, SCORING, ScoringConfig, resolve_device
from ml.preprocess.pipeline import (
    PreprocessResult,
    preprocess_signature,
    to_model_input,
)
from ml.scoring.calibrate import Band, ScoreCalibrator
from ml.scoring.compare import ComparisonScore, compare_to_references
from ml.scoring.znorm import CohortNormalizer, CohortStats

__all__ = ["Verifier", "VerificationResult", "EnrolmentBundle"]


@dataclass
class EnrolmentBundle:
    """A customer's stored specimens, prepared once at enrolment."""

    signer_id: str
    embeddings: np.ndarray  # (N, D), L2-normalised
    cohort_stats: CohortStats | None = None
    canvases: list[np.ndarray] = field(default_factory=list)  # for the overlay and copy check

    @property
    def n_references(self) -> int:
        return int(self.embeddings.shape[0])


@dataclass
class VerificationResult:
    """Everything the employee screen needs, plus everything the audit log does."""

    score: float  # 0-100 calibrated confidence
    band: Band
    guidance: str
    comparison: ComparisonScore
    normalized_score: float
    query_canvas: np.ndarray
    ink_fraction: float
    warnings: list[str] = field(default_factory=list)
    suspected_copy: bool = False
    calibrated: bool = True
    model_version: str = ""

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "band": self.band.value,
            "guidance": self.guidance,
            "normalized_score": round(self.normalized_score, 4),
            "comparison": self.comparison.to_dict(),
            "ink_fraction": round(self.ink_fraction, 5),
            "warnings": self.warnings,
            "suspected_copy": self.suspected_copy,
            "calibrated": self.calibrated,
            "model_version": self.model_version,
            "advisory_only": True,
        }


def _canvas_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary canvases."""
    if a.shape != b.shape:
        return 0.0
    mask_a, mask_b = a > 0, b > 0
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(mask_a & mask_b) / union)


class Verifier:
    """Loads a trained model and scores signatures against stored specimens."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        cohort: CohortNormalizer | None = None,
        calibrator: ScoreCalibrator | None = None,
        device: str | None = None,
        cfg: ScoringConfig = SCORING,
        model_version: str = "",
    ):
        self.device = device or resolve_device()
        self.model = model.to(self.device).eval()
        self.cohort = cohort
        self.calibrator = calibrator or ScoreCalibrator.identity()
        self.cfg = cfg
        self.model_version = model_version

    # -- construction -----------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        checkpoint: Path | str | None = None,
        *,
        cohort_path: Path | str | None = None,
        calibrator_path: Path | str | None = None,
        device: str | None = None,
    ) -> Verifier:
        """Load a verifier from the artifact directory.

        Missing cohort or calibrator files are tolerated so the stack can run
        before they have been produced, but the result is flagged
        ``calibrated=False`` and must be labelled as such in the UI.
        """
        from ml.embed.models import build_model

        checkpoint = Path(checkpoint or ARTIFACT_ROOT / "signet_track_b.pt")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

        model = build_model(payload.get("architecture", "signet"))
        model.load_state_dict(payload["model_state"])

        cohort = None
        cohort_path = Path(cohort_path or ARTIFACT_ROOT / "cohort.npz")
        if cohort_path.exists():
            cohort = CohortNormalizer.load(cohort_path)

        calibrator = None
        calibrator_path = Path(calibrator_path or ARTIFACT_ROOT / "calibrator.json")
        if calibrator_path.exists():
            calibrator = ScoreCalibrator.load(calibrator_path)

        provenance = payload.get("provenance", {})
        version = f"{payload.get('architecture', '?')}@{provenance.get('git_commit', '?')[:8]}"

        return cls(
            model,
            cohort=cohort,
            calibrator=calibrator,
            device=device,
            model_version=version,
        )

    # -- embedding --------------------------------------------------------

    def preprocess(self, image: np.ndarray, *, strict: bool = True) -> PreprocessResult:
        return preprocess_signature(image, strict=strict)

    @torch.no_grad()
    def embed_canvases(self, canvases: list[np.ndarray]) -> np.ndarray:
        """Embed preprocessed canvases. Returns (N, D), L2-normalised."""
        if not canvases:
            raise ValueError("No canvases to embed")
        batch = np.stack([to_model_input(c) for c in canvases])
        tensor = torch.from_numpy(batch).to(self.device)
        out = self.model(tensor)
        return torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy().astype(np.float64)

    def embed_images(self, images: list[np.ndarray], *, strict: bool = True) -> np.ndarray:
        canvases = [self.preprocess(img, strict=strict).image for img in images]
        return self.embed_canvases(canvases)

    # -- enrolment --------------------------------------------------------

    def enrol(self, signer_id: str, reference_images: list[np.ndarray]) -> EnrolmentBundle:
        """Prepare a customer's stored specimens for fast repeated verification.

        The cohort statistics are computed here, once, rather than on every
        verification: they only change when the customer's specimens change.
        """
        if not reference_images:
            raise ValueError(f"Customer {signer_id} has no reference signatures to enrol")

        canvases = [self.preprocess(img).image for img in reference_images]
        embeddings = self.embed_canvases(canvases)
        stats = self.cohort.enrolment_stats(embeddings) if self.cohort else None
        return EnrolmentBundle(
            signer_id=signer_id, embeddings=embeddings, cohort_stats=stats, canvases=canvases
        )

    # -- verification -----------------------------------------------------

    def verify(self, query_image: np.ndarray, enrolment: EnrolmentBundle) -> VerificationResult:
        """Score a freshly captured signature against a customer's specimens.

        Raises :class:`~ml.preprocess.pipeline.BlankSignatureError` when the
        crop holds too little ink to score. The caller should surface that as
        "no signature detected, rescan" rather than as a low score.
        """
        preprocessed = self.preprocess(query_image, strict=True)
        query_embedding = self.embed_canvases([preprocessed.image])[0]

        comparison = compare_to_references(query_embedding, enrolment.embeddings)

        warnings = list(preprocessed.warnings)
        if self.cohort is not None:
            normalized = self.cohort.snorm(
                comparison.raw,
                query_embedding,
                references=enrolment.embeddings,
                enrolment=enrolment.cohort_stats,
            )
        else:
            # Without a cohort the raw similarity is passed straight through.
            # It is not comparable across customers, so the caller must not
            # apply a shared threshold to it.
            normalized = comparison.raw
            warnings.append("no_cohort_normalisation")

        score = self.calibrator.score_0_100(normalized)
        band = self.calibrator.band(normalized, self.cfg)

        if self.calibrator.is_placeholder:
            warnings.append("uncalibrated_score_placeholder")
        if comparison.is_single_reference:
            warnings.append("single_reference_lower_confidence")

        suspected_copy = self._is_suspected_copy(preprocessed.image, enrolment, score)
        if suspected_copy:
            warnings.append("suspected_photocopy_of_stored_specimen")

        return VerificationResult(
            score=score,
            band=band,
            guidance=band.guidance,
            comparison=comparison,
            normalized_score=normalized,
            query_canvas=preprocessed.image,
            ink_fraction=preprocessed.ink_fraction,
            warnings=warnings,
            suspected_copy=suspected_copy,
            calibrated=not self.calibrator.is_placeholder,
            model_version=self.model_version,
        )

    def _is_suspected_copy(
        self, query_canvas: np.ndarray, enrolment: EnrolmentBundle, score: float
    ) -> bool:
        """Detect a photocopy of the stored specimen pasted onto the form.

        A genuine signature is never reproduced exactly — natural variation
        guarantees it. A near-pixel-perfect overlay with the stored specimen
        therefore indicates a copy, which is a fraud signal, not the strongest
        possible match. Without this check the highest-scoring case the system
        can produce is an attack.
        """
        if score < self.cfg.duplicate_score_min or not enrolment.canvases:
            return False
        return any(
            _canvas_iou(query_canvas, ref) >= self.cfg.duplicate_iou_min
            for ref in enrolment.canvases
        )


def load_default_verifier() -> Verifier:
    """Convenience loader used by the API at startup."""
    return Verifier.from_artifacts()
