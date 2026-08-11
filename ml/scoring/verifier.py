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
from ml.preprocess.trace import PipelineTrace, vector_strip
from ml.scoring.calibrate import Band, ScoreCalibrator
from ml.scoring.compare import ComparisonScore, compare_to_references, intra_reference_mean
from ml.scoring.znorm import CohortNormalizer, CohortStats

__all__ = ["Verifier", "VerificationResult", "EnrolmentBundle", "ArtifactMismatch"]


@dataclass
class EnrolmentBundle:
    """A customer's stored specimens, prepared once at enrolment."""

    signer_id: str
    embeddings: np.ndarray  # (N, D), L2-normalised
    cohort_stats: CohortStats | None = None
    canvases: list[np.ndarray] = field(default_factory=list)  # for the overlay and copy check
    # Mean pairwise similarity among this customer's own specimens, computed
    # once at enrolment. None means "not cached" and it is recomputed per
    # verification, which is correct but wasteful.
    reference_mean: float | None = None

    @property
    def n_references(self) -> int:
        return int(self.embeddings.shape[0])

    def resolved_reference_mean(self) -> float:
        if self.reference_mean is not None:
            return self.reference_mean
        return intra_reference_mean(self.embeddings)


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


class ArtifactMismatch(RuntimeError):
    """A cohort or calibrator does not belong to the loaded weights."""


def _assert_same_weights(path: Path, artifact_id: str, model_id: str, checkpoint: Path) -> None:
    """Refuse an artifact that was not produced by these weights.

    An unstamped artifact predates the check and cannot be verified either way.
    Refusing it too is deliberate: regenerating is cheap, and the failure this
    guards against is silent and produces confident wrong numbers.
    """
    if artifact_id == model_id:
        return

    detail = (
        f"was produced by weights {artifact_id}"
        if artifact_id
        else "carries no weights stamp, so it predates this check and cannot be verified"
    )
    raise ArtifactMismatch(
        f"{path.name} {detail}, but the loaded checkpoint is {model_id}.\n"
        "Scores computed from mismatched artifacts land in a normal-looking range and "
        "are meaningless. Regenerate both against this checkpoint:\n"
        f"    python -m ml.eval.benchmark --checkpoint {checkpoint} --split test\n"
        "then re-embed the stored specimens:\n"
        "    python -m api.reenrol --apply"
    )


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

        A cohort or calibrator produced by *different weights* is not tolerated.
        Both are functions of a specific embedding space, and pairing them with
        another model produces scores in an entirely normal-looking range that
        mean nothing at all. That shipped: ``cohort.npz`` and
        ``calibrator.json`` were built by one checkpoint while the service
        loaded another, and the result was two thirds of skilled forgeries
        printing 99.5 out of 100 — indistinguishable, on screen, from the
        system working perfectly.

        Raises:
            ArtifactMismatch: the cohort or calibrator belongs to other weights.
        """
        from ml.embed.models import load_checkpoint
        from ml.embed.provenance import weights_id as compute_weights_id

        checkpoint = Path(checkpoint or ARTIFACT_ROOT / "signet_track_b.pt")
        model, payload = load_checkpoint(checkpoint)

        model_id = payload.get("weights_id") or compute_weights_id(payload["model_state"])

        cohort = None
        cohort_path = Path(cohort_path or ARTIFACT_ROOT / "cohort.npz")
        if cohort_path.exists():
            cohort = CohortNormalizer.load(cohort_path)
            _assert_same_weights(cohort_path, getattr(cohort, "weights_id", ""), model_id, checkpoint)

        calibrator = None
        calibrator_path = Path(calibrator_path or ARTIFACT_ROOT / "calibrator.json")
        if calibrator_path.exists():
            calibrator = ScoreCalibrator.load(calibrator_path)
            _assert_same_weights(calibrator_path, calibrator.weights_id, model_id, checkpoint)

        return cls(
            model,
            cohort=cohort,
            calibrator=calibrator,
            device=device,
            model_version=f"{payload.get('architecture', '?')}@{model_id}",
        )

    # -- embedding --------------------------------------------------------

    def preprocess(
        self, image: np.ndarray, *, strict: bool = True, trace: PipelineTrace | None = None
    ) -> PreprocessResult:
        return preprocess_signature(image, strict=strict, trace=trace)

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
            signer_id=signer_id,
            embeddings=embeddings,
            cohort_stats=stats,
            canvases=canvases,
            reference_mean=intra_reference_mean(embeddings),
        )

    # -- verification -----------------------------------------------------

    def verify(
        self,
        query_image: np.ndarray,
        enrolment: EnrolmentBundle,
        *,
        trace: PipelineTrace | None = None,
    ) -> VerificationResult:
        """Score a freshly captured signature against a customer's specimens.

        Raises :class:`~ml.preprocess.pipeline.BlankSignatureError` when the
        crop holds too little ink to score. The caller should surface that as
        "no signature detected, rescan" rather than as a low score.

        Args:
            trace: optional recorder capturing every intermediate stage for
                display. Observational only; the result does not depend on it.
        """
        preprocessed = self.preprocess(query_image, strict=True, trace=trace)

        if trace:
            trace.add(
                "model_input",
                "Network input",
                "The canvas is resized and centre-cropped to the exact tensor the network "
                "was trained on. This is the last thing a human can look at; from here on "
                "the signature is numbers.",
                image=(to_model_input(preprocessed.image)[0] * 255).astype(np.uint8),
                invert_for_display=True,
                tensor=f"1x{to_model_input(preprocessed.image).shape[1]}x"
                f"{to_model_input(preprocessed.image).shape[2]}",
            )

        query_embedding = self.embed_canvases([preprocessed.image])[0]

        if trace:
            trace.add(
                "embedding",
                "Embedding",
                "The network maps the signature to a point on a unit sphere. Each column "
                "below is one dimension — amber positive, blue negative. Nothing about "
                "identity is stored; only this vector is compared.",
                image=vector_strip(query_embedding),
                kind="vector",
                dimensions=int(query_embedding.shape[0]),
                model=self.model_version or "unversioned",
            )

        comparison = compare_to_references(
            query_embedding,
            enrolment.embeddings,
            writer_normalise=self.cfg.writer_normalise,
            reference_mean=enrolment.resolved_reference_mean(),
        )

        if trace:
            trace.add(
                "compare",
                "Compared to specimens",
                "Cosine similarity against every specimen on file. The specimens are "
                "combined rather than taking the single best match, so one unusually "
                "close reference cannot carry the result.",
                kind="compare",
                per_reference=[round(float(v), 4) for v in comparison.per_reference],
                combined=round(
                    float(comparison.raw + comparison.intra_reference_mean), 4
                ),
                best=round(float(comparison.max_similarity), 4),
                worst=round(float(comparison.min_similarity), 4),
                n_references=int(comparison.n_references),
            )

        warnings = list(preprocessed.warnings)
        if self.cohort is not None and self.cfg.cohort_normalise:
            normalized = self.cohort.snorm(
                comparison.raw,
                query_embedding,
                references=enrolment.embeddings,
                enrolment=enrolment.cohort_stats,
            )
        else:
            # The comparison score goes through unchanged. It is already
            # comparable across customers when writer normalisation applied,
            # because it is expressed relative to each customer's own specimen
            # consistency. Where it did not — a single specimen on file — it is
            # not, and that case is warned about below.
            normalized = comparison.raw
        if not comparison.is_writer_normalised and self.cfg.writer_normalise:
            warnings.append("score_not_writer_normalised")

        if trace:
            if comparison.is_writer_normalised:
                caption = (
                    "The similarity is re-expressed against how consistently this customer "
                    "signs — the average agreement among their own specimens. The question "
                    "becomes 'is this as close to the specimens as they are to each other?', "
                    "which is what a skilled forgery is built to defeat. Someone with a very "
                    "repeatable hand is held to a stricter standard than someone whose own "
                    "signature varies."
                )
            else:
                caption = (
                    "Only one specimen is on file, so there is no way to measure how "
                    "consistently this customer signs and the similarity is used as-is. "
                    "Scores for single-specimen customers are less comparable than others."
                )
            trace.add(
                "cohort",
                "Normalised to the customer",
                caption,
                kind="score",
                raw_similarity=round(
                    float(comparison.raw + comparison.intra_reference_mean), 4
                ),
                specimen_agreement=round(float(comparison.intra_reference_mean), 4),
                normalised=round(float(normalized), 4),
                method=(
                    "S-norm (cohort)"
                    if (self.cohort is not None and self.cfg.cohort_normalise)
                    else "writer-internal"
                    if comparison.is_writer_normalised
                    else "none (raw passed through)"
                ),
            )

        score = self.calibrator.score_0_100(normalized)
        band = self.calibrator.band(normalized, self.cfg)

        if trace:
            trace.add(
                "calibration",
                "Calibrated to a score",
                "An isotonic calibration fitted on held-out signers converts the "
                "normalised similarity into a 0-100 confidence, so the number means the "
                "same thing for every customer. The band is advice; the decision is yours.",
                kind="score",
                normalised=round(float(normalized), 4),
                score=round(float(score), 1),
                band=band.value,
                calibrated=not self.calibrator.is_placeholder,
            )

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
