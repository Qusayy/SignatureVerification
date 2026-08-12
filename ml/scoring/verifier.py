"""End-to-end verification: image in, advisory score out.

This is the single entry point the API calls. It owns the whole chain —
preprocess, embed, compare against the customer's stored specimen, calibrate,
band — so that the training pipeline and the live service can never drift apart
in how they treat an image.

The deployment protocol is one specimen per customer. The calibrator records the
protocol it was fitted for and this module refuses a mismatch, because a curve
fitted for one and applied to another produces plausible, wrong numbers.

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
from ml.scoring.calibrate import Band, CalibratorSchemaError, ScoreCalibrator
from ml.scoring.compare import ComparisonScore, compare_to_references, intra_reference_mean

__all__ = [
    "Verifier",
    "VerificationResult",
    "EnrolmentBundle",
    "ArtifactMismatch",
    "CalibratorUnavailable",
]


@dataclass
class EnrolmentBundle:
    """A customer's stored specimens, prepared once at enrolment."""

    signer_id: str
    embeddings: np.ndarray  # (N, D), L2-normalised
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


class ArtifactMismatch(RuntimeError):
    """A calibrator does not belong to the loaded weights."""


class CalibratorUnavailable(RuntimeError):
    """No usable calibrator, so no meaningful score can be produced.

    Raised at load time rather than tolerated, because the alternative is a dash
    on the gauge at the counter — where the operator has already captured the
    signature, already waited, and can do nothing with the non-answer. A service
    that cannot score should not accept the request in the first place.
    """


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
        calibrator: ScoreCalibrator,
        device: str | None = None,
        cfg: ScoringConfig = SCORING,
        model_version: str = "",
    ):
        self.device = device or resolve_device()
        self.model = model.to(self.device).eval()
        self.calibrator = calibrator
        self.cfg = cfg
        self.model_version = model_version

    # -- construction -----------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        checkpoint: Path | str | None = None,
        *,
        calibrator_path: Path | str | None = None,
        device: str | None = None,
        cfg: ScoringConfig = SCORING,
    ) -> Verifier:
        """Load a verifier from the artifact directory.

        Refuses rather than degrades. A calibrator that is missing, from other
        weights, from an older schema, or fitted for a different number of
        specimens cannot produce a meaningful number, and the alternative to
        refusing is a confident-looking one. Both failures have shipped here:
        artifacts from one checkpoint served alongside another printed 99.5 for
        two thirds of skilled forgeries, and a curve fitted on six specimens
        applied to one scored a 4.7% match at 69/100.

        Raises:
            ArtifactMismatch: the calibrator belongs to other weights.
            CalibratorSchemaError: the calibrator predates the current scoring.
            CalibratorUnavailable: absent, or fitted for another protocol.
        """
        from ml.embed.models import load_checkpoint
        from ml.embed.provenance import weights_id as compute_weights_id

        checkpoint = Path(checkpoint or ARTIFACT_ROOT / "signet_track_b.pt")
        model, payload = load_checkpoint(checkpoint)

        model_id = payload.get("weights_id") or compute_weights_id(payload["model_state"])

        calibrator_path = Path(calibrator_path or ARTIFACT_ROOT / "calibrator.json")
        if not calibrator_path.exists():
            raise CalibratorUnavailable(
                f"No calibrator at {calibrator_path}. Without it a similarity cannot be "
                "turned into a confidence, and serving a number anyway is how a 4.7% "
                "match came to read 69/100. Produce one:\n"
                f"    python -m ml.eval.benchmark --checkpoint {checkpoint} --split test"
            )

        # Raises CalibratorSchemaError on a pre-rework artifact, which was
        # fitted on writer-normalised margins and cannot be applied to a
        # similarity.
        calibrator = ScoreCalibrator.load(calibrator_path)
        _assert_same_weights(calibrator_path, calibrator.weights_id, model_id, checkpoint)

        if calibrator.protocol_references != cfg.calibration_references:
            raise CalibratorUnavailable(
                f"{calibrator_path.name} was fitted for "
                f"{calibrator.protocol_references} specimen(s) per customer, but this "
                f"service verifies against {cfg.calibration_references}. A curve fitted "
                "for one protocol and applied to another produces plausible, wrong "
                "numbers — that mismatch is what this check exists to prevent. "
                "Regenerate:\n"
                f"    python -m ml.eval.benchmark --checkpoint {checkpoint} --split test"
            )

        return cls(
            model,
            calibrator=calibrator,
            device=device,
            cfg=cfg,
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
        """Embed a customer's stored specimen(s) once, for repeated verification."""
        if not reference_images:
            raise ValueError(f"Customer {signer_id} has no reference signatures to enrol")

        canvases = [self.preprocess(img).image for img in reference_images]
        return EnrolmentBundle(
            signer_id=signer_id,
            embeddings=self.embed_canvases(canvases),
            canvases=canvases,
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

        comparison = compare_to_references(query_embedding, enrolment.embeddings, cfg=self.cfg)

        if trace:
            trace.add(
                "compare",
                "Compared to the stored specimen",
                "Cosine similarity between this signature and what is on file. Nothing "
                "is subtracted from it and nothing is normalised away — the number below "
                "is what the calibration curve reads.",
                kind="compare",
                per_reference=[round(float(v), 4) for v in comparison.per_reference],
                combined=round(float(comparison.similarity), 4),
                best=round(float(comparison.max_similarity), 4),
                worst=round(float(comparison.min_similarity), 4),
                n_references=int(comparison.n_references),
            )

        warnings = list(preprocessed.warnings)

        score = self.calibrator.score_0_100(comparison.similarity)
        band = self.calibrator.band(comparison.similarity)
        green_min, red_max = self.calibrator.effective_edges()

        if trace:
            reaching = self.calibrator.share_reaching(comparison.similarity, "genuine")
            forgers = self.calibrator.share_reaching(comparison.similarity, "skilled")
            context = ""
            if reaching is not None:
                context = (
                    f" About {reaching:.0%} of genuine signatures and {forgers:.0%} of "
                    "practised forgeries reach this level."
                    if forgers is not None
                    else f" About {reaching:.0%} of genuine signatures reach this level."
                )
            trace.add(
                "calibration",
                "Calibrated to a score",
                "A curve fitted on held-out signers converts the similarity into the "
                "chance this signature is genuine. The bands come from the same "
                "measurement: green admits at most "
                f"{self.calibrator.operating_points.get('green_max_far', 0.05):.0%} of "
                "forgeries. The band is advice; the decision is yours." + context,
                kind="score",
                similarity=round(float(comparison.similarity), 4),
                score=round(float(score), 1),
                band=band.value,
                green_from=green_min,
                red_below=red_max,
            )

        suspected_copy = self._is_suspected_copy(preprocessed.image, enrolment, comparison)
        if suspected_copy:
            warnings.append("suspected_photocopy_of_stored_specimen")

        return VerificationResult(
            score=score,
            band=band,
            guidance=band.guidance,
            comparison=comparison,
            normalized_score=comparison.similarity,
            query_canvas=preprocessed.image,
            ink_fraction=preprocessed.ink_fraction,
            warnings=warnings,
            suspected_copy=suspected_copy,
            # Always true on the serving path. A calibrator that cannot produce
            # a meaningful number is refused at load time rather than producing
            # a dash at the counter, where the operator has already spent the
            # time and can do nothing with it.
            calibrated=True,
            model_version=self.model_version,
        )

    def score_diagnostics(self, result: VerificationResult) -> dict:
        """Everything between the similarity and the number on screen.

        Exists because "why did a 9% match score 90?" is not answerable from
        the result alone, and the people who hit it are looking at a browser,
        not a terminal. Every quantity here is one that can be checked against
        the arithmetic shown beside it.
        """
        calibrator = self.calibrator
        similarity = result.comparison.similarity
        green_min, red_max = calibrator.effective_edges()

        domain_low = float(calibrator.x.min())
        domain_high = float(calibrator.x.max())
        clamped = None
        if similarity < domain_low:
            clamped = "below"
        elif similarity > domain_high:
            clamped = "above"

        return {
            "similarity": round(float(similarity), 5),
            "score": round(float(result.score), 1),
            "band": result.band.value,
            "green_min": green_min,
            "red_max": red_max,
            "green_max_far": calibrator.operating_points.get("green_max_far"),
            "red_max_frr": calibrator.operating_points.get("red_max_frr"),
            "genuine_share_at_or_above": calibrator.share_reaching(similarity, "genuine"),
            "impostor_share_at_or_above": calibrator.share_reaching(similarity, "skilled"),
            "calibrator_domain": [round(domain_low, 5), round(domain_high, 5)],
            "calibrator_clamped": clamped,
            "calibrator_distinct_scores": calibrator.distinct_scores,
            "calibrator_fit_samples": [calibrator.n_fit_genuine, calibrator.n_fit_impostor],
            "calibrator_thin_fit": calibrator.thin_fit,
            "protocol_references": calibrator.protocol_references,
            "model_version": self.model_version,
        }

    def _is_suspected_copy(
        self,
        query_canvas: np.ndarray,
        enrolment: EnrolmentBundle,
        comparison: ComparisonScore,
    ) -> bool:
        """Detect a photocopy of the stored specimen pasted onto the form.

        A genuine signature is never reproduced exactly — natural variation
        guarantees it. A near-pixel-perfect overlay with the stored specimen
        therefore indicates a copy, which is a fraud signal, not the strongest
        possible match. Without this check the highest-scoring case the system
        can produce is an attack.

        Gated on the **similarity**, not on the calibrated score. It used to
        require a score ≥ 99, which stopped being reachable the moment honest
        calibration lowered the ceiling — a fraud control silently disabled by
        an unrelated improvement. The similarity threshold comes from the
        calibrator (the 99.9th percentile of genuine), so it tracks the model
        rather than a constant.
        """
        if comparison.similarity < self.calibrator.duplicate_similarity_min:
            return False
        if not enrolment.canvases:
            return False
        return any(
            _canvas_iou(query_canvas, ref) >= self.cfg.duplicate_iou_min
            for ref in enrolment.canvases
        )


def load_default_verifier() -> Verifier:
    """Convenience loader used by the API at startup."""
    return Verifier.from_artifacts()
