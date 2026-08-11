"""Checkpoint provenance.

Every checkpoint records where its training data came from. Six months after a
successful POC, when someone asks "can we legally deploy this weight?", the
answer has to come from the file itself rather than from anyone's memory.

:func:`assert_deployable` is the gate to run in the deployment pipeline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.config import TRACK_B

__all__ = [
    "Provenance",
    "assert_deployable",
    "read_provenance",
    "weights_id",
    "read_weights_id",
    "UNKNOWN_WEIGHTS_ID",
]

# Recorded for artifacts produced before weights were identified, so the
# distinction between "verified to match" and "cannot be verified" survives.
UNKNOWN_WEIGHTS_ID = ""


def weights_id(state: Mapping[str, Any]) -> str:
    """A content hash of a model's weights.

    **Why not the git commit.** ``model_version`` used to be
    ``f"{architecture}@{git_commit[:8]}"``, which identifies the *code* that
    produced a weight and not the weight itself. Two consequences, both of
    which bit:

    * Every checkpoint in ``artifacts/`` reports ``signet@unknown``, because
      training ran where ``git rev-parse`` failed. Any two of them compare
      equal, so every staleness check silently passes.
    * Retraining twice at the same commit — the normal inner loop of an
      accuracy experiment — produces two different models with one version
      string.

    A hash over the tensors has neither problem: it changes when and only when
    the weights change, and it is computable from a checkpoint alone with no
    repository, no history and no build metadata.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        if not hasattr(value, "detach"):  # ints and floats appear in some heads
            digest.update(repr(value).encode("utf-8"))
            continue
        tensor = value.detach().to("cpu").contiguous()
        digest.update(f"{tuple(tensor.shape)}|{tensor.dtype}".encode())
        try:
            digest.update(tensor.numpy().tobytes())
        except (TypeError, ValueError):
            # bfloat16 and friends have no numpy equivalent. The dtype is
            # already in the digest, so widening here cannot collide two
            # genuinely different dtypes.
            digest.update(tensor.float().numpy().tobytes())
    return digest.hexdigest()[:16]


def read_weights_id(checkpoint_path: Path | str) -> str:
    """The recorded weights id of a checkpoint, computing it if absent.

    Checkpoints written before this existed carry no id. Rather than refusing
    them, recompute from ``model_state`` — the hash is a pure function of the
    weights, so a recomputed id is exactly as trustworthy as a stored one.
    """
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded = payload.get("weights_id")
    if recorded:
        return str(recorded)
    return weights_id(payload["model_state"])


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class Provenance:
    """Everything needed to judge whether a weight may be deployed."""

    licence_track: str
    sources: list[str]
    architecture: str
    n_writers: int
    n_train_images: int
    train_signers: int
    scripts: dict[str, int] = field(default_factory=dict)
    pretrained_init: str | None = None
    manifest_path: str = ""
    git_commit: str = field(default_factory=_git_commit)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""
    # The run that produced this weight. Absent from earlier checkpoints, which
    # is why every one of them claims the module-default hyperparameters
    # regardless of what was actually passed on the command line.
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_provenance(checkpoint_path: Path | str) -> Provenance:
    """Load provenance from a checkpoint without loading the weights."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "provenance" not in payload:
        raise ValueError(
            f"{checkpoint_path} carries no provenance record and cannot be cleared for "
            "deployment. Retrain with ml/embed/train.py."
        )
    return Provenance(**payload["provenance"])


def assert_deployable(checkpoint_path: Path | str) -> Provenance:
    """Raise unless this checkpoint may legally be deployed to a deployment site.

    Run this in the deployment pipeline, not by hand.
    """
    prov = read_provenance(checkpoint_path)
    problems: list[str] = []

    if prov.licence_track != TRACK_B.name:
        problems.append(
            f"trained on licence track {prov.licence_track!r}, which is research-only"
        )
    if prov.pretrained_init:
        problems.append(f"initialised from {prov.pretrained_init!r}, whose licence is not clean")

    if problems:
        raise PermissionError(
            f"{Path(checkpoint_path).name} is not deployable: "
            + "; ".join(problems)
            + ". See docs/licensing.md."
        )
    return prov


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or gate a checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero unless the checkpoint is cleared for production",
    )
    args = parser.parse_args()

    if args.gate:
        prov = assert_deployable(args.checkpoint)
        print(f"{args.checkpoint.name} is cleared for production deployment.")
    else:
        prov = read_provenance(args.checkpoint)
    print(json.dumps(prov.to_dict(), indent=2))


if __name__ == "__main__":
    main()
