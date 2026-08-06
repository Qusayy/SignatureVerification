"""Checkpoint provenance.

Every checkpoint records where its training data came from. Six months after a
successful POC, when someone asks "can we legally deploy this weight?", the
answer has to come from the file itself rather than from anyone's memory.

:func:`assert_deployable` is the gate to run in the deployment pipeline.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.config import TRACK_B

__all__ = ["Provenance", "assert_deployable", "read_provenance"]


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
