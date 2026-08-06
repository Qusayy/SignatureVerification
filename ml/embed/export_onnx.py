"""Export a trained embedding model to ONNX.

Why bother, given the API already runs PyTorch: an ONNX artifact lets the organisation
serve the model from a runtime with a smaller footprint and no Python in the
serving path, and it is the natural handover format if inference later moves
into a .NET service or an existing model-serving platform.

The export is verified numerically against the PyTorch model before it is
written out. An ONNX file that loads but computes something slightly different
is worse than no ONNX file, because the difference shows up as a quiet accuracy
regression rather than an error.

Provenance travels with the export: the licence track is written into the ONNX
metadata so a Track A weight cannot be laundered into production by changing
format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.config import ARTIFACT_ROOT, PREPROCESS
from ml.embed.models import build_model

__all__ = ["export"]


def export(
    checkpoint: Path,
    output: Path,
    *,
    opset: int = 18,
    tolerance: float = 1e-4,
    batch_size: int = 4,
) -> dict:
    """Export ``checkpoint`` to ONNX and verify it numerically."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture", "signet")

    model = build_model(architecture)
    model.load_state_dict(payload["model_state"])
    model.eval()

    example = torch.rand(batch_size, 1, *PREPROCESS.crop_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output),
        input_names=["signature"],
        output_names=["embedding"],
        # Batch is dynamic so enrolment can embed several specimens at once
        # while verification embeds one.
        dynamic_axes={"signature": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    # --- numerical verification ------------------------------------------
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    provenance = payload.get("provenance", {})
    for key, value in {
        "licence_track": provenance.get("licence_track", "unknown"),
        "architecture": architecture,
        "sources": ",".join(provenance.get("sources", [])),
        "git_commit": provenance.get("git_commit", "unknown"),
        "crop_size": "x".join(str(v) for v in PREPROCESS.crop_size),
        "input_convention": "float32 [0,1], single channel, ink high, background 0",
    }.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = str(value)
    onnx.save(onnx_model, str(output))

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(example).numpy()
    actual = session.run(["embedding"], {"signature": example.numpy()})[0]

    max_diff = float(np.abs(expected - actual).max())
    if max_diff > tolerance:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX export differs from PyTorch by {max_diff:.2e} (tolerance {tolerance:.0e}). "
            "The export was deleted rather than shipped — a silently divergent model would "
            "surface as an unexplained accuracy regression."
        )

    return {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "architecture": architecture,
        "opset": opset,
        "embedding_dim": int(expected.shape[1]),
        "max_abs_difference": max_diff,
        "licence_track": provenance.get("licence_track", "unknown"),
        "size_mb": round(output.stat().st_size / (1024 * 1024), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_ROOT / "signet_track_b.pt")
    parser.add_argument("--output", type=Path, default=None)
    # 18 is the lowest opset torch's exporter implements natively; asking for
    # less triggers a lossy down-conversion pass for no benefit.
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    output = args.output or args.checkpoint.with_suffix(".onnx")
    print(json.dumps(export(args.checkpoint, output, opset=args.opset, tolerance=args.tolerance), indent=2))


if __name__ == "__main__":
    main()
