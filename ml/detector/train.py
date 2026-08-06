"""Train the Stage A signature detector.

Uses torchvision's Faster R-CNN, which is BSD-3 licensed and therefore safe to
ship commercially — unlike the popular YOLO-based signature detectors, which
are AGPL-3.0 and would require either an Ultralytics enterprise licence or
open-sourcing the organisation's application. See ``docs/licensing.md``.

The ImageNet-pretrained backbone is available but off by default and blocked
entirely on ``--track b``, for the same reason it is blocked in the embedding
model: ImageNet's terms are research-oriented, and a production weight needs a
clean origin.

Training data is a directory of page images each paired with a JSON file
holding a ``bbox`` of ``[x, y, width, height]`` — the layout
:mod:`ml.data.synth` writes, and the layout an annotation pass over your own
forms should produce::

    forms/
      page_0001.png
      page_0001.json     {"bbox": [420, 980, 560, 180]}

Until such annotations exist, :mod:`ml.detector.heuristic` runs with no
training at all, and the API falls back to it automatically.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.config import ARTIFACT_ROOT, TRAIN, resolve_device

__all__ = ["FormDataset", "train_detector"]


class FormDataset(Dataset):
    """Pages with a single ground-truth signature box each."""

    def __init__(self, root: Path, *, max_side: int = 1024):
        self.root = Path(root)
        self.max_side = max_side
        self.items = [
            (image, image.with_suffix(".json"))
            for image in sorted(self.root.glob("*.png"))
            if image.with_suffix(".json").exists()
        ]
        if not self.items:
            raise FileNotFoundError(
                f"No annotated pages in {root}. Expected page.png alongside page.json "
                'containing {"bbox": [x, y, width, height]}.'
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label_path = self.items[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        x, y, w, h = json.loads(label_path.read_text())["bbox"]

        # Downscale large scans, adjusting the box to match. Detection does not
        # need full resolution; the crop is re-read from the original page.
        scale = min(1.0, self.max_side / max(image.shape))
        if scale < 1.0:
            image = cv2.resize(
                image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            x, y, w, h = (v * scale for v in (x, y, w, h))

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
        target = {
            # torchvision expects [x0, y0, x1, y1].
            "boxes": torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32),
            "labels": torch.ones((1,), dtype=torch.int64),  # single class: signature
        }
        return tensor, target


def _collate(batch):
    return tuple(zip(*batch, strict=True))


def build_detector(*, pretrained: bool = False, track: str = "a") -> torch.nn.Module:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if track == "b" and pretrained:
        raise ValueError(
            "ImageNet/COCO-pretrained detector weights cannot back a Track B (production) "
            "model. See docs/licensing.md."
        )

    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # background + signature
        return model

    # `weights_backbone` has its OWN default (ImageNet) and is NOT covered by
    # `weights=None`. Leaving it implicit silently downloads ImageNet weights
    # into a supposedly clean-origin model — exactly the contamination the
    # licence policy exists to prevent, and invisible unless you happen to
    # watch the network. It must be pinned to None explicitly.
    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)


def train_detector(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    dataset = FormDataset(args.forms)
    n_val = max(1, int(len(dataset) * 0.15))
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=_collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, collate_fn=_collate, num_workers=args.num_workers
    )

    model = build_detector(pretrained=args.pretrained, track=args.track).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output = Path(args.out or ARTIFACT_ROOT / f"detector_track_{args.track}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training detector on {len(train_set)} pages ({device}), validating on {len(val_set)}.")
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        total = 0.0
        for images, targets in train_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())

        scheduler.step()
        mean_iou = evaluate_iou(model, val_loader, device)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seconds": round(time.time() - started, 1),
                    "loss": round(total / max(len(train_loader), 1), 4),
                    "val_mean_iou": round(mean_iou, 4),
                }
            )
        )

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "fasterrcnn_resnet50_fpn",
                    "licence_track": "track_b_production" if args.track == "b" else "track_a_research",
                    "pretrained_init": "coco/fasterrcnn" if args.pretrained else None,
                    "val_mean_iou": mean_iou,
                },
                output,
            )

    print(f"\nBest validation mean IoU: {best_iou:.4f}\nCheckpoint: {output.resolve()}")
    return output


@torch.no_grad()
def evaluate_iou(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    """Mean IoU of the top-scoring prediction against ground truth.

    Recall matters more than precision at this stage: a loose crop still
    verifies correctly, because preprocessing crops to the ink regardless. A
    missed signature stops the employee.
    """
    from ml.detector.evaluate import iou as box_iou

    model.eval()
    scores = []
    for images, targets in loader:
        outputs = model([i.to(device) for i in images])
        for output, target in zip(outputs, targets, strict=True):
            if len(output["boxes"]) == 0:
                scores.append(0.0)
                continue
            best = output["boxes"][output["scores"].argmax()].cpu().numpy()
            truth = target["boxes"][0].numpy()
            to_xywh = lambda b: (b[0], b[1], b[2] - b[0], b[3] - b[1])  # noqa: E731
            scores.append(box_iou(to_xywh(best), to_xywh(truth)))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage A signature detector")
    parser.add_argument("--forms", type=Path, default=Path("data/synthetic/forms"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=TRAIN.num_workers)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", default=None)
    parser.add_argument("--track", choices=["a", "b"], default="a")
    parser.add_argument(
        "--pretrained", action="store_true", help="COCO-pretrained backbone (Track A only)"
    )
    parser.add_argument("--out", type=Path, default=None)
    train_detector(parser.parse_args())


if __name__ == "__main__":
    main()
