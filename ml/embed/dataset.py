"""Dataset and batch sampling for embedding training.

Batch composition matters more here than in ordinary image classification.
Batch-hard triplet mining can only form a triplet if the batch actually
contains, for the same writer, both another genuine sample and a forgery. A
plain shuffled loader over thousands of writers almost never does, so most
batches would contribute nothing to the forgery term and training would quietly
collapse to identity-only learning.

:class:`WriterBatchSampler` therefore builds each batch as P writers x K
samples, mixing genuine samples and forgeries for each chosen writer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ml.config import PREPROCESS
from ml.data.augment import DEFAULT_AUGMENT, AugmentConfig, augment_capture
from ml.data.manifest import Manifest, Record
from ml.preprocess.pipeline import preprocess_signature, to_model_input

__all__ = ["SignatureDataset", "WriterBatchSampler", "corpus_fingerprint"]


def _corpus_fingerprint(manifest: Manifest) -> str:
    """A short hash identifying which corpus, and which split assignment, this is.

    Covers the root, every image path, and every split label — the three things
    that decide what a cached canvas at a given position actually contains.
    Two manifests sharing a cache directory therefore cannot collide, and
    re-splitting the same manifest invalidates its entries rather than
    silently reusing them under new labels.
    """
    digest = hashlib.sha1()
    digest.update(str(manifest.root).encode("utf-8"))
    for record in sorted(manifest.records, key=lambda r: r.image_path):
        digest.update(f"{record.image_path}|{record.split}".encode())
    return digest.hexdigest()[:12]


# Public alias; the underscore version is the one used internally.
corpus_fingerprint = _corpus_fingerprint


def _record_cache_owner(cache_dir: Path, manifest: Manifest, fingerprint: str) -> None:
    """Leave a note in the cache directory saying which corpora wrote into it.

    Purely diagnostic — the keys are already collision-free. It exists so that
    a directory full of opaque .npy files can be traced back to the manifest
    that produced them, which is the first question anyone asks when a cache
    is suspected.
    """
    meta_path = cache_dir / "cache_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        meta = {}

    entry = {"root": str(manifest.root), "records": len(manifest.records)}
    if meta.get(fingerprint) == entry:
        return
    meta[fingerprint] = entry
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass  # a read-only cache is usable; the note is not essential


class SignatureDataset(Dataset):
    """Serves preprocessed signature tensors with writer and genuineness labels.

    Augmentation is applied to the raw image *before* preprocessing, so the
    model trains on exactly the pipeline output it will see live.
    """

    def __init__(
        self,
        manifest: Manifest,
        split: str,
        *,
        augment: bool = False,
        augment_config: AugmentConfig = DEFAULT_AUGMENT,
        writer_index: dict[str, int] | None = None,
        cache_dir: Path | None = None,
        seed: int = 1337,
    ):
        """Args:
        writer_index: mapping from signer to identity-head class index. Pass
            the training split's index to validation/test datasets; their own
            signers are deliberately absent from it and will map to -1.
        """
        self.manifest = manifest
        self.records: list[Record] = manifest.by_split(split)  # type: ignore[arg-type]
        if not self.records:
            raise ValueError(f"No records in split {split!r}. Run ml.data.manifest split first.")

        self.split = split
        self.augment = augment
        self.augment_config = augment_config
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Namespaced by corpus, not by position in it. See `_canvas_cached`.
        self.corpus_fingerprint = _corpus_fingerprint(manifest)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _record_cache_owner(self.cache_dir, manifest, self.corpus_fingerprint)

        signers = sorted({r.signer_id for r in self.records})
        self.writer_index = writer_index or {s: i for i, s in enumerate(signers)}
        self.n_writers = len(self.writer_index)

        # Index for the PK sampler.
        self.by_writer: dict[str, list[int]] = {}
        for i, record in enumerate(self.records):
            self.by_writer.setdefault(record.signer_id, []).append(i)

    def __len__(self) -> int:
        return len(self.records)

    def _load_raw(self, record: Record) -> np.ndarray:
        path = self.manifest.resolve(record)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        return image

    def _canvas_cached(self, index: int, record: Record) -> np.ndarray:
        """Preprocess once and reuse. Only valid when not augmenting.

        The key is content-addressed — corpus fingerprint plus image path —
        rather than ``f"{split}_{index}"``. Position in a manifest is not a
        stable identity: ``ml.data.ingest`` appends, and ``manifest split``
        reshuffles which signers land in which split, so both renumber records
        without changing anything a positional key would notice.

        The consequence was specific and silent. Augmented reads bypass this
        cache entirely, so a stale entry never reaches training — it reaches
        *validation and test*, which is to say every number the project
        reports, while nothing errors and the images look plausible.
        """
        if self.cache_dir is None:
            return preprocess_signature(self._load_raw(record), strict=False).image

        stem = hashlib.sha1(record.image_path.encode("utf-8")).hexdigest()[:16]
        cache_path = self.cache_dir / f"{self.corpus_fingerprint}_{self.split}_{stem}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        canvas = preprocess_signature(self._load_raw(record), strict=False).image
        np.save(cache_path, canvas)
        return canvas

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        if self.augment:
            rng = np.random.default_rng(self.seed + index * 7919 + int(torch.randint(0, 1 << 30, (1,))))
            raw = augment_capture(self._load_raw(record), rng, self.augment_config)
            canvas = preprocess_signature(raw, strict=False).image
            tensor = to_model_input(canvas, train=True, rng=rng)
        else:
            canvas = self._canvas_cached(index, record)
            tensor = to_model_input(canvas, train=False)

        # Writer labels only exist for the split the identity head was built
        # on. Validation and test signers are disjoint from training signers by
        # design — that disjointness is what writer-independence means — so an
        # unseen signer maps to -1 rather than raising. Nothing downstream of
        # training consumes this field.
        writer = self.writer_index.get(record.signer_id, -1)

        return {
            "image": torch.from_numpy(tensor),
            "writer": torch.tensor(writer, dtype=torch.long),
            "is_genuine": torch.tensor(record.label == "genuine", dtype=torch.bool),
            "index": torch.tensor(index, dtype=torch.long),
        }


class WriterBatchSampler(Sampler[list[int]]):
    """P writers x K samples per batch, mixing genuine samples and forgeries.

    Args:
        writers_per_batch: P — how many distinct writers appear in a batch.
        samples_per_writer: K — how many samples of each. Must be >= 2 so a
            positive pair exists.
        forgery_ratio: target share of each writer's K samples that are
            forgeries. The rest are genuine. Clamped by what the writer has.
    """

    def __init__(
        self,
        dataset: SignatureDataset,
        *,
        writers_per_batch: int = 8,
        samples_per_writer: int = 4,
        forgery_ratio: float = 0.5,
        batches_per_epoch: int | None = None,
        seed: int = 1337,
    ):
        if samples_per_writer < 2:
            raise ValueError("samples_per_writer must be at least 2 to form a positive pair")

        self.dataset = dataset
        self.p = writers_per_batch
        self.k = samples_per_writer
        self.forgery_ratio = forgery_ratio
        self.rng = np.random.default_rng(seed)

        self.genuine_by_writer: dict[str, list[int]] = {}
        self.forgery_by_writer: dict[str, list[int]] = {}
        for writer, indices in dataset.by_writer.items():
            for i in indices:
                target = (
                    self.genuine_by_writer
                    if dataset.records[i].label == "genuine"
                    else self.forgery_by_writer
                )
                target.setdefault(writer, []).append(i)

        # A writer needs at least two genuine samples to anchor a triplet.
        self.eligible = [w for w, g in self.genuine_by_writer.items() if len(g) >= 2]
        if len(self.eligible) < self.p:
            self.p = max(1, len(self.eligible))
        if not self.eligible:
            raise ValueError("No writer has two genuine samples; cannot form training batches")

        self.batches_per_epoch = batches_per_epoch or max(1, len(dataset) // (self.p * self.k))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            writers = self.rng.choice(
                self.eligible, size=min(self.p, len(self.eligible)), replace=False
            )
            for writer in writers:
                genuine = self.genuine_by_writer.get(writer, [])
                forgeries = self.forgery_by_writer.get(writer, [])

                n_forgery = min(int(round(self.k * self.forgery_ratio)), len(forgeries))
                # Always keep two genuine samples so a positive pair survives.
                n_forgery = min(n_forgery, max(0, self.k - 2))
                n_genuine = self.k - n_forgery

                picks = list(
                    self.rng.choice(genuine, size=n_genuine, replace=len(genuine) < n_genuine)
                )
                if n_forgery:
                    picks += list(
                        self.rng.choice(
                            forgeries, size=n_forgery, replace=len(forgeries) < n_forgery
                        )
                    )
                batch.extend(int(i) for i in picks)
            yield batch


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "writer": torch.stack([b["writer"] for b in batch]),
        "is_genuine": torch.stack([b["is_genuine"] for b in batch]),
        "index": torch.stack([b["index"] for b in batch]),
    }


def canvas_shape() -> tuple[int, int]:
    return PREPROCESS.crop_size
