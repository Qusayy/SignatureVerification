"""Encrypted image storage.

Signature images are biometric personal data. They are encrypted before they
touch a disk or an object store, so a leaked backup or a mislaid volume does
not expose them.

Two backends behind one interface:

* **local** — encrypted files under a directory. The default, so the stack runs
  on a laptop for the demo with no infrastructure.
* **s3** — any S3-compatible object store, including the MinIO service in
  ``docker-compose.yml`` and whatever the organisation runs in its datacenter.

The encryption key must be supplied via ``SV_IMAGE_ENCRYPTION_KEY`` in
production. Without it a throwaway key is generated per process, which means
images written by one run cannot be read by the next — obvious in a demo,
catastrophic in production, and warned about loudly at startup.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from api.settings import Settings, get_settings

__all__ = ["ImageStore", "get_store", "encode_png", "decode_png"]


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return buffer.tobytes()


def decode_png(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Stored object is not a decodable image")
    return image


class ImageStore(ABC):
    """Content-addressed, encrypted image storage."""

    def __init__(self, key: str):
        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    @abstractmethod
    def _write(self, key: str, payload: bytes) -> None: ...

    @abstractmethod
    def _read(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    def put_image(self, image: np.ndarray, *, prefix: str = "images") -> str:
        key = f"{prefix}/{uuid.uuid4().hex}.png.enc"
        self._write(key, self._fernet.encrypt(encode_png(image)))
        return key

    def get_image(self, key: str) -> np.ndarray:
        try:
            return decode_png(self._fernet.decrypt(self._read(key)))
        except InvalidToken as exc:
            raise ValueError(
                f"Could not decrypt {key}. The image encryption key has changed since it was "
                "written — set SV_IMAGE_ENCRYPTION_KEY to a stable value."
            ) from exc


class LocalImageStore(ImageStore):
    def __init__(self, key: str, root: Path):
        super().__init__(key)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Refuse traversal outside the storage root.
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError(f"Refusing to access {key!r} outside the storage root")
        return path

    def _write(self, key: str, payload: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def _read(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"No stored object {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).exists()
        except ValueError:
            return False


class S3ImageStore(ImageStore):
    def __init__(self, key: str, settings: Settings):
        super().__init__(key)
        import boto3

        self.bucket = settings.storage_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.storage_endpoint,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def _write(self, key: str, payload: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)

    def _read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


_store: ImageStore | None = None


def get_store() -> ImageStore:
    global _store
    if _store is None:
        settings = get_settings()
        key = settings.resolved_encryption_key()
        if settings.storage_endpoint:
            _store = S3ImageStore(key, settings)
        else:
            _store = LocalImageStore(key, settings.storage_local_root)
    return _store


def reset_store() -> None:
    """Drop the cached store. Used by tests."""
    global _store
    _store = None
