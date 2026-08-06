"""Upload handling."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from api.settings import get_settings

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/bmp",
    "image/webp",
    "application/octet-stream",  # some scanner drivers send this
}


async def read_upload(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image, enforcing the size and type limits.

    Decoding is done with OpenCV rather than trusting the declared content
    type: the file is scored as an image regardless of what it claims to be, so
    that is what must be validated.
    """
    settings = get_settings()

    if upload.content_type and upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported content type {upload.content_type}",
        )

    payload = await upload.read()
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{upload.filename} is empty")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"{upload.filename} exceeds the {settings.max_upload_bytes // (1024 * 1024)}MB limit",
        )

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{upload.filename} is not a readable image"
        )
    return image
