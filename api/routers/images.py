"""Serving stored signature images.

Images are decrypted on the fly and served to authenticated employees only.
They are never written to a public directory and never cached by intermediaries
— ``Cache-Control: no-store`` is set on every response, because a biometric
image sitting in a proxy cache is a data protection incident waiting to happen.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.models.tables import Employee
from api.security.auth import current_employee
from api.services.storage import encode_png, get_store

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("/{key:path}")
def get_image(
    key: str,
    invert: bool = False,
    _employee: Employee = Depends(current_employee),
) -> Response:
    """Serve a stored image.

    Args:
        invert: flip black and white before sending. Preprocessed canvases are
            stored ink-high on a black background, which is what the model
            consumes but reads as a photographic negative to a person. Callers
            displaying a canvas ask for ``?invert=1``; callers feeding one back
            into the pipeline do not.
    """
    store = get_store()
    try:
        image = store.get_image(key)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such image") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    if invert:
        image = 255 - image

    return Response(
        content=encode_png(image),
        media_type="image/png",
        headers={"Cache-Control": "no-store, private", "X-Content-Type-Options": "nosniff"},
    )
