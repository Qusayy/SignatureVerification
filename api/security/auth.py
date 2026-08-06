"""Authentication and authorisation.

Every request that touches customer data is authenticated, and every
verification and decision is attributed to a named employee. That attribution
is the whole basis of the audit trail — an anonymous verification record is
worth very little in a dispute.

JWT bearer tokens are used because they are stateless and simple to run on
premise. An organisation deployment would normally federate this to the existing
directory (LDAP / Entra ID / Kerberos) and keep only the authorisation logic
here; the dependency below is deliberately the single seam where that swap
happens.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import get_session
from api.models.tables import Employee
from api.settings import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=True)

_BCRYPT_ROUNDS = 12


def _prepare(password: str) -> bytes:
    """Reduce a password to a fixed 44-byte token before bcrypt sees it.

    bcrypt silently truncates anything past 72 bytes, so a long passphrase
    would have its tail ignored. SHA-256 then base64 (the ``bcrypt_sha256``
    construction) keeps the full entropy and always fits. Base64 rather than
    raw digest bytes because bcrypt also stops at the first NUL byte.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(plain), hashed.encode("ascii"))
    except (ValueError, TypeError):
        # A malformed stored hash must fail closed, not raise a 500.
        return False


def create_access_token(employee: Employee) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": employee.id,
        "username": employee.username,
        "location": employee.location,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def authenticate(session: Session, username: str, password: str) -> Employee | None:
    employee = session.execute(
        select(Employee).where(Employee.username == username)
    ).scalar_one_or_none()
    if employee is None or not employee.is_active:
        return None
    if not verify_password(password, employee.password_hash):
        return None
    return employee


def current_employee(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> Employee:
    """Resolve the authenticated employee, or fail with 401."""
    settings = get_settings()
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise credentials_error from exc

    employee_id = payload.get("sub")
    if not employee_id:
        raise credentials_error

    employee = session.get(Employee, employee_id)
    if employee is None or not employee.is_active:
        raise credentials_error
    return employee
