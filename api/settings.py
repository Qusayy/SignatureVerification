"""API configuration.

Defaults are chosen so ``uvicorn api.main:app`` works on a laptop with no
infrastructure — SQLite and local encrypted file storage — while the same code
runs against PostgreSQL and S3-compatible object storage in the organisation's
datacenter purely through environment variables.

Every default that would be unsafe in production is validated at startup by
:meth:`Settings.production_warnings`, which the app logs loudly rather than
failing silently.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config import ARTIFACT_ROOT, DATA_ROOT

DEV_JWT_SECRET = "dev-only-insecure-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SV_", env_file=".env", extra="ignore")

    environment: str = "development"

    # --- Storage ---------------------------------------------------------
    database_url: str = f"sqlite:///{(DATA_ROOT / 'sigver.db').as_posix()}"
    storage_endpoint: str | None = None  # set to use S3/MinIO instead of local files
    storage_bucket: str = "signatures"
    storage_access_key: str | None = None
    storage_secret_key: str | None = None
    storage_local_root: Path = DATA_ROOT / "storage"

    # --- Security --------------------------------------------------------
    jwt_secret: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60
    # Fernet key for encrypting signature images at rest. Generated per process
    # when unset, which means images written by one run cannot be read by the
    # next — acceptable for a demo, fatal in production, and warned about.
    image_encryption_key: str | None = None

    # --- Model artifacts -------------------------------------------------
    checkpoint_path: Path = ARTIFACT_ROOT / "signet_track_b.pt"
    cohort_path: Path = ARTIFACT_ROOT / "cohort.npz"
    calibrator_path: Path = ARTIFACT_ROOT / "calibrator.json"
    device: str = "auto"
    # Refuse to start unless the loaded checkpoint is cleared for production.
    require_deployable_checkpoint: bool = False

    # --- Behaviour -------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    max_upload_bytes: int = 20 * 1024 * 1024
    # The system never decides. This is not configurable by design; it is
    # surfaced here so the UI and the audit log can assert it.
    advisory_only: bool = True

    def production_warnings(self) -> list[str]:
        problems: list[str] = []
        if self.jwt_secret == DEV_JWT_SECRET:
            problems.append("SV_JWT_SECRET is the built-in development value")
        elif len(self.jwt_secret.encode()) < 32:
            # RFC 7518 §3.2: an HMAC-SHA256 key must be at least as long as the
            # digest it produces, or the signature is weaker than it looks.
            problems.append(
                f"SV_JWT_SECRET is {len(self.jwt_secret.encode())} bytes; HS256 requires at "
                "least 32. Generate one with `python -c \"import secrets; "
                'print(secrets.token_urlsafe(48))"`'
            )
        if not self.image_encryption_key:
            problems.append(
                "SV_IMAGE_ENCRYPTION_KEY is unset; a throwaway key is generated per process "
                "and stored images will be unreadable after restart"
            )
        if self.database_url.startswith("sqlite"):
            problems.append("SQLite is in use; PostgreSQL is expected in the datacenter")
        if not self.storage_endpoint:
            problems.append("Local filesystem storage is in use instead of object storage")
        if not self.require_deployable_checkpoint:
            problems.append(
                "SV_REQUIRE_DEPLOYABLE_CHECKPOINT is off; a research-licensed (Track A) "
                "model could be served"
            )
        return problems

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    def resolved_encryption_key(self) -> str:
        if self.image_encryption_key:
            return self.image_encryption_key
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode()


@lru_cache
def get_settings() -> Settings:
    return Settings()


def new_secret() -> str:
    return secrets.token_urlsafe(48)
