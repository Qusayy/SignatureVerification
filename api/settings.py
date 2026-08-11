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
    # Fernet key for encrypting signature images at rest. When unset outside
    # production a key is generated *once* and cached on disk — see
    # `resolved_encryption_key`. In production an unset key is fatal.
    image_encryption_key: str | None = None
    # Where the generated development key is cached. Inside DATA_ROOT, which is
    # git-ignored, so it never reaches a repository.
    dev_key_path: Path = DATA_ROOT / ".image_key"

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
                f"SV_IMAGE_ENCRYPTION_KEY is unset; a generated key cached at "
                f"{self.dev_key_path} is in use. Deleting that file makes every stored "
                "image permanently unreadable"
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
        """The Fernet key used to encrypt stored images.

        Outside production a missing key is generated **once** and cached on
        disk rather than regenerated per process. That distinction is the whole
        point of this method: with a per-process key, `python -m api.seed`
        writes images under one key and `uvicorn` reads them under another, so
        every specimen thumbnail comes back as a 500 with no obvious cause.
        The cached key is worth far less than a real one — it sits beside the
        data it protects — but it is a development convenience, not a security
        control, and production refuses to run without an explicit key.
        """
        if self.image_encryption_key:
            return self.image_encryption_key

        from cryptography.fernet import Fernet

        if self.is_production:
            raise RuntimeError(
                "SV_IMAGE_ENCRYPTION_KEY must be set when SV_ENVIRONMENT=production. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; "
                'print(Fernet.generate_key().decode())"'
            )

        path = self.dev_key_path
        if path.exists():
            cached = path.read_text(encoding="utf-8").strip()
            if cached:
                return cached

        key = Fernet.generate_key().decode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key, encoding="utf-8")
        try:  # best effort; POSIX only, and a no-op on Windows
            path.chmod(0o600)
        except OSError:
            pass
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()


def new_secret() -> str:
    return secrets.token_urlsafe(48)
