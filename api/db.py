"""Database session management."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.models.tables import Base
from api.settings import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url

    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # SQLite needs the directory to exist and, for FastAPI's threadpool,
        # cross-thread connections.
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(connection, _record):  # pragma: no cover - driver level
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Create tables if absent.

    Fine for the POC. Production should use Alembic migrations so schema
    changes are reviewable and reversible.
    """
    Base.metadata.create_all(bind=get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a transactional session.

    The trailing commit is a safety net, not the primary one. FastAPI runs
    dependency teardown *after* the response has been sent, so an endpoint that
    relies on it returns an id to the client before the row is durable — and a
    client that immediately acts on that id races the commit and gets a 404.
    Mutating endpoints must therefore call :func:`commit` themselves before
    building their response. See ``api/routers/verify.py``.
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def commit(session: Session) -> None:
    """Make pending changes durable before the response is built.

    Call this in any endpoint that hands the client an identifier it may use in
    a follow-up request.
    """
    session.commit()


def reset_engine() -> None:
    """Drop cached engine and sessionmaker. Used by tests."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
