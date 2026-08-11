"""Database session management."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.models.tables import Base
from api.settings import get_settings

logger = logging.getLogger(__name__)

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


def schema_drift() -> dict[str, list[str]]:
    """Compare the live database against the models.

    ``create_all`` creates missing tables and does nothing else — it will not
    add a column to a table that already exists. A database carried over from
    an older revision therefore keeps its old shape, and every query touching a
    renamed or added column fails at runtime with a 500 that names a column
    rather than the real problem.

    Returns missing tables, missing columns, and columns present in the
    database but absent from the models (usually the other half of a rename).
    """
    engine = get_engine()
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())

    missing_tables: list[str] = []
    missing_columns: list[str] = []
    unexpected_columns: list[str] = []

    for name, table in Base.metadata.tables.items():
        if name not in existing:
            missing_tables.append(name)
            continue
        live = {c["name"] for c in inspector.get_columns(name)}
        expected = {c.name for c in table.columns}
        missing_columns += [f"{name}.{c}" for c in sorted(expected - live)]
        unexpected_columns += [f"{name}.{c}" for c in sorted(live - expected)]

    return {
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
    }


def schema_is_stale(drift: dict[str, list[str]] | None = None) -> bool:
    """True when the database cannot satisfy the current models.

    Missing *tables* are not staleness — ``init_db`` creates those. A missing
    *column* on an existing table is, because nothing creates it.
    """
    drift = drift if drift is not None else schema_drift()
    return bool(drift["missing_columns"])


def init_db() -> None:
    """Create absent tables and warn loudly about drift in the ones that exist.

    Fine for the POC. Production should use Alembic migrations so schema
    changes are reviewable and reversible.
    """
    Base.metadata.create_all(bind=get_engine())

    drift = schema_drift()
    if drift["missing_columns"]:
        logger.error(
            "DATABASE SCHEMA IS OUT OF DATE. Missing columns: %s. %sEvery request touching "
            "them will fail with a 500. Run `python -m api.doctor` for the fix.",
            ", ".join(drift["missing_columns"]),
            (
                f"Columns present but no longer in the models: "
                f"{', '.join(drift['unexpected_columns'])}. "
                if drift["unexpected_columns"]
                else ""
            ),
        )


def recreate_schema() -> None:
    """Drop every table and rebuild it. Destroys all stored rows.

    Used by ``api.seed --reset``. Encrypted image blobs are not touched here;
    the seed clears those separately, because orphaned blobs are harmless while
    orphaned rows are not.
    """
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


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
