import typing as t
from pathlib import Path
from threading import Lock

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from geoparser.paths import geoparser_data_dir

_engine: Engine | None = None
_engine_lock = Lock()


def enable_foreign_keys(dbapi_connection: t.Any, connection_record: t.Any) -> None:  # noqa: ARG001 - signature fixed by SQLAlchemy's connect event
    """Enable foreign key enforcement for annotator database connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_database_location() -> Path:
    """Return the annotator database path under the configured data root."""
    return geoparser_data_dir() / "annotator" / "annotator.db"


def get_engine() -> Engine:
    """Return the annotator database engine, creating it on first use."""
    global _engine  # noqa: PLW0603 - lazy singleton; test fixtures patch _engine directly

    patched_engine = globals().get("engine")
    if patched_engine is not None:
        return patched_engine
    # A stale read here only costs taking the lock; the check inside decides.
    engine = _engine  # pragma: no mutate
    if engine is None:
        with _engine_lock:
            engine = _engine
            if engine is None:
                db_location = get_database_location()
                db_location.parent.mkdir(parents=True, exist_ok=True)
                engine = _engine = create_engine(f"sqlite:///{db_location}", echo=False)
                event.listen(engine, "connect", enable_foreign_keys)
    return engine


def __getattr__(name: str):
    """Preserve lazy access to the former module-level configuration names."""
    if name == "engine":
        return get_engine()
    if name == "db_location":
        return get_database_location()
    if name == "sqlite_url":
        return f"sqlite:///{get_database_location()}"
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def create_db_and_tables(engine: Engine | None = None) -> None:
    SQLModel.metadata.create_all(engine if engine is not None else get_engine())


def get_db() -> t.Iterator[Session]:
    db = Session(get_engine())
    try:
        yield db
    finally:
        db.close()
