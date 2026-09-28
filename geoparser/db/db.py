from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from sqlalchemy import Engine, event, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.schema import Index
from sqlmodel import Session, SQLModel, create_engine

import geoparser.db.models  # noqa: F401
from geoparser.paths import geoparser_data_dir

_engine: Engine | None = None
_engine_lock = Lock()


def _database_url() -> str:
    """Resolve the project database URL when the database is first used."""
    override = os.getenv("DATABASE_URL")
    if override is not None:
        return override
    return f"sqlite:///{geoparser_data_dir() / 'geoparser.db'}"


def _database_path(database_url: str) -> str:
    """Return the database component parsed from a SQLAlchemy URL."""
    return make_url(database_url).database or ""


def _sqlite_file_path(database_url: str) -> Path | None:
    """Return a filesystem path for a file-backed SQLite URL."""
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        return None
    if url.database == ":memory:" or url.database.startswith("file:"):
        return None
    return Path(url.database)


def get_database_path() -> Path | None:
    """Return the path configured for the project database."""
    current_engine = globals().get("engine", _engine)
    if current_engine is not None:
        return _sqlite_file_path(str(current_engine.url))
    return _sqlite_file_path(_database_url())


def get_engine() -> Engine:
    """Return the project database engine, creating it on first use."""
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
                database_url = _database_url()
                database_path = _sqlite_file_path(database_url)
                if database_path is not None:
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                engine = _engine = create_engine(
                    database_url,
                    echo=False,  # Set to True for SQL debugging
                    connect_args={"check_same_thread": False},
                    poolclass=NullPool,  # NullPool is recommended for SQLite
                )
    return engine


def __getattr__(name: str):
    """Preserve lazy access to the former module-level configuration names."""
    if name == "engine":
        return get_engine()
    if name == "DATABASE_URL":
        return _database_url()
    if name == "db_path":
        path = get_database_path()
        return str(path) if path is not None else None
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Event listener for SQLite foreign keys
# This applies to ALL Engine instances (including test engines)
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ARG001 - signature fixed by SQLAlchemy's connect event
    """
    Configure SQLite connections on connect.

    Enables foreign key enforcement for all SQLite connections.

    Args:
        dbapi_connection: Database API connection object
        connection_record: SQLAlchemy connection record
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _ensure_database_directory() -> None:
    """Create the parent directory when a file-backed SQLite database is used."""
    url = get_engine().url
    if url.get_backend_name() != "sqlite":
        return
    database = url.database
    if database is None or database == ":memory:" or database.startswith("file:"):
        return
    Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)


# SQL keywords and SQLite identifiers are case-insensitive, so case mutations
# of these literals cannot change what they match.
# The pragmas need `fmt: skip` to survive: without it the formatter wraps
# these lines and moves the comment off the statement, where mutmut ignores it.
_TABLE_EXISTS_SQL = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"  # pragma: no mutate  # fmt: skip
_COLUMN_EXISTS_SQL = "SELECT 1 FROM pragma_table_info('{table}') WHERE name=:column"  # pragma: no mutate  # fmt: skip
_REFERENT_TABLE = "referent"  # pragma: no mutate
_FOREIGN_KEY_INDEXES = (
    ("document", "project_id"),
    ("reference", "document_id"),
    ("reference", "recognizer_id"),
    ("referent", "reference_id"),
    ("referent", "resolver_id"),
    ("resolution", "reference_id"),
    ("resolution", "resolver_id"),
    ("recognition", "document_id"),
    ("recognition", "recognizer_id"),
)


def _check_database_compatibility() -> None:
    """
    Fail early if the database was created by an incompatible older version.

    Older releases stored gazetteer data (gazetteer/source/feature/name tables)
    inside this database and linked referents to it via a feature foreign key.
    Gazetteers now live in separate artifact files, so such databases cannot be
    used as-is.

    Raises:
        RuntimeError: If a legacy database layout is detected.
    """
    engine = get_engine()
    with engine.connect() as connection:

        def _table_exists(name: str) -> bool:
            result = connection.execute(text(_TABLE_EXISTS_SQL), {"name": name})
            return result.first() is not None

        def _table_has_column(table: str, column: str) -> bool:
            result = connection.execute(
                text(_COLUMN_EXISTS_SQL.format(table=table)), {"column": column}
            )
            return result.first() is not None

        legacy_gazetteer_tables = any(
            _table_exists(name) for name in ("gazetteer", "source", "feature", "name")
        )
        legacy_referent_layout = _table_exists(
            _REFERENT_TABLE
        ) and not _table_has_column(_REFERENT_TABLE, "feature_identifier")

        if legacy_gazetteer_tables or legacy_referent_layout:
            # pragma: no mutate start - the wording of this guidance is not
            # behaviour; a test pins that it names the database file.
            msg = (
                "Your geoparser database was created by an older version and is not compatible "
                "with this release:\n\n"
                f"{get_database_path()}\n\n"
                "The Irchel Geoparser is still in active development, and the database format "
                "may change between releases. There is no automatic upgrade path yet, so you "
                "will need to delete the database file and reinstall the gazetteers to continue. "
                "Doing so also removes any projects and results stored in the database. "
            )
            raise RuntimeError(msg)
            # pragma: no mutate end


def _columns_for_table(
    inspector: Inspector,
    table_name: str,
    columns_by_table: dict[str, set[str]],
) -> set[str]:
    if table_name not in columns_by_table:
        columns_by_table[table_name] = {
            column["name"] for column in inspector.get_columns(table_name)
        }
    return columns_by_table[table_name]


def _foreign_key_index(table_name: str, column_name: str) -> Index:
    table = SQLModel.metadata.tables[table_name]
    index_name = f"ix_{table_name}_{column_name}"
    for index in table.indexes:
        if index.name == index_name:
            return index

    msg = f"Missing model index definition for {table_name}.{column_name}"
    raise RuntimeError(msg)


def _ensure_foreign_key_index(
    connection: Connection,
    inspector: Inspector,
    table_name: str,
    column_name: str,
    columns_by_table: dict[str, set[str]],
) -> None:
    columns = _columns_for_table(inspector, table_name, columns_by_table)
    if column_name not in columns:
        return
    _foreign_key_index(table_name, column_name).create(connection, checkfirst=True)


def _ensure_foreign_key_indexes() -> None:
    """Add missing SQLite foreign-key indexes without rebuilding tables."""
    engine = get_engine()
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as connection:
        inspector = inspect(connection)
        columns_by_table: dict[str, set[str]] = {}
        for table_name, column_name in _FOREIGN_KEY_INDEXES:
            _ensure_foreign_key_index(
                connection, inspector, table_name, column_name, columns_by_table
            )


def create_db_and_tables() -> None:
    """
    Create all database tables.

    Make sure all models are imported before calling this function.
    The database engine is created on first use, so this function also
    initializes the database directory when needed.
    """
    _ensure_database_directory()
    _check_database_compatibility()
    SQLModel.metadata.create_all(get_engine())
    _ensure_foreign_key_indexes()


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Get a database session using context manager pattern.

    This is the preferred way to get a database session. The session
    is automatically closed when the context exits.

    Yields:
        SQLModel Session for database operations
    """
    _ensure_database_directory()
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_connection() -> Iterator[Connection]:
    """
    Get a database connection using context manager pattern.

    For operations that need direct connection access (like executing raw
    SQL). This is the preferred way to get a connection as it accesses the
    engine at runtime, respecting any patches applied during testing.

    Yields:
        SQLAlchemy Connection for database operations
    """
    _ensure_database_directory()
    with get_engine().connect() as connection:
        yield connection
