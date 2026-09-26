from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, event, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine

import geoparser.db.models  # noqa: F401
from geoparser.paths import geoparser_data_dir

# Database URL configuration (SQLite)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{geoparser_data_dir() / 'geoparser.db'}",
)

# Ensure parent directory exists
db_path = DATABASE_URL.replace("sqlite:///", "")
Path(db_path).parent.mkdir(parents=True, exist_ok=True)


# Event listener for SQLite foreign keys
# This applies to ALL Engine instances (including test engines)
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
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


# Create engine once at module level
engine: Engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    connect_args={"check_same_thread": False},
    poolclass=NullPool,  # NullPool is recommended for SQLite
    pool_pre_ping=True,  # Keeps connections fresh for long-lived apps
)


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
            raise RuntimeError(
                "Your geoparser database was created by an older version and is not compatible "
                "with this release:\n\n"
                f"{db_path}\n\n"
                "The Irchel Geoparser is still in active development, and the database format "
                "may change between releases. There is no automatic upgrade path yet, so you "
                "will need to delete the database file and reinstall the gazetteers to continue. "
                "Doing so also removes any projects and results stored in the database. "
            )
            # pragma: no mutate end


def _ensure_foreign_key_indexes() -> None:
    """Add missing SQLite foreign-key indexes without rebuilding tables."""
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as connection:
        inspector = inspect(connection)
        columns_by_table: dict[str, set[str]] = {}
        for table_name, column_name in _FOREIGN_KEY_INDEXES:
            if table_name not in columns_by_table:
                columns_by_table[table_name] = {
                    column["name"] for column in inspector.get_columns(table_name)
                }
            if column_name not in columns_by_table[table_name]:
                continue

            table = SQLModel.metadata.tables[table_name]
            index_name = f"ix_{table_name}_{column_name}"
            index = next(
                (index for index in table.indexes if index.name == index_name), None
            )
            if index is None:
                raise RuntimeError(
                    f"Missing model index definition for {table_name}.{column_name}"
                )
            index.create(connection, checkfirst=True)


def create_db_and_tables() -> None:
    """
    Create all database tables.

    Make sure all models are imported before calling this function.
    For this application, tables are created automatically at module import.
    This function is provided for explicit table creation if needed.
    """
    _check_database_compatibility()
    SQLModel.metadata.create_all(engine)
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
    session = Session(engine, expire_on_commit=False)
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
    with engine.connect() as connection:
        yield connection
