"""
Unit tests for database configuration and test fixtures.

Tests the database setup following SQLAlchemy best practices and
test fixtures that redirect database operations to test databases.
"""

import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, text


@pytest.mark.unit
class TestTestEngineFixture:
    """Test the test_engine fixture (configured like production)."""

    def test_provides_engine(self, test_engine):
        """Test that test_engine fixture provides an engine."""
        assert isinstance(test_engine, Engine)
        assert ":memory:" in str(test_engine.url)

    def test_enables_foreign_keys(self, test_session):
        """Test that foreign keys are enabled via global event listener."""
        result = test_session.exec(text("PRAGMA foreign_keys"))
        foreign_keys_enabled = result.scalar()
        assert foreign_keys_enabled == 1

    def test_creates_tables(self, test_session):
        """Test that tables are created automatically."""
        result = test_session.exec(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='project'")
        )
        table_name = result.scalar()
        assert table_name == "project"


@pytest.mark.unit
class TestTestSessionFixture:
    """Test the test_session fixture."""

    def test_provides_session(self, test_session):
        """Test that test_session fixture provides a session."""
        assert isinstance(test_session, Session)

    def test_uses_real_get_session(self, test_session):
        """Test that test_session uses the real get_session() function."""
        # The test_session fixture uses get_session() which automatically
        # uses the test database thanks to the patch_db fixture
        # This ensures we're testing the actual production code path

        # Create a simple record to verify the session works
        from geoparser.db.crud import ProjectRepository
        from geoparser.db.models import ProjectCreate

        project_create = ProjectCreate(name="test_project")
        project = ProjectRepository.create(test_session, project_create)

        assert project.name == "test_project"
        assert project.id is not None


@pytest.mark.unit
class TestPatchDbFixture:
    """Test the patch_db autouse fixture."""

    def test_redirects_engine_access(self, test_engine):
        """Test that accessing engine from db.db uses test engine."""
        from geoparser.db.db import engine

        # The autouse patch_db fixture should redirect this to test_engine
        assert ":memory:" in str(engine.url)

    def test_redirects_get_session(self, test_session):
        """Test that get_session() uses test database."""
        from geoparser.db.db import get_session

        # get_session() should use the test database
        with get_session() as session:
            result = session.connection().execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='project'"
                )
            )
            table_name = result.scalar()
            assert table_name == "project"

    def test_redirects_get_connection(self, test_session):
        """Test that get_connection() uses test database."""
        from geoparser.db.db import get_connection

        # get_connection() should use the test database
        with get_connection() as connection:
            result = connection.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='project'"
                )
            )
            table_name = result.scalar()
            assert table_name == "project"


@pytest.mark.unit
class TestDatabaseCompatibilityCheck:
    """Test the legacy-database compatibility check in create_db_and_tables()."""

    @staticmethod
    def _make_engine():
        from sqlalchemy.pool import StaticPool

        return create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    @pytest.mark.parametrize("table", ["gazetteer", "source", "feature", "name"])
    def test_rejects_each_legacy_gazetteer_table(self, table):
        """Any one of the four old gazetteer tables is enough to reject."""
        # Arrange
        from unittest.mock import patch

        import geoparser.db.db as db

        legacy_engine = self._make_engine()
        with legacy_engine.connect() as connection:
            connection.execute(text(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)"))
            connection.commit()

        # Act & Assert
        with patch.object(db, "engine", legacy_engine), pytest.raises(RuntimeError):
            db.create_db_and_tables()


@pytest.mark.unit
def test_database_path_comes_from_sqlalchemy_url():
    """The database component is parsed correctly for non-SQLite URLs too."""
    from geoparser.db.db import _database_path

    assert (
        _database_path("postgresql+psycopg://user:secret@localhost:5432/geoparser")
        == "geoparser"
    )
    assert _database_path("sqlite://") == ""


@pytest.mark.unit
def test_import_does_not_create_the_database_parent_directory(tmp_path):
    """Importing db configuration leaves filesystem setup until first use."""
    database_file = tmp_path / "created-on-use" / "geoparser.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite:///{database_file}"
    project_root = Path(__file__).resolve().parents[3]

    result = subprocess.run(
        [sys.executable, "-c", "import geoparser.db.db"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not database_file.parent.exists()


@pytest.mark.unit
class TestDatabaseCompatibilityCheckCases:
    """Exercise compatibility layouts with both current and legacy schemas."""

    @staticmethod
    def _make_engine():
        from sqlalchemy.pool import StaticPool

        return create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    def test_accepts_an_empty_database(self):
        """
        A database with none of the legacy tables is usable.

        Without this the check could be inverted and still look correct: every
        legacy name would then report "present" only for tables that are in
        fact absent, and the positive cases alone would not notice.
        """
        # Arrange
        from unittest.mock import patch

        import geoparser.db.db as db

        engine = self._make_engine()

        # Act & Assert - must not raise
        with patch.object(db, "engine", engine):
            db.create_db_and_tables()

    def test_accepts_a_referent_table_that_has_feature_identifier(self):
        """The current referent layout is accepted."""
        # Arrange
        from unittest.mock import patch

        import geoparser.db.db as db

        engine = self._make_engine()
        with engine.connect() as connection:
            connection.execute(
                text(
                    "CREATE TABLE referent "
                    "(id INTEGER PRIMARY KEY, feature_identifier TEXT)"
                )
            )
            connection.commit()

        # Act & Assert - must not raise
        with patch.object(db, "engine", engine):
            db.create_db_and_tables()

    def test_rejection_names_the_database_file(self):
        """The error tells the user which file to delete."""
        # Arrange
        from unittest.mock import patch

        import geoparser.db.db as db

        legacy_engine = self._make_engine()
        with legacy_engine.connect() as connection:
            connection.execute(text("CREATE TABLE gazetteer (id INTEGER PRIMARY KEY)"))
            connection.commit()

        # Act & Assert
        with (
            patch.object(db, "engine", legacy_engine),
            pytest.raises(RuntimeError, match=re.escape(str(db.db_path))),
        ):
            db.create_db_and_tables()

    def test_raises_for_legacy_gazetteer_tables(self):
        """A database holding old gazetteer tables is rejected clearly."""
        from unittest.mock import patch

        import geoparser.db.db as db

        legacy_engine = self._make_engine()
        with legacy_engine.connect() as connection:
            connection.execute(
                text("CREATE TABLE gazetteer (id INTEGER PRIMARY KEY, name TEXT)")
            )
            connection.commit()

        with patch.object(db, "engine", legacy_engine), pytest.raises(RuntimeError):
            db.create_db_and_tables()

    def test_raises_for_legacy_referent_layout(self):
        """A referent table without feature_identifier is rejected clearly."""
        from unittest.mock import patch

        import geoparser.db.db as db

        legacy_engine = self._make_engine()
        with legacy_engine.connect() as connection:
            connection.execute(
                text(
                    "CREATE TABLE referent (id INTEGER PRIMARY KEY, feature_id INTEGER)"
                )
            )
            connection.commit()

        with patch.object(db, "engine", legacy_engine), pytest.raises(RuntimeError):
            db.create_db_and_tables()

    def test_allows_fresh_database(self):
        """An empty database is fine and gets its tables created."""
        from unittest.mock import patch

        import geoparser.db.db as db

        fresh_engine = self._make_engine()
        with patch.object(db, "engine", fresh_engine):
            db.create_db_and_tables()

        with fresh_engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='referent'"
                )
            )
            assert result.first() is not None

    def test_allows_current_database(self):
        """A current database layout is accepted."""
        import geoparser.db.db as db

        # The autouse patch_db fixture points db.engine at the test engine,
        # which already has the current tables from create_all().
        db.create_db_and_tables()


@pytest.mark.unit
class TestSetSqlitePragma:
    """Test the _set_sqlite_pragma event listener."""

    def test_enables_foreign_keys_on_connect(self, test_session):
        """Test that foreign key enforcement is switched on for connections."""
        result = test_session.exec(text("PRAGMA foreign_keys"))
        assert result.scalar() == 1

    def test_skips_non_sqlite_connections(self):
        """Test that non-SQLite connections are left unmodified."""
        from unittest.mock import Mock

        from geoparser.db.db import _set_sqlite_pragma

        connection = Mock()

        # Act - a non-sqlite3 connection should be ignored without error
        _set_sqlite_pragma(connection, None)

        # Assert
        connection.cursor.assert_not_called()


@pytest.mark.unit
def test_ensure_database_directory_skips_non_sqlite_engines(monkeypatch):
    """Non-SQLite database URLs need no local filesystem setup."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sqlalchemy.engine import make_url

    import geoparser.db.db as db

    mkdir = Mock()
    monkeypatch.setattr(
        db, "_engine", SimpleNamespace(url=make_url("postgresql://localhost/geoparser"))
    )
    monkeypatch.setattr(Path, "mkdir", mkdir)

    db._ensure_database_directory()

    mkdir.assert_not_called()


@pytest.mark.unit
def test_ensure_database_directory_creates_sqlite_parent(tmp_path, monkeypatch):
    """File-backed SQLite databases create their directory on first use."""
    from types import SimpleNamespace

    from sqlalchemy.engine import URL

    import geoparser.db.db as db

    database_file = tmp_path / "created-on-use" / "geoparser.db"
    url = URL.create("sqlite", database=str(database_file))
    monkeypatch.setattr(db, "_engine", SimpleNamespace(url=url))

    db._ensure_database_directory()

    assert database_file.parent.is_dir()


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite:///file:shared?mode=memory&cache=shared",
    ],
)
def test_ensure_database_directory_skips_non_file_sqlite_urls(url, monkeypatch):
    """Memory and URI SQLite URLs do not need parent directories."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sqlalchemy.engine import make_url

    import geoparser.db.db as db

    mkdir = Mock()
    monkeypatch.setattr(db, "_engine", SimpleNamespace(url=make_url(url)))
    monkeypatch.setattr(Path, "mkdir", mkdir)

    db._ensure_database_directory()

    mkdir.assert_not_called()


@pytest.mark.unit
def test_ensure_database_directory_is_idempotent(tmp_path, monkeypatch):
    """Repeated use succeeds when the SQLite parent already exists."""
    from types import SimpleNamespace

    from sqlalchemy.engine import URL

    import geoparser.db.db as db

    database_file = tmp_path / "nested" / "deeper" / "geoparser.db"
    monkeypatch.setattr(
        db,
        "_engine",
        SimpleNamespace(url=URL.create("sqlite", database=str(database_file))),
    )

    db._ensure_database_directory()
    db._ensure_database_directory()

    assert database_file.parent.is_dir()


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
_FOREIGN_KEY_TABLES = (
    "project",
    "recognizer",
    "resolver",
    "document",
    "reference",
    "referent",
    "resolution",
    "recognition",
)


@dataclass(frozen=True)
class _ForeignKeySeed:
    """Related identifiers needed to build the foreign-key index fixture."""

    project_ids: list[uuid.UUID]
    document_ids: list[uuid.UUID]
    reference_ids: list[uuid.UUID]
    referent_ids: list[uuid.UUID]
    resolution_ids: list[uuid.UUID]
    recognition_ids: list[uuid.UUID]
    recognizer_ids: list[str]
    resolver_ids: list[str]


def _uuid_ids(row_count: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(row_count)]


def _named_ids(prefix: str, row_count: int) -> list[str]:
    return [f"{prefix}-{index}" for index in range(row_count)]


def _new_foreign_key_seed(row_count: int) -> _ForeignKeySeed:
    return _ForeignKeySeed(
        project_ids=_uuid_ids(row_count),
        document_ids=_uuid_ids(row_count),
        reference_ids=_uuid_ids(row_count),
        referent_ids=_uuid_ids(row_count),
        resolution_ids=_uuid_ids(row_count),
        recognition_ids=_uuid_ids(row_count),
        recognizer_ids=_named_ids("recognizer", row_count),
        resolver_ids=_named_ids("resolver", row_count),
    )


def _insert_rows(connection, table: str, rows: list[dict]) -> None:
    connection.execute(SQLModel.metadata.tables[table].insert(), rows)


def _seed_lookup_tables(connection, ids: _ForeignKeySeed) -> None:
    _insert_rows(
        connection,
        "project",
        [
            {"id": value, "name": f"project-{index}"}
            for index, value in enumerate(ids.project_ids)
        ],
    )
    _insert_rows(
        connection,
        "recognizer",
        [{"id": value, "name": value, "config": {}} for value in ids.recognizer_ids],
    )
    _insert_rows(
        connection,
        "resolver",
        [{"id": value, "name": value, "config": {}} for value in ids.resolver_ids],
    )


def _seed_document_table(connection, ids: _ForeignKeySeed) -> None:
    _insert_rows(
        connection,
        "document",
        [
            {
                "id": document_id,
                "text": f"Document {index}",
                "project_id": ids.project_ids[index],
            }
            for index, document_id in enumerate(ids.document_ids)
        ],
    )


def _seed_reference_table(connection, ids: _ForeignKeySeed) -> None:
    _insert_rows(
        connection,
        "reference",
        [
            {
                "id": reference_id,
                "document_id": ids.document_ids[index],
                "recognizer_id": ids.recognizer_ids[index],
                "start": 0,
                "end": 1,
                "text": "x",
            }
            for index, reference_id in enumerate(ids.reference_ids)
        ],
    )


def _seed_derived_tables(connection, ids: _ForeignKeySeed) -> None:
    _insert_rows(
        connection,
        "referent",
        [
            {
                "id": ids.referent_ids[index],
                "reference_id": ids.reference_ids[index],
                "resolver_id": ids.resolver_ids[index],
                "gazetteer_name": "test",
                "feature_identifier": str(index),
            }
            for index in range(len(ids.referent_ids))
        ],
    )
    _insert_rows(
        connection,
        "resolution",
        [
            {
                "id": ids.resolution_ids[index],
                "reference_id": ids.reference_ids[index],
                "resolver_id": ids.resolver_ids[index],
            }
            for index in range(len(ids.resolution_ids))
        ],
    )
    _insert_rows(
        connection,
        "recognition",
        [
            {
                "id": ids.recognition_ids[index],
                "document_id": ids.document_ids[index],
                "recognizer_id": ids.recognizer_ids[index],
            }
            for index in range(len(ids.recognition_ids))
        ],
    )


def _sample_foreign_key_values(ids: _ForeignKeySeed) -> dict:
    return {
        ("document", "project_id"): ids.project_ids[0].hex,
        ("reference", "document_id"): ids.document_ids[0].hex,
        ("reference", "recognizer_id"): ids.recognizer_ids[0],
        ("referent", "reference_id"): ids.reference_ids[0].hex,
        ("referent", "resolver_id"): ids.resolver_ids[0],
        ("resolution", "reference_id"): ids.reference_ids[0].hex,
        ("resolution", "resolver_id"): ids.resolver_ids[0],
        ("recognition", "document_id"): ids.document_ids[0].hex,
        ("recognition", "recognizer_id"): ids.recognizer_ids[0],
    }


def _seed_foreign_key_index_data(engine, row_count: int = 256) -> dict:
    """Populate enough relational rows to verify SQLite's index choices."""
    import geoparser.db.models  # noqa: F401 - register tables in SQLModel metadata

    ids = _new_foreign_key_seed(row_count)
    with engine.begin() as connection:
        _seed_lookup_tables(connection, ids)
        _seed_document_table(connection, ids)
        _seed_reference_table(connection, ids)
        _seed_derived_tables(connection, ids)
    return _sample_foreign_key_values(ids)


def _assert_foreign_key_indexes_and_query_plans(engine, values: dict) -> None:
    """Assert every foreign-key index exists and is selected by SQLite."""
    with engine.connect() as connection:
        for table, column in _FOREIGN_KEY_INDEXES:
            _assert_foreign_key_index(connection, table, column)
            _assert_index_is_selected(connection, table, column, values)


def _assert_foreign_key_index(connection, table: str, column: str) -> None:
    """Check that SQLite created the index implied by the foreign key."""
    index_name = f"ix_{table}_{column}"
    indexes = {
        row[1] for row in connection.exec_driver_sql(f"PRAGMA index_list('{table}')")
    }
    assert index_name in indexes


def _assert_index_is_selected(
    connection, table: str, column: str, values: dict
) -> None:
    """Check that a representative equality lookup selects its index."""
    index_name = f"ix_{table}_{column}"
    plan = connection.exec_driver_sql(
        f"EXPLAIN QUERY PLAN SELECT id FROM {table} WHERE {column} = ?",
        (values[(table, column)],),
    ).all()
    details = " ".join(row[3] for row in plan)
    assert index_name in details


def _count_foreign_key_tables(connection) -> dict[str, int]:
    """Count rows in tables covered by the foreign-key indexes."""
    return {
        table: connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar_one()
        for table in _FOREIGN_KEY_TABLES
    }


@pytest.mark.unit
class TestForeignKeyIndexes:
    """Test fresh schema indexes and the SQLite startup migration."""

    @staticmethod
    def _make_engine():
        from sqlalchemy.pool import StaticPool

        return create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    def test_fresh_database_uses_all_foreign_key_indexes(self):
        """Fresh tables get each foreign-key index and query plans use it."""
        from unittest.mock import patch

        import geoparser.db.db as db

        engine = self._make_engine()
        try:
            with patch.object(db, "engine", engine):
                db.create_db_and_tables()
                values = _seed_foreign_key_index_data(engine)
                _assert_foreign_key_indexes_and_query_plans(engine, values)
        finally:
            engine.dispose()

    def test_existing_database_migration_is_idempotent_and_preserves_data(self):
        """Startup adds missing indexes to existing tables without data loss."""
        from unittest.mock import patch

        from sqlmodel import SQLModel

        import geoparser.db.db as db

        engine = self._make_engine()
        try:
            SQLModel.metadata.create_all(engine)
            values = _seed_foreign_key_index_data(engine)
            with engine.begin() as connection:
                for table, column in _FOREIGN_KEY_INDEXES:
                    connection.exec_driver_sql(
                        f"DROP INDEX IF EXISTS ix_{table}_{column}"
                    )
                rows_before = _count_foreign_key_tables(connection)

            with patch.object(db, "engine", engine):
                db.create_db_and_tables()
                db.create_db_and_tables()
                _assert_foreign_key_indexes_and_query_plans(engine, values)

            with engine.connect() as connection:
                rows_after = _count_foreign_key_tables(connection)
            assert rows_after == rows_before
            assert set(rows_after.values()) == {256}
        finally:
            engine.dispose()

    def test_non_sqlite_engine_skips_foreign_key_indexes(self):
        from unittest.mock import Mock, patch

        import geoparser.db.db as db

        engine = Mock()
        engine.dialect.name = "postgresql"

        with patch.object(db, "get_engine", return_value=engine):
            db._ensure_foreign_key_indexes()

        engine.begin.assert_not_called()

    def test_missing_foreign_key_column_is_skipped(self):
        from unittest.mock import patch

        import geoparser.db.db as db

        engine = self._make_engine()
        try:
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE project (id TEXT PRIMARY KEY)")

            with (
                patch.object(db, "get_engine", return_value=engine),
                patch.object(db, "_FOREIGN_KEY_INDEXES", (("project", "missing"),)),
            ):
                db._ensure_foreign_key_indexes()
        finally:
            engine.dispose()

    def test_missing_foreign_key_column_does_not_skip_later_indexes(self):
        from unittest.mock import patch

        from sqlmodel import SQLModel

        import geoparser.db.db as db

        engine = self._make_engine()
        table_name, column_name = db._FOREIGN_KEY_INDEXES[0]
        try:
            SQLModel.metadata.create_all(engine)

            with (
                patch.object(db, "get_engine", return_value=engine),
                patch.object(
                    db,
                    "_FOREIGN_KEY_INDEXES",
                    ((table_name, "missing"), (table_name, column_name)),
                ),
            ):
                db._ensure_foreign_key_indexes()

            with engine.connect() as connection:
                indexes = {
                    row[1]
                    for row in connection.exec_driver_sql(
                        f"PRAGMA index_list('{table_name}')"
                    )
                }
            assert f"ix_{table_name}_{column_name}" in indexes
        finally:
            engine.dispose()

    def test_missing_model_index_definition_raises(self):
        from unittest.mock import patch

        import geoparser.db.db as db
        import geoparser.db.models  # noqa: F401 - register tables in SQLModel metadata

        engine = self._make_engine()
        try:
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE project (id TEXT PRIMARY KEY)")

            with (
                patch.object(db, "get_engine", return_value=engine),
                patch.object(db, "_FOREIGN_KEY_INDEXES", (("project", "id"),)),
                pytest.raises(
                    RuntimeError,
                    match=r"Missing model index definition for project\.id",
                ),
            ):
                db._ensure_foreign_key_indexes()
        finally:
            engine.dispose()
