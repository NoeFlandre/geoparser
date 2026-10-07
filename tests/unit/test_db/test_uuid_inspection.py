"""Inspect hand-written legacy DDL without opening the application's database."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from geoparser.db.uuid_inspection import inspect_sqlite_uuid_storage, main

pytestmark = pytest.mark.unit

LEGACY_DDL = """
CREATE TABLE project (id CHAR(32) PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE recognizer (id TEXT PRIMARY KEY);
CREATE TABLE resolver (id TEXT PRIMARY KEY);
CREATE TABLE document (
    id CHAR(32) PRIMARY KEY, text TEXT NOT NULL,
    project_id UUID NOT NULL REFERENCES project(id) ON DELETE CASCADE
);
CREATE TABLE context (
    id CHAR(32) PRIMARY KEY, tag TEXT NOT NULL,
    project_id UUID NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    UNIQUE(project_id, tag)
);
CREATE TABLE reference (
    id CHAR(32) PRIMARY KEY, start INTEGER NOT NULL, end INTEGER NOT NULL,
    document_id UUID NOT NULL REFERENCES document(id) ON DELETE CASCADE
);
CREATE TABLE recognition (
    id CHAR(32) PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES document(id) ON DELETE CASCADE
);
CREATE TABLE referent (
    id CHAR(32) PRIMARY KEY, feature_identifier TEXT NOT NULL,
    reference_id UUID NOT NULL REFERENCES reference(id) ON DELETE CASCADE
);
CREATE TABLE resolution (
    id CHAR(32) PRIMARY KEY,
    reference_id UUID NOT NULL REFERENCES reference(id) ON DELETE CASCADE
);
"""

FOREIGN_KEYS = (
    "document.project_id",
    "context.project_id",
    "reference.document_id",
    "recognition.document_id",
    "referent.reference_id",
    "resolution.reference_id",
)
CANONICAL = "abcdef1234567890abcdef1234567890"
DOCUMENT_ID = "abcdef1234567890abcdef1234567891"


def make_database(tmp_path, ddl=LEGACY_DDL):
    path = tmp_path / "existing # database?.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(ddl)
    return path


def column_report(report, name):
    return next(column for column in report.columns if column.name == name)


def insert_document(path, project_id, document_id=DOCUMENT_ID):
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO project VALUES (?, 'existing')", (project_id,))
        connection.execute(
            "INSERT INTO document VALUES (?, 'text', ?)", (document_id, project_id)
        )


def test_empty_legacy_schema_still_needs_rebuild(tmp_path):
    path = make_database(tmp_path)

    report = inspect_sqlite_uuid_storage(path)

    assert report.requires_migration
    assert not report.blocked
    assert report.schema_issues == ()
    assert report.foreign_key_violations == ()
    assert_legacy_columns(report)


def assert_legacy_columns(report):
    assert len(report.columns) == 13
    assert {
        column.name for column in report.columns if column.requires_migration
    } == set(FOREIGN_KEYS)
    for name in FOREIGN_KEYS:
        assert_empty_legacy_column(column_report(report, name))


def assert_empty_legacy_column(column):
    assert column.declared_type == "UUID"
    assert column.affinity == "NUMERIC"
    assert column.storage_classes == {}
    assert column.row_count == 0


def test_text_only_legacy_data_is_not_mistaken_for_fixed_schema(tmp_path):
    path = make_database(tmp_path)
    insert_document(path, CANONICAL)

    report = inspect_sqlite_uuid_storage(path)
    column = column_report(report, "document.project_id")

    assert report.requires_migration
    assert not report.blocked
    assert_clean_text_column(column)


def assert_clean_text_column(column):
    assert column.storage_classes == {"text": 1}
    assert column.noncanonical_text_count == 0
    assert column.nontext_count == 0
    assert column.exact_text_orphan_count == 0


@pytest.mark.parametrize("declared_type", ["CHAR(32)", "TEXT", "VARCHAR(32)"])
def test_fixed_text_affinity_preserves_numeric_looking_ids(tmp_path, declared_type):
    path = make_database(tmp_path, LEGACY_DDL.replace(" UUID ", f" {declared_type} "))
    insert_document(path, "01234567890123456789012345678901")

    report = inspect_sqlite_uuid_storage(path)

    assert not report.requires_migration
    assert not report.blocked
    assert column_report(report, "document.project_id").storage_classes == {"text": 1}


@pytest.mark.parametrize(
    ("value", "storage_class"),
    [
        ("00000000000000000000000000000001", "integer"),
        ("01234567890123456789012345678901", "real"),
        ("12345678901234567890123456789012", "real"),
        ("00000000000000000000000000001e10", "integer"),
        (b"abcdef1234567890abcdef1234567890", "blob"),
    ],
)
def test_nontext_uuid_storage_blocks_migration(tmp_path, value, storage_class):
    path = make_database(tmp_path)
    insert_document(path, value)

    report = inspect_sqlite_uuid_storage(path)
    column = column_report(report, "document.project_id")

    assert report.blocked
    assert column.storage_classes == {storage_class: 1}
    assert column.nontext_count == 1
    assert column.row_count == 1


@pytest.mark.parametrize(
    "value", [CANONICAL.upper(), "abcdef12-3456-7890-abcd-ef1234567890", "no UUID", ""]
)
def test_noncanonical_uuid_text_blocks_migration(tmp_path, value):
    path = make_database(tmp_path)
    insert_document(path, value)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert column_report(report, "document.project_id").noncanonical_text_count == 1
    assert column_report(report, "project.id").noncanonical_text_count == 1


def test_nullable_uuid_values_are_blockers(tmp_path):
    path = make_database(tmp_path, LEGACY_DDL.replace("UUID NOT NULL", "UUID"))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO document VALUES (?, 'text', NULL)", (DOCUMENT_ID,)
        )

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert column_report(report, "document.project_id").storage_classes == {"null": 1}
    assert column_report(report, "document.project_id").nontext_count == 1


def test_numeric_parent_storage_is_also_reported(tmp_path):
    path = make_database(tmp_path, LEGACY_DDL.replace("id CHAR(32)", "id UUID"))
    insert_document(path, "12345678901234567890123456789012")

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert column_report(report, "project.id").nontext_count == 1
    assert column_report(report, "project.id").requires_migration


def test_dangling_canonical_text_is_a_foreign_key_violation(tmp_path):
    path = make_database(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO document VALUES (?, 'text', ?)", (DOCUMENT_ID, CANONICAL)
        )

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert report.foreign_key_violations == (("document", 1, "project", 0),)
    assert column_report(report, "document.project_id").exact_text_orphan_count == 1


def test_numeric_affinity_join_cannot_hide_rounded_uuid_corruption(tmp_path):
    path = make_database(tmp_path)
    first = "12345678901234567890123456789012"
    second = "12345678901234567890123456789013"
    insert_document(path, first)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO project VALUES (?, 'second')", (second,))
        assert connection.execute(
            "SELECT count(*) FROM document JOIN project ON document.project_id=project.id"
        ).fetchone() == (2,)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert column_report(report, "document.project_id").nontext_count == 1
    assert report.foreign_key_violations


def test_exact_text_checks_do_not_accept_nocase_relationships(tmp_path):
    path = make_database(
        tmp_path, LEGACY_DDL.replace("CHAR(32)", "CHAR(32) COLLATE NOCASE")
    )
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO project VALUES (?, 'project')", (CANONICAL,))
        connection.execute(
            "INSERT INTO document VALUES (?, 'text', ?)",
            (DOCUMENT_ID, CANONICAL.upper()),
        )

    report = inspect_sqlite_uuid_storage(path)

    assert report.foreign_key_violations == ()
    assert report.blocked
    assert column_report(report, "document.project_id").exact_text_orphan_count == 1


def test_missing_schema_is_reported_without_creating_it(tmp_path):
    path = make_database(tmp_path, "CREATE TABLE unrelated (value TEXT);")
    before = path.read_bytes()

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert "Missing table: project" in report.schema_issues
    assert column_report(report, "project.id").declared_type is None
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("project_id UUID", "other_id UUID", "Missing column: document.project_id"),
        ("project_id UUID", "project_id BLOB", "Unsupported type: document.project_id"),
        (
            "ON DELETE CASCADE",
            "ON DELETE RESTRICT",
            "Unsupported foreign key: document.project_id",
        ),
        (
            "id CHAR(32) PRIMARY KEY",
            "id CHAR(32)",
            "Unsupported primary key: project.id",
        ),
    ],
)
def test_unsupported_key_schemas_are_blockers(tmp_path, old, new, expected):
    path = make_database(tmp_path, LEGACY_DDL.replace(old, new, 1))

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert any(issue.startswith(expected) for issue in report.schema_issues)


def test_foreign_key_check_failure_is_reported(tmp_path):
    path = make_database(
        tmp_path, LEGACY_DDL.replace("id CHAR(32) PRIMARY KEY", "id CHAR(32)", 1)
    )

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert report.foreign_key_error is not None
    assert "foreign key mismatch" in report.foreign_key_error


def test_view_cannot_substitute_for_expected_table(tmp_path):
    ddl = LEGACY_DDL.replace(
        "CREATE TABLE project (id CHAR(32) PRIMARY KEY, name TEXT NOT NULL);",
        "CREATE VIEW project AS SELECT 'id' AS id;",
    )
    path = make_database(tmp_path, ddl)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert "Unsupported object: project is a view" in report.schema_issues


def test_older_gazetteer_layout_is_reported(tmp_path):
    path = make_database(tmp_path, LEGACY_DDL + "CREATE TABLE gazetteer (id INTEGER);")

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert "Unsupported legacy table: gazetteer" in report.schema_issues


def test_inspection_is_read_only_and_never_starts_application(tmp_path, monkeypatch):
    import geoparser.db.db as database

    path = make_database(tmp_path)
    insert_document(path, CANONICAL)
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    monkeypatch.setattr(database, "get_engine", lambda: pytest.fail("engine startup"))
    monkeypatch.setattr(
        database, "create_db_and_tables", lambda: pytest.fail("schema startup")
    )
    monkeypatch.setattr(
        database.SQLModel.metadata, "create_all", lambda: pytest.fail("DDL")
    )

    first = inspect_sqlite_uuid_storage(path)
    second = inspect_sqlite_uuid_storage(path)

    assert first == second
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before


def test_missing_path_does_not_create_database_or_directory(tmp_path):
    path = tmp_path / "missing" / "new.sqlite"

    with pytest.raises(FileNotFoundError):
        inspect_sqlite_uuid_storage(path)

    assert not path.parent.exists()


def test_directory_is_not_a_database(tmp_path):
    with pytest.raises(ValueError, match="existing SQLite file"):
        inspect_sqlite_uuid_storage(tmp_path)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecars_require_a_consistent_standalone_backup(tmp_path, suffix):
    path = make_database(tmp_path)
    sidecar = Path(str(path) + suffix)
    sidecar.write_bytes(b"do not alter")

    with pytest.raises(ValueError, match="standalone snapshot"):
        inspect_sqlite_uuid_storage(path)

    assert sidecar.read_bytes() == b"do not alter"


def test_invalid_sqlite_input_is_not_repaired(tmp_path):
    path = tmp_path / "not.sqlite"
    path.write_text("Not a database", encoding="utf-8")

    with pytest.raises(sqlite3.DatabaseError):
        inspect_sqlite_uuid_storage(path)

    assert path.read_text(encoding="utf-8") == "Not a database"


@pytest.mark.parametrize(
    ("ddl", "exit_code"),
    [(LEGACY_DDL, 1), (LEGACY_DDL.replace(" UUID ", " CHAR(32) "), 0)],
)
def test_cli_outputs_json_with_status(tmp_path, capsys, ddl, exit_code):
    path = make_database(tmp_path, ddl)

    assert main([str(path)]) == exit_code
    report = json.loads(capsys.readouterr().out)
    assert report["requires_migration"] == (exit_code == 1)
    assert not report["blocked"]
    assert len(report["columns"]) == 13


def test_cli_blocked_data_exit_code(tmp_path, capsys):
    path = make_database(tmp_path)
    insert_document(path, "00000000000000000000000000000001")

    assert main([str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["blocked"]


def test_cli_rejects_missing_path_without_json(tmp_path, capsys):
    assert main([str(tmp_path / "missing.sqlite")]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Cannot inspect" in output.err


def test_module_cli_does_not_create_configured_database(tmp_path):
    path = make_database(tmp_path)
    app_directory = tmp_path / "untouched"
    result = subprocess.run(
        [sys.executable, "-m", "geoparser.db.uuid_inspection", str(path)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GEOPARSER_DATA_DIR": str(app_directory)},
    )

    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["requires_migration"]
    assert not app_directory.exists()


@pytest.mark.parametrize("declared_type", ["INTEGER", "REAL", "", "FLOATING POINT"])
def test_other_sqlite_affinities_are_unsupported(tmp_path, declared_type):
    ddl = LEGACY_DDL.replace("project_id UUID", f"project_id {declared_type}", 1)
    path = make_database(tmp_path, ddl)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert any(
        issue.startswith("Unsupported type: document.project_id")
        for issue in report.schema_issues
    )


def test_virtual_table_is_not_an_ordinary_key_table(tmp_path):
    ddl = LEGACY_DDL.replace(
        "CREATE TABLE project (id CHAR(32) PRIMARY KEY, name TEXT NOT NULL);",
        "CREATE VIRTUAL TABLE project USING fts5(id, name);",
    )
    path = make_database(tmp_path, ddl)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert "Unsupported virtual table: project" in report.schema_issues


def test_generated_uuid_column_is_reported(tmp_path):
    ddl = LEGACY_DDL.replace(
        "project_id UUID NOT NULL REFERENCES project(id) ON DELETE CASCADE",
        f"project_id CHAR(32) GENERATED ALWAYS AS ('{CANONICAL}') VIRTUAL REFERENCES project(id) ON DELETE CASCADE",
        1,
    )
    path = make_database(tmp_path, ddl)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert (
        "Unsupported generated or hidden column: document.project_id"
        in report.schema_issues
    )


def test_missing_parent_prevents_exact_text_check(tmp_path):
    ddl = LEGACY_DDL.replace(
        "CREATE TABLE project (id CHAR(32) PRIMARY KEY, name TEXT NOT NULL);", ""
    )
    path = make_database(tmp_path, ddl)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert column_report(report, "document.project_id").exact_text_orphan_count is None


def test_wal_mode_standalone_snapshot_creates_no_sidecars(tmp_path):
    path = make_database(tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    connection.close()
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}

    report = inspect_sqlite_uuid_storage(path)

    assert not report.blocked
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before


def test_cli_entry_point_exits_with_report_status(tmp_path, monkeypatch, capsys):
    import runpy

    import geoparser.db.uuid_inspection as inspector

    path = make_database(tmp_path)
    monkeypatch.setattr(sys, "argv", ["uuid_inspection", str(path)])

    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(inspector.__file__), run_name="__main__")

    assert result.value.code == 1
    assert json.loads(capsys.readouterr().out)["snapshot_path"] == str(path)


def test_composite_foreign_key_is_not_a_supported_single_uuid_relationship(tmp_path):
    ddl = LEGACY_DDL.replace(
        "name TEXT NOT NULL);", "name TEXT NOT NULL, UNIQUE(id, name));", 1
    ).replace(
        "project_id UUID NOT NULL REFERENCES project(id) ON DELETE CASCADE",
        "project_id UUID NOT NULL, FOREIGN KEY(project_id, text) REFERENCES project(id, name) ON DELETE CASCADE",
        1,
    )
    path = make_database(tmp_path, ddl)

    report = inspect_sqlite_uuid_storage(path)

    assert report.blocked
    assert any(
        issue.startswith("Unsupported foreign key: document.project_id")
        for issue in report.schema_issues
    )


def test_exact_orphan_checks_use_indexed_parent_candidates(tmp_path, monkeypatch):
    path = make_database(tmp_path, LEGACY_DDL.replace(" UUID ", " CHAR(32) "))
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO project VALUES (?, 'project')", (CANONICAL,))
        documents = [(f"{value:032x}", CANONICAL) for value in range(1, 2001)]
        connection.executemany("INSERT INTO document VALUES (?, 'text', ?)", documents)
        connection.executemany(
            "INSERT INTO reference VALUES (?, 0, 4, ?)",
            [(identifier, identifier) for identifier, _ in documents],
        )
    connect = sqlite3.connect
    instruction_batches = []

    def budget():
        instruction_batches.append(1)
        return len(instruction_batches) > 1000

    def bounded_connection(*args, **kwargs):
        result = connect(*args, **kwargs)
        result.set_progress_handler(budget, 1000)
        return result

    monkeypatch.setattr(sqlite3, "connect", bounded_connection)

    report = inspect_sqlite_uuid_storage(path)

    assert column_report(report, "reference.document_id").exact_text_orphan_count == 0
    assert len(instruction_batches) < 1000
