"""Read-only UUID checks for completed, standalone SQLite backup snapshots.

Never pass a live database. Immutable mode bypasses locking and assumes that no
process can change the snapshot. Sidecar checks cannot establish that guarantee.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

_FOREIGN_KEYS = {
    "document": ("project_id", "project"),
    "context": ("project_id", "project"),
    "reference": ("document_id", "document"),
    "recognition": ("document_id", "document"),
    "referent": ("reference_id", "reference"),
    "resolution": ("reference_id", "reference"),
}
_TABLES = ("project", *_FOREIGN_KEYS)
_CANONICAL_UUID = re.compile(r"[0-9a-f]{32}")
_STORAGE_SQL = 'SELECT typeof("{column}"), count(*) FROM "{table}" GROUP BY 1'
_TEXT_SQL = 'SELECT "{column}" FROM "{table}" WHERE typeof("{column}") = \'text\''
_ORPHANS_SQL = """
SELECT count(*) FROM "{table}" AS child
WHERE typeof(child."{column}") = 'text'
AND NOT EXISTS (
    SELECT 1 FROM "{parent}" AS parent
    WHERE typeof(parent.id) = 'text'
    AND parent.id = CAST(child."{column}" AS TEXT)
    AND CAST(parent.id AS BLOB) = CAST(child."{column}" AS BLOB)
)
"""


@dataclass(frozen=True)
class ColumnInspection:
    """Counts for one UUID column, without including private record values."""

    name: str
    declared_type: str | None
    affinity: str | None
    storage_classes: dict[str, int]
    row_count: int
    noncanonical_text_count: int
    nontext_count: int
    exact_text_orphan_count: int | None
    requires_migration: bool

    @property
    def blocked(self) -> bool:
        """Whether stored values prevent a lossless, unchanged-value rebuild."""
        return bool(
            self.noncanonical_text_count
            or self.nontext_count
            or self.exact_text_orphan_count
        )


@dataclass(frozen=True)
class UuidInspection:
    """A snapshot report, not an authorization or a migration plan."""

    snapshot_path: str
    columns: tuple[ColumnInspection, ...]
    schema_issues: tuple[str, ...]
    foreign_key_violations: tuple[tuple[str, int | None, str, int], ...]
    foreign_key_error: str | None
    requires_migration: bool
    blocked: bool


@dataclass(frozen=True)
class _ColumnDefinition:
    declared_type: str
    primary_key: int
    hidden: int


_Schema = dict[str, dict[str, _ColumnDefinition]]


def _snapshot_path(snapshot_path: str | Path) -> Path:
    path = Path(snapshot_path).expanduser().resolve(strict=True)
    if not path.is_file():
        message = "Inspection requires an existing SQLite file."
        raise ValueError(message)
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(str(path) + suffix).exists():
            message = (
                "Inspection requires a completed standalone snapshot without sidecars."
            )
            raise ValueError(message)
    return path


def _affinity(declared_type: str) -> str:
    normalized = declared_type.upper()
    groups = (
        ("INTEGER", ("INT",)),
        ("TEXT", ("CHAR", "CLOB", "TEXT")),
        ("BLOB", ("BLOB",)),
        ("REAL", ("REAL", "FLOA", "DOUB")),
    )
    for affinity, fragments in groups:
        if any(fragment in normalized for fragment in fragments):
            return affinity
    return "NUMERIC" if normalized else "BLOB"


def _table_columns(
    connection: sqlite3.Connection, table: str, issues: list[str]
) -> dict[str, _ColumnDefinition]:
    result = connection.execute(
        "SELECT type, sql FROM sqlite_schema WHERE name = ?", (table,)
    ).fetchone()
    if result is None:
        issues.append(f"Missing table: {table}")
        return {}
    if result[0] != "table":
        issues.append(f"Unsupported object: {table} is a {result[0]}")
        return {}
    if result[1].upper().startswith("CREATE VIRTUAL TABLE"):
        issues.append(f"Unsupported virtual table: {table}")
        return {}
    return {
        row[1]: _ColumnDefinition(row[2], row[5], row[6])
        for row in connection.execute("SELECT * FROM pragma_table_xinfo(?)", (table,))
    }


def _column_issues(table: str, columns: dict[str, _ColumnDefinition]) -> list[str]:
    issues = []
    names = ("id", *_FOREIGN_KEYS.get(table, ())[:1])
    for name in names:
        issues.extend(_definition_issues(f"{table}.{name}", columns.get(name)))
    primary_key = [name for name, column in columns.items() if column.primary_key]
    if primary_key != ["id"]:
        issues.append(
            f"Unsupported primary key: {table}.id must be the sole primary key"
        )
    return issues


def _definition_issues(name: str, definition: _ColumnDefinition | None) -> list[str]:
    if definition is None:
        return [f"Missing column: {name}"]
    if definition.hidden:
        return [f"Unsupported generated or hidden column: {name}"]
    if definition.declared_type.upper() == "UUID":
        return []
    if _affinity(definition.declared_type) != "TEXT":
        return [f"Unsupported type: {name} declared {definition.declared_type!r}"]
    return []


def _foreign_key_issues(connection: sqlite3.Connection, table: str) -> list[str]:
    column, parent = _FOREIGN_KEYS[table]
    definitions = connection.execute(
        'SELECT id, seq, "table", "from", "to", on_update, on_delete '
        'FROM pragma_foreign_key_list(?) WHERE "from" = ?',
        (table, column),
    ).fetchall()
    if len(definitions) != 1:
        return [f"Unsupported foreign key: {table}.{column} must have one definition"]
    definition = definitions[0]
    key_columns = connection.execute(
        "SELECT count(*) FROM pragma_foreign_key_list(?) WHERE id = ?",
        (table, definition[0]),
    ).fetchone()[0]
    if key_columns != 1:
        return [f"Unsupported foreign key: {table}.{column} belongs to a composite key"]
    expected = (0, parent, column, "id", "NO ACTION", "CASCADE")
    if tuple(definition[1:]) != expected:
        return [f"Unsupported foreign key: {table}.{column} must reference {parent}.id"]
    return []


def _read_schema(connection: sqlite3.Connection) -> tuple[_Schema, list[str]]:
    issues: list[str] = []
    schema = {table: _table_columns(connection, table, issues) for table in _TABLES}
    for table, columns in schema.items():
        issues.extend(_column_issues(table, columns))
    for table in _FOREIGN_KEYS:
        issues.extend(_foreign_key_issues(connection, table))
    issues.extend(_legacy_schema_issues(connection, schema))
    return schema, issues


def _legacy_schema_issues(connection: sqlite3.Connection, schema: _Schema) -> list[str]:
    legacy = connection.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table' "
        "AND name IN ('gazetteer', 'source', 'feature', 'name') ORDER BY name"
    )
    issues = [f"Unsupported legacy table: {row[0]}" for row in legacy]
    if "feature_identifier" not in schema["referent"]:
        issues.append("Unsupported referent schema: missing feature_identifier")
    return issues


def _exact_orphans(
    connection: sqlite3.Connection, table: str, schema: _Schema
) -> int | None:
    column, parent = _FOREIGN_KEYS[table]
    if "id" not in schema[parent]:
        return None
    query = _ORPHANS_SQL.format(table=table, column=column, parent=parent)
    return connection.execute(query).fetchone()[0]


def _inspect_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    schema: _Schema,
) -> ColumnInspection:
    definition = schema[table].get(column)
    if definition is None:
        return ColumnInspection(
            f"{table}.{column}", None, None, {}, 0, 0, 0, None, requires_migration=False
        )
    storage = dict(
        connection.execute(_STORAGE_SQL.format(table=table, column=column)).fetchall()
    )
    noncanonical = sum(
        _CANONICAL_UUID.fullmatch(row[0]) is None
        for row in connection.execute(_TEXT_SQL.format(table=table, column=column))
    )
    orphans = _exact_orphans(connection, table, schema) if column != "id" else None
    affinity = _affinity(definition.declared_type)
    return ColumnInspection(
        name=f"{table}.{column}",
        declared_type=definition.declared_type,
        affinity=affinity,
        storage_classes=storage,
        row_count=sum(storage.values()),
        noncanonical_text_count=noncanonical,
        nontext_count=sum(storage.values()) - storage.get("text", 0),
        exact_text_orphan_count=orphans,
        requires_migration=affinity != "TEXT",
    )


def _check_foreign_keys(
    connection: sqlite3.Connection,
) -> tuple[tuple[tuple[str, int | None, str, int], ...], str | None]:
    try:
        return tuple(connection.execute("PRAGMA foreign_key_check").fetchall()), None
    except sqlite3.DatabaseError as error:
        return (), str(error)


def _inspect_snapshot(connection: sqlite3.Connection, path: Path) -> UuidInspection:
    schema, issues = _read_schema(connection)
    columns = [_inspect_column(connection, table, "id", schema) for table in _TABLES]
    columns.extend(
        _inspect_column(connection, table, column, schema)
        for table, (column, _) in _FOREIGN_KEYS.items()
    )
    violations, error = _check_foreign_keys(connection)
    schema_blocked = any((issues, violations, error))
    values_blocked = any(column.blocked for column in columns)
    return UuidInspection(
        snapshot_path=str(path),
        columns=tuple(columns),
        schema_issues=tuple(issues),
        foreign_key_violations=violations,
        foreign_key_error=error,
        requires_migration=_requires_migration(columns),
        blocked=any((schema_blocked, values_blocked)),
    )


def _requires_migration(columns: list[ColumnInspection]) -> bool:
    return any(column.requires_migration for column in columns)


def inspect_sqlite_uuid_storage(snapshot_path: str | Path) -> UuidInspection:
    """Inspect an existing, completed standalone backup, never a live database.

    The caller must establish quiescence. Immutable mode prevents database and
    sidecar writes but does not lock out writers. This function rejects existing
    WAL, shared-memory, and rollback-journal sidecars; their absence is no proof
    that a database is idle or that copying its main file produced a valid backup.

    Missing files raise FileNotFoundError. Invalid inputs raise ValueError or
    sqlite3.DatabaseError. No application startup, schema creation, repair, or
    migration occurs. Stored UUID values are counted, never guessed or exposed.
    """
    path = _snapshot_path(snapshot_path)
    with closing(
        sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    ) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        return _inspect_snapshot(connection, path)


def _exit_code(report: UuidInspection) -> int:
    if report.blocked:
        return 2
    return 1 if report.requires_migration else 0


def main(argv: list[str] | None = None) -> int:
    """Write an inspection report as JSON, with no migration option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot_path",
        help="completed standalone SQLite backup, never a live database",
    )
    args = parser.parse_args(argv)
    try:
        report = inspect_sqlite_uuid_storage(args.snapshot_path)
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        sys.stderr.write(f"Cannot inspect snapshot: {error}\n")
        return 2
    sys.stdout.write(json.dumps(asdict(report), indent=2) + "\n")
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
