# Inspect and rebuild legacy SQLite UUID storage

Geoparser now uses SQLAlchemy `Uuid` for UUID foreign keys. On SQLite this produces
`CHAR(32)`, matching the UUID primary keys. Changing the models does not change an
existing database. Neither application startup nor `create_all()` rebuilds these
columns, and this release performs no automatic UUID migration.

The affected legacy columns are:

- `document.project_id`
- `context.project_id`
- `reference.document_id`
- `recognition.document_id`
- `referent.reference_id`
- `resolution.reference_id`

Their old declared type was `UUID`. SQLite assigns that declaration NUMERIC
affinity. A UUID consisting of decimal digits, or one that resembles exponential
notation, can become an INTEGER or REAL. Leading zeros can disappear, and REAL
storage can lose digits. `CHAR(32)` has TEXT affinity and preserves the UUID's
32-character lowercase hexadecimal representation.
[SQLite documents these affinity rules and conversions.](https://www.sqlite.org/datatype3.html)

An empty legacy column still needs rebuilding. So does a legacy column whose
current values all happen to be TEXT. Future inserts remain exposed to conversion.

The persistence regression tests exercise SQLite. DDL compilation checks show
that `Uuid` matches the primary-key types on PostgreSQL, MySQL, and SQL Server,
but no live servers for those backends were tested. This fix does not introduce
or establish support for additional database backends.

## Permission and scope

Inspection is read-only. The tool has no migration option and never calls project
startup, `get_engine()`, or `create_all()`. It does not repair, checkpoint, copy,
rename, or delete a database.

Migrating real data is a separate operation that requires the database owner's
explicit approval of the source, destination, backup, rebuild, and eventual
cutover. Approval to inspect is not approval to migrate. This guide describes a
review procedure, not an executable migration script. Do not run exploratory
migration SQL against the user's database.

## Prepare a standalone snapshot

Stop application processes and other database writers. Identify the exact source
path and keep them stopped through backup preparation and any approved cutover.
Do not use an application startup command to find or open the database.

Obtain approval before creating a backup of private data. Keep it in an
owner-approved location with appropriate access controls. Use a completed,
consistent SQLite backup, such as one produced by the
[SQLite backup API](https://www.sqlite.org/backup.html). Retain an untouched backup
and a separate disposable inspection or rehearsal copy. Record the source path,
backup time, file sizes, and cryptographic hashes.

Do not copy only a live database's main file. Committed changes may still reside
in its WAL. A normal read-only SQLite connection can also need existing WAL and
shared-memory files, or permission to create sidecars.
[SQLite's WAL documentation explains these read-only requirements.](https://www.sqlite.org/wal.html#read_only_databases)

The inspector accepts only an existing standalone snapshot with no sibling
`-wal`, `-shm`, or `-journal` files. Their absence is a guard, not proof of
quiescence or a valid backup. Never remove a sidecar to make inspection proceed.
Finish a proper backup instead.

The inspector opens the snapshot with `mode=ro&immutable=1` and enables
`query_only`. Immutable mode prevents writes and bypasses locking. You must
ensure no process changes that file during inspection. Passing a changing live
file can return stale or incorrect results. See
[SQLite's immutable URI parameter](https://www.sqlite.org/uri.html#uriimmutable).

## Run the inspection

Pass the backup path explicitly. The configured application database and
`DATABASE_URL` are not selected automatically.

```bash
python -m geoparser.db.uuid_inspection /approved/backups/geoparser-snapshot.sqlite
```

The callable API has the same snapshot-only contract:

```python
from geoparser.db.uuid_inspection import inspect_sqlite_uuid_storage

report = inspect_sqlite_uuid_storage("/approved/backups/geoparser-snapshot.sqlite")
```

The JSON report contains:

- `columns`, with declared type, SQLite affinity, row count, storage-class counts,
  non-TEXT count, noncanonical TEXT count, and exact-text orphan count
- All seven UUID primary keys as well as the six foreign keys above
- `schema_issues` for missing or unsupported UUID key definitions and recognized
  incompatible legacy layouts
- `foreign_key_violations` from `PRAGMA foreign_key_check`, and
  `foreign_key_error` if SQLite cannot check the schema
- `requires_migration`, which remains true for a NUMERIC-affinity UUID declaration
- `blocked`, which means the snapshot cannot be treated as a lossless UUID
  type-only rebuild candidate

The report includes counts, not stored UUID values or document text. It checks
known UUID key definitions, not the entire schema or physical database integrity.
A clean report does not replace the complete inventory and checks below.

Exit status `0` means these checks found TEXT-affinity UUID columns and no
blockers. Status `1` means a rebuild is needed and these checks found no blockers.
Status `2` means a blocker or an input/read error. Read errors go to stderr and
may produce no JSON. Missing paths are never created.

### Stop on ambiguous or damaged data

A non-TEXT UUID, including INTEGER, REAL, BLOB, or NULL, blocks the procedure.
Noncanonical TEXT also requires separate review. Canonical storage is exactly 32
lowercase hexadecimal characters, with no hyphens, spaces, or other characters.

Do not cast numbers to text, pad zeros, round values, choose a matching parent,
normalize noncanonical strings, or drop failing rows. A REAL value may represent
several original UUIDs. Recovery requires authoritative original identifiers or
a verified earlier backup, followed by a separately approved recovery plan.

Ordinary joins can hide this problem. NUMERIC comparisons can match one rounded
child UUID to multiple distinct TEXT parent UUIDs. The inspector checks stored
classes, SQLite foreign keys, and byte-exact TEXT relationships. A successful
join, an ORM query that returned rows, or `foreign_key_check` alone is not proof
that identifiers survived unchanged.

## Rehearse a new-copy rebuild

Only proceed after approval and after resolving every blocker. Keep the original
and untouched backup unchanged. Build a new candidate database from the verified
snapshot. Never rebuild the original file in place.

1. Inventory the full schema and all table rows. Save every table definition,
   foreign-key action, primary key, UNIQUE and CHECK constraint, default, index,
   view, and trigger. Include application-specific additions. Record user and
   schema versions, encoding, and relevant database settings.
2. Prepare reviewed DDL that changes only the six UUID foreign-key declarations
   to `CHAR(32)`. If other UUID columns need changes, review that expanded scope
   separately. Preserve `UNIQUE(context.project_id, context.tag)`, every existing
   index, and all foreign-key actions.
3. On the disposable candidate, follow
   [SQLite's documented 12-step table-rebuild procedure](https://www.sqlite.org/lang_altertable.html#making_other_kinds_of_table_schema_changes).
   Disable foreign-key enforcement before opening the transaction. Save dependent
   schema objects, create replacement tables under new names, copy the data,
   remove the old candidate tables, rename replacements, and recreate affected
   indexes, triggers, and views. Do not start by renaming the original table.
4. Use explicit column lists and unchanged stored values for every copy. Do not
   use `INSERT OR IGNORE`, `REPLACE`, deduplication, or lossy casts. Keep all
   related table replacements in one transaction so intermediate relationships
   cannot become a committed result. Do not use a helper that implicitly commits
   partway through the rebuild.
5. Before committing, compare every table's row count and its complete rows
   against the baseline, keyed by the original primary key. Compare every column
   value and its SQLite storage class within each row. For tables without a
   primary key, compare complete-row multisets, including duplicate counts.
   Independent per-column multisets can miss values swapped between rows. Check
   NULLs and compare UUID TEXT byte for byte; do not rely on totals or samples.
   Confirm all declared types and all dependent schema objects against the
   reviewed inventory.
6. Run `PRAGMA foreign_key_check` and require no rows. Run
   `PRAGMA integrity_check` and require the single result `ok`. Any error, count
   mismatch, value change, missing object, or new UUID blocker requires a rollback
   and investigation. A foreign-key failure must never be fixed by dropping rows.
7. Commit only after those checks pass. Re-enable foreign-key enforcement outside
   the transaction, close and reopen the candidate, then repeat the schema,
   full-row/value, integrity, foreign-key, and UUID inspection checks on a new
   completed standalone snapshot of it. Exercise representative ORM reads and
   insert/delete/cascade behavior only on a further disposable copy.

SQLite does not allow changing foreign-key enforcement in the middle of a
transaction. Physical integrity checks also do not replace foreign-key checks.
[Review SQLite's foreign-key guidance before preparing the migration.](https://www.sqlite.org/foreignkeys.html)

## Interruption, repeat runs, and cutover

If work fails before commit, roll back the candidate transaction. If the process
is interrupted, keep the original and untouched backup. Treat the candidate as
unverified until SQLite recovery completes and every validation passes. Do not
manually delete journals or partially created objects from an uncertain copy.
The safe retry is a fresh candidate from the verified backup.

Inspection is repeatable and changes nothing. Rebuild planning must also be
idempotent. Inspect first. If all target columns already have TEXT affinity and
all checks pass, do not rebuild them again. A partially completed or unfamiliar
schema is a blocker, not a signal to repeat destructive steps. Record which
snapshot, reviewed DDL, and validation results produced each candidate.

Cutover needs separate explicit approval. Keep writers stopped, verify that the
live source has not changed since the baseline, and confirm the destination and
rollback plan. Replace or switch the configured path only to a fully verified
candidate. Retain the original and backup for the approved retention period.
Restart the application after cutover and check its reads. If the replacement
accepts new writes, rollback is no longer a simple file swap; reconcile those
writes before any separately approved rollback.
