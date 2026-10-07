# UUID storage checkpoint for local continuation

Work stopped on 2026-10-07 at the owner's request. This branch preserves the
implementation, tests, documentation, and verification evidence. It is not ready
to merge. No user database was opened, migrated, or modified.

## Scope

Issue: https://github.com/NoeFlandre/geoparser/issues/173

Branch: `fix/sqlite-uuid-affinity-173`

Base: `fe84a6261e0ec23f5e05252798013e24c0c147b4`

- Change all six project UUID foreign-key declarations from SQLAlchemy `UUID`
  to `Uuid`. The additional affected column is `context.project_id`.
- Test numeric, leading-zero, exponent-looking, and ordinary UUID storage.
  Include the exact `_seed_reference_table` insertion path that failed on PR #172.
- Provide a read-only inspector for explicit, completed standalone SQLite
  backup snapshots. It never selects the configured project database or runs
  normal database startup. It reports unsupported schema, corrupted UUID
  storage, foreign-key violations, and byte-exact relationship failures.
- Document backup preparation and an explicit copy-first rebuild procedure.
  There is no executable migration or automatic upgrade.

## Completed checks

- TDD before the six-type fix: 44 failed, 18 passed. Afterward: all 62 passed.
  Evidence is in `storage-red.txt` and `storage-green.txt`.
- Database subset: 419 passed before the inspection module was added.
- Final inspector subset: 50 passed. Inspector 173/173 and test 284/284 executable
  lines were covered. Its 57 functions had maximum CRAP 5.00. The storage tests'
  11 functions had maximum CRAP 4.00.
- Independent final focused review: 147 tests passed across storage, inspection,
  and database tests. The reviewer found no remaining blockers.
- Independent file-backed checks closed and reopened the database, verified
  JSON string inputs and UUID outputs, checked all six parent relationships,
  and confirmed selective cascades between two numeric-looking UUID branches.
- Independent inspection of 10,000 linked rows improved from 7.676 seconds to
  0.120 seconds after adding an indexed candidate lookup. The final comparison
  remains byte-exact. Snapshot bytes and directory entries stayed unchanged.
- Ruff, formatting, whole-tree ty, strict documentation, dependency import audit,
  architecture checks, source distribution, and wheel build passed before the
  last indexed-query addition. Final Ruff passed after that addition. Recheck
  the remaining tools on the exact final branch head.

The logs contain only synthetic test data. Exported logs have trailing whitespace
removed; original execution output is otherwise retained. Generated mutation trees, Python
environments, dependency caches, and temporary databases are not source changes
and are not included in this checkpoint.

## Incomplete checks and blockers

- The first unit pass had 2204 passes, nine missing-DuckDB-spatial setup errors,
  and one navigation error while the new guide was still being written.
  The extension was subsequently installed into an isolated test HOME from an
  existing official installation, and the guide was completed. A final complete
  unit run remains necessary.
- The full suite was interrupted with exit 130 after the owner requested a stop.
  Do not interpret partial progress as a passing suite.
- Targeted mutation generation completed 80 files, then statistics collection
  failed to import `tests.unit.test_db.test_uuid_inspection` from the copied
  tree. No inspector mutants were executed or scored. The cloud process reused
  a sibling worktree's editable environment. Use a fresh local environment and
  verify imports come from the intended mutation tree before retrying.
- No full mutation, final whole-tree CRAP, hosted CI, merge, or post-merge
  verification was completed. No live non-SQLite database was tested; tests
  cover SQLite persistence and backend DDL compilation only.

## Resume locally

1. Fetch the branch into a separate worktree. Preserve other branches and changes.
   Use a fresh worktree-local `.venv` with the repository-pinned uv 0.11.16 and
   `uv sync --locked`. Read the repository's current instructions.
2. Use pstack guidance pinned at
   `backnotprop/pstack@157aae39a733135e93d8b5b19ff62c6a84b0ad56`.
   Keep TDD, independent behavioral oracles, DRY modules, Ruff, ty, mutation
   testing, 100% package line coverage, and CRAP < 6. Do not weaken any gate.
3. Run the focused tests and final lint/type/documentation checks. Install the
   official DuckDB spatial extension in the isolated test environment. Complete
   the full suite, deterministic benchmark contracts, coverage, and CRAP checks
   described in CONTRIBUTING.md.
4. Fix mutation import isolation, run the inspector campaign, and address
   survivors with behavioral tests. Also run the repository's required full
   mutation scope. Tests changed, so its unchanged selector requests the full
   sweep. Do not change that selector or its allowances.
5. Open a separate draft PR for #173. Obtain independent review and green required
   checks on the exact latest head before requesting or performing an authorized
   merge. Verify the merged code and post-merge CI afterward.
6. Keep Otter PR #172 separate. Once the UUID fix is merged, refresh #172 onto the
   new main, revalidate its exact head, and follow its own review/merge process.

Run all database tests against disposable data. Do not execute the inspector
against a live database or change existing user databases. Any actual backup,
migration, data recovery, deletion, or cutover needs the owner's separate
approval. Do not reconstruct precision-lost UUIDs by casting, padding, matching
a parent, or silently discarding rows.
