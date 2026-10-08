# Results with status (issue #141 handoff)

Status vocabulary:

- PASSED: ran on the exact content named, passed.
- FAILED: ran and failed. Listed for traceability. Resolved, as noted.
- SKIPPED: deliberately not run, with reason.
- UNRUN: not run in this environment.
- STALE: passed on an earlier state. Not rerun on the final head. The difference is noted.

Environment for all runs: locked dependencies (`uvx --from uv==0.11.16 uv sync --locked --no-default-groups --group lint --group test`), Python 3.13.16 in the project `.venv`, unless noted. The DuckDB spatial extension was installed for the gazetteer tests.

## A. Final head 72fb776 (content identical to the head of `issue-141-cli-parsers` and of the handoff branch's code)

| Result | Command | Output (exact summary line) | Status |
|---|---|---|---|
| 1 | `pytest tests/unit/test_cli tests/unit/test_quality -q -p no:cacheprovider --no-cov` | `377 passed` | PASSED |
| 2 | `ruff check .` | `All checks passed!` | PASSED |
| 3 | `ruff format --check .` | `465 files already formatted` | PASSED |
| 4 | `ty check --config-file ty-lint.toml geoparser scripts tests` | `All checks passed!` | PASSED |
| 5 | `ty check geoparser scripts tests` | `All checks passed!` | PASSED |
| 6 | Python 3.14.5 subset: the test file with `PROJECT_ROOT` inlined, project dependencies absent, `FORCE_COLOR=1`, `pytest -o addopts="" -q` | `33 passed` | PASSED |
| 7 | Same as 6 with `python -m pytest` | `33 passed` | PASSED |
| 8 | Logic probe of the module-mode program-token check on 9 usage lines, plus one leading-space line and the empty-remainder line. Expected T,T,F,F,F,F,F,T,F for the nine; the leading-space line is False. | matched expectation; `'usage: '` raises IndexError (fails closed) | PASSED (logic probe, not a test run) |
| 9 | Independent delta re-review of 72fb776 (read-only) | verdict approve | PASSED (see REVIEWS.md) |

## B. Full unit suite (on 36ac09b, one assertion before 72fb776)

| Result | Command | Output | Status |
|---|---|---|---|
| 10 | `pytest tests/unit -q` (coverage on, exit 0) | `2194 passed, 72 warnings` | STALE for 72fb776: a single assertion changed afterwards. The targeted rerun in row 1 covers that change. |
| 11 | Coverage from row 10 | `scripts/changelog.py` 98%, `scripts/check_architecture.py` 97%, `test_script_conventions.py` 99% | STALE (same reason) |

Earlier, before the 3.14 fixes, the same command showed `9 errors` from the missing DuckDB spatial extension. That was environmental. After installing the extension the suite passed. Resolved, recorded as FAILED-then-resolved.

## C. CLI differential harness (42 cases: file and module modes; help, usage errors, runtime errors, missing files, default paths)

| Result | Check | Output | Status |
|---|---|---|---|
| 12 | Python 3.13: current scripts vs baseline from the original `main` (e19b95b) | diff empty; baseline sha256 prefix `3bfd8248279887cb` | PASSED (scripts unchanged since e498ed6, so current for 72fb776) |
| 13 | Python 3.14: original (e19b95b) vs refactor | identical output | PASSED |
| 14 | Python 3.13 vs 3.14 for the refactor | differs only in the program name in module mode (`python3.14 -m scripts.changelog`); a Python-version artifact that the original also has | INFORMATIONAL |

## D. Static and dependency checks

| Result | Check | Output | Status |
|---|---|---|---|
| 15 | `deptry .` | `Success! No dependency issues found.` | STALE: run on an earlier state. Dependencies unchanged since. |
| 16 | `uv lock --check` | not run | UNRUN (no dependency change) |

## E. CRAP (unit-only coverage, `scripts/crap.py --max-crap 6`)

| Result | Check | Output | Status |
|---|---|---|---|
| 17 | Every function added or changed | all score 5 or lower (`_is_parser_call` 5.00; `build_parser` 1.00 in both scripts; `main` 4.00 and 2.00) | STALE: measured before the module-mode token change (36ac09b) and the startswith restore (72fb776). Not re-measured after either. |
| 18 | Whole-tree unit-only run | 320 existing functions at or above threshold; none in changed code | STALE: same earlier tree as row 17. Reflects unit-only coverage. CI uses combined coverage, which was NOT reproduced. |

## F. Failures seen during the work (all resolved)

| Event | Resolution |
|---|---|
| Python 3.14: module-mode help expected `usage: changelog.py`; 3.14 prints `python3.14 -m scripts.changelog` | Test made exact on the program token (36ac09b, then 72fb776). Scripts unchanged. |
| Python 3.14: golden help tests failed when run as `python -m pytest` (program name derived from launch mode) | Golden tests pin `parser.prog` (d109524) |
| FORCE_COLOR=1 on 3.14 coloured help and broke golden text | Golden tests and subprocess environment set `NO_COLOR=1` (d109524) |
| Early test file had a marker not allowed by the brief, and three functions over the CRAP threshold | Fixed on the superseded branch of work (3f9f6a0 era). Final file has none. |
| Delta review of 36ac09b: changes requested (literal `startswith("usage: ")` dropped) | Restored in 72fb776. Re-review approved. |

## G. Not run, by design or by constraint

| Item | Status | Reason |
|---|---|---|
| CI cells: Windows, macOS, Python 3.10 | UNRUN | not reproducible in this environment |
| Full unit suite on Python 3.11 and 3.12 | UNRUN here | earlier spot checks of `py_compile` and CLI paths on system 3.11.17 and 3.12.3 were reported by an earlier run and were NOT reproduced in this handoff |
| Integration and e2e suites | SKIPPED | need model downloads (forbidden for this task) |
| Mutation testing, Docker, strict MkDocs build, `quality_gauntlet.py` end to end | SKIPPED | outside the slice, or forbidden (models, Docker, paid compute) |
| Combined-coverage CRAP (CI) | UNRUN | needs integration coverage |
| Any Grid'5000, Mac, labeling or data-processing job | SKIPPED | excluded by the task |

## H. GitHub CI on 72fb776

See `CI-STATUS.md`. Status at last check: queued. UNRUN as far as this document knows.
