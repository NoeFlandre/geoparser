# Handoff: issue #141 (partial) - parser construction in two CLI scripts

Status: candidate ready for review as a draft. Issue #141 stays open. Nothing merged, nothing closed.

This branch is `handoff/issue-141-cli-parsers-72fb776`. Its code is identical to the
candidate `issue-141-cli-parsers` at `72fb77662f9710c3abc32230d96ad37830129da2`. The one
commit on top of that adds only `handoff/issue-141/`. The exact head SHA of this branch is in
the final report and in the archive's INVENTORY.txt, because a commit cannot contain its own SHA.

## 1. Identity

- Repository: NoeFlandre/geoparser (https://github.com/NoeFlandre/geoparser)
- Issue: #141 "[cli] Unify argparse conventions across scripts" (open, stays open)
- Draft PR: #182 (https://github.com/NoeFlandre/geoparser/pull/182), base `main`, head `issue-141-cli-parsers`. The body says "Refs #141", not "Closes".
- Base: `main` = `e19b95b6107b734813e0b24b7550808501fa8e90` (unchanged at handoff)

## 2. Goal and exact completed scope

Goal: a partial #141. Move the inline argparse setup of two scripts into `build_parser()`,
with no behaviour change, and add characterisation tests.

Completed. Exactly three files differ from `main`:

- `scripts/changelog.py`: `build_parser()` added; `main(argv)` calls it. Flags, defaults, help, output, exit codes unchanged.
- `scripts/check_architecture.py`: `build_parser()` added; `main(argv)` calls it. Same guarantees.
- `tests/unit/test_cli/test_script_conventions.py` (new): characterisation tests (golden help text, direct and `python -m` execution, error exit codes and stderr, filesystem neutrality of parser construction, and an AST check that only `build_parser()` constructs the parser).

Deliberately not done:

- No other script changed. `scripts/_cli.py` (the issue's suggested shared helper) was not introduced.
- Not touched: `pilot.py`, `evaluation.py`, benchmark/PAN-X code, Grid scripts, dependencies, workflows, mutation configuration, `tests/conftest.py`, fixtures, `pyproject.toml`, `uv.lock`, `.github/`, `ty-lint.toml`, existing tests.

## 3. Branches and full commit SHAs

| Ref | Full SHA | Where | Status |
|---|---|---|---|
| `main` | `e19b95b6107b734813e0b24b7550808501fa8e90` | GitHub | base, unchanged |
| `issue-141-cli-parsers` | `72fb77662f9710c3abc32230d96ad37830129da2` | https://github.com/NoeFlandre/geoparser/tree/issue-141-cli-parsers | candidate; PR #182 head |
| `handoff/issue-141-cli-parsers-72fb776` | see INVENTORY.txt / final report | https://github.com/NoeFlandre/geoparser/tree/handoff/issue-141-cli-parsers-72fb776 | = 72fb776 + this docs commit |

Superseded candidates. These are local commits that were never pushed. Their diffs are in
`handoff/issue-141/superseded/`. Their commit objects are in the recovery bundle as refs
`refs/handoff/superseded/*`, not as GitHub branches, so they cannot be mistaken for merge candidates.

| Commit | Full SHA | Why superseded |
|---|---|---|
| 7aa4a9d | `7aa4a9d006066cab28d3e93b6ed196c4bcffdd24` | first squash; before test fixes |
| d109524 | `d1095242826fa76054ebf3fc58e8528b182f7e09` | 3.14 and colour fixes |
| 36ac09b | `36ac09b9bdd39c9b4e2b877d60e470efd69f0421` | module-mode check made exact; a delta review then found the startswith check had been dropped |

Intermediate commits withheld. These five local commits have messages that name a model in a
`Co-Authored-By` trailer. The system rules forbid that in commit messages, and you asked us to
follow that rule. They are NOT on GitHub and NOT in the bundle. Their diffs are kept as
message-free patches in `handoff/issue-141/superseded/`. Their commit objects will be lost when
this environment is discarded.

| Commit | Full SHA |
|---|---|
| 1950da9 | `1950da9cb4ea83e01c3e3fb6cd2607fc14ccc8eb` |
| e498ed6 | `e498ed6a8908c43c152ae2a7d7cc8782d4427708` |
| 0d24c5b | `0d24c5bfcbb55be03972657a85290a174c2a447c` |
| ca29d05 | `ca29d051247d2b88af22285d47757373473a15d5` |
| 3f9f6a0 | `3f9f6a0d32889aa63ce919a4b3581fe09c4d6635` |

Scripts are identical from e498ed6 to 72fb776. The test file was edited after e498ed6 and
its last edits are in 72fb776.

## 4. Tests and reviews

See `evidence/RESULTS.md` for the commands, the exact outputs, and the status of each result
(passed, failed, skipped, unrun, stale). See `evidence/REVIEWS.md` for each review verdict and
the SHA it applies to. In short:

- Final head 72fb776: targeted and CLI/quality tests (377 passed), lint, format, both type
  checks, and the Python 3.14 subset (33 passed, both launch styles) all passed.
- The full unit suite (2194 passed) ran on 36ac09b. 72fb776 changes one assertion after it, so
  that result is stale; the targeted rerun covers the change.
- The independent delta re-review of 72fb776 approved.
- Remote CI for 72fb776 was still queued at handoff. Results are not known here. See `evidence/CI-STATUS.md`.

## 5. Remaining work (not started)

- Remaining inline-argparse scripts in #141: `scripts/benchmark/compare.py`, `scripts/benchmark/publish.py`,
  `scripts/changed_mutation_patterns.py`, `scripts/crap.py`, `scripts/mutation_evidence.py`, `scripts/mutation_gate.py`,
  `scripts/mutation_replay.py`, `scripts/quality_gauntlet.py`. Excluded from this slice on purpose:
  `scripts/pilot.py`, `scripts/panx_benchmark`, and `scripts/benchmark/__main__.py` (which already has `build_parser`).
- Decision needed: whether to introduce a shared `scripts/_cli.py` for the remaining scripts.
  The issue suggests it; this slice did not.
- CI cells not reproduced locally: Windows, macOS, Python 3.10, the full suite on 3.11 and 3.12.
- CRAP with combined unit, integration and e2e coverage (CI's authoritative run) not reproduced.
- `scripts/quality_gauntlet.py` was not run end to end (mutation stages, Docker, strict MkDocs, model-loading integration tests).

## 6. Defects and observations

- Pre-existing, unchanged, pinned by a test: `check_architecture --package <missing path>`
  exits 0 and prints "(0 modules)". It fails open. Needs an owner decision. It is not an
  authorised fix. Test: `test_architecture_scans_a_missing_path_as_an_empty_package`.
- Minor test-hardening items from reviews, not applied:
  - The `refuse()` guard does not patch `Path.is_file`, `Path.is_dir`, `Path.stat`, `Path.glob`, `Path.write_text`, or `os.listdir`.
  - The body of `refuse()` is excluded from coverage by the project's `exclude_also = ["raise AssertionError"]`.
  - `allow_abbrev` is not tested.
  - One reviewer nit referred to a function that no longer exists in the file.

## 7. Dependencies and environment

- No dependency changed. `uv.lock` untouched.
- The repository pins uv `==0.11.16` (`pyproject.toml`). A host uv of another version refuses to run. Use `uvx --from uv==0.11.16 uv sync --locked --no-default-groups --group lint --group test`.
- Gazetteer tests need the DuckDB `spatial` extension. CI installs it in `test.yml` (around line 100). Without it, 9 gazetteer tests error at setup.
- The environment-only `.venv` is not part of this handoff.

## 8. Ownership overlaps with open PRs (checked at handoff; by file)

Open PRs at handoff: #121, #132, #157, #171, #172, #175, #176, #177, #179, #180, #181, #182 (ours).

- #175 (`issue-152-tomllib`): `pyproject.toml`, `scripts/changed_mutation_patterns.py`, `tests/unit/test_quality/test_crap.py`, `test_project_contract.py`. Overlaps remaining item `changed_mutation_patterns.py`.
- #176 (`issue-149-test-markers`): `tests/conftest.py` (shared fixture), `tests/unit/test_quality/test_test_hygiene.py`. Do not edit without coordination.
- #157 (`fix/issue-150-py-typed`): `pyproject.toml`, `test_project_contract.py`.
- #172 (`feat/otter-recognizers-161`, draft): `pyproject.toml`, `tests/unit/test_modules/...`.
- #180 (`claude/142-pilot-timing-classes`, draft): `scripts/pilot.py` (excluded), and a new file `tests/unit/test_cli/test_pilot_timing.py` in the same directory as our new test file. No file collision.
- #171 (`feat/benchmark-protocol-160`): adds `scripts/benchmark_protocol/*` and `tests/unit/test_benchmark_protocol/*`. Check before touching any benchmark CLI.
- #121 (`codex/spacy-panx-crosslingual-transfer`, draft): `scripts/panx_benchmark/*` (excluded area).
- #132, #177, #179, #181: no overlap with the slice files.
- PR #182 (ours): no conflict with the open PRs above at handoff time.

## 9. Pending decisions

1. Whether PR #182 stays draft, becomes ready, or is closed. Nothing has been decided. No human has approved it yet.
2. Fail-open policy for `check_architecture --package <missing path>`.
3. `scripts/_cli.py`: introduce it for the remaining scripts or not.
4. Whether to accept the withheld five intermediate commits as lost (their diffs are kept), given the attribution rule.
5. Whether the untested items in section 6 should be fixed in a follow-up.

## 10. Processes and jobs

- No process from the working environment is running at handoff.
- No subagent is running. The three review sessions created earlier are archived and completed.
- GitHub Actions runs for 72fb776 were queued at last check. Triggered by the push to PR #182. Not cancelled. Results after handoff are not covered by this document. See `evidence/CI-STATUS.md`.

## 11. Safest first steps for the next owner

1. Verify the remote. Fetch `handoff/issue-141-cli-parsers-72fb776` and check `git rev-parse HEAD` matches the SHA in INVENTORY.txt. Check `git diff --quiet 72fb776 HEAD -- scripts tests` exits 0.
2. Check CI for `72fb776` on PR #182 before acting on it. Do not merge.
3. Reproduce the locked environment (command in section 7), install the DuckDB extension, run `pytest tests/unit/test_cli tests/unit/test_quality -q`, `ruff check .`, `ruff format --check .`, `ty check --config-file ty-lint.toml geoparser scripts tests`.
4. Decide the fail-open policy and the `_cli.py` question before touching the eight remaining scripts.
5. Coordinate before editing `changed_mutation_patterns.py` (overlaps #175) and `tests/conftest.py` (#176).

## 12. Attribution

Nothing pushed to GitHub by this handoff contains a model name. The `Claude-Session` trailer is retained. The `Co-Authored-By` line the attribution reminder requested is omitted because the system rules forbid model names in commit messages and pushed artifacts. You said to stop reopening this question.
