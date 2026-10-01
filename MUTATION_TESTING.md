# Mutation testing status

A living record of the campaign to leave no surviving mutant. Update the
numbers and the checklist below whenever you work on it.

## How to run it

```bash
uv run mutmut run                 # full sweep, regenerates mutants/
uv run mutmut export-cicd-stats
uv run python scripts/mutation_gate.py --max-survivors 0 --max-no-tests 69
```

Inspect one function's survivors with `uv run mutmut results` and
`uv run mutmut show <mutant>`. A targeted re-check after writing a test is
`uv run mutmut run <mutant-name>`, which is seconds rather than minutes.

Pragmas only take effect when the mutant tree is regenerated, so delete
`mutants/` before a run that is meant to pick them up.

## CI evidence and timeout follow-up

The pull-request mutation job uploads an artifact named
`mutation-evidence-<run-id>-<commit>`, even when the mutation gate fails. It
contains the run, stats-export, and gate logs; raw `mutmut results --all true`
output; the exported stats when available; and `report.json`. The report records
the tested commit and run metadata, changed-module patterns, every parsed
mutant ID and outcome, a one-mutant replay command, and `mutmut show` output for
each unresolved result. Mutants left unchecked after a baseline failure are
identified without pretending that they were killed; their count is also
compared with the exported total.

To replay an individual result, copy its command from `report.json`, for
example:

```bash
uv run --no-sync mutmut run --max-children 1 geoparser.services.recognition.fit__mutmut_3
```

The `81fabc7` and `7cdd960` campaigns predate this artifact and retained only
aggregate logs. Their timeout counts remain inconclusive; the identities and
causes of the 32 timeouts in the newest recorded run are not known.

## Recorded run history

The counts below are historical snapshots recorded in the named documentation
revision; they are not measurements of the current working tree. In particular,
the older `212` no-test count and the later `69` count came from different
campaigns and are not contradictory.

| Snapshot recorded in | Run | Mutants | Killed | Survived | No tests | Timeout | Segfault | Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2d993fd` (2026-09-11) | Clean sweep after context extraction | 2028 | 1816 | **0** | 212 | 0 | 0 | — |
| `8d07b9b` (2026-09-12) | Clean sweep after model pass | 1999 | 1786 | **0** | 212 | 1 | 0 | 31.2/s |
| `f92c594` (2026-09-28) | Fit coverage, before latest assertions | 3568 | 3424 | **62** | 69 | 13 | 0 | — |
| `81fabc7` (2026-10-01) | Full sweep after quality refactor | 3736 | 3696 | **4** | 0 | 36 | 0 | 99.9% of decided outcomes |
| `7cdd960` (2026-10-01) | Full sweep after fit assertions and PAN-X span scoring | 3736 | 3704 | **0** | 0 | 32 | 0 | 100% of decided outcomes |

The September 28, 2026 campaign sums to 3,568 outcomes: 3,424 killed, 62
survived, 69 had no covering unit test, and 13 timed out. It did **not** pass
the zero-survivor gate. The quality gauntlet and CI still enforce
`--max-survivors 0 --max-no-tests 69`; neither limit has been raised. `69` is
the last measured no-tests count and the unchanged ceiling, not a claim that a
new run on the current tree has passed.

The completed sweep on `81fabc76532002dede430534b60a7110c12db279` generated
3,736 mutants: 3,696 were killed, four survived, 36 timed out, and none were
left without tests. The resolved-outcome kill rate was 99.9% (3,696 of 3,700
decided outcomes); the 36 timeouts are inconclusive and are **not** counted as
killed. The gate failed on four `RecognitionService.fit` survivors whose
mutations changed the recognizer-kind label passed to `require_fit`. A focused
test now checks the missing-fit error names a recognizer and its name. The
later exact-head run below confirms those four mutants no longer survive. All
36 timeouts were in
`SimilarityMixin._calculate_similarity_batches`, `_non_empty`,
`_flat_similarities`, or `SentenceTransformerResolver._search_tier`,
`_gather_candidates`, `_embed_candidates`, `_evaluate_candidates`,
`_pending_pairs`, `_document_similarities`, and `_evaluate_document`. They need
their own runtime/evidence triage; the resolved-outcome percentage does not
settle them.

The full sweep on exact head
`7cdd960dc559089aacb1614019a0f6dc0c3a409e` generated 3,736 mutants: 3,704
killed, zero survived, zero had no covering tests, and 32 timed out. The
resolved-outcome kill rate was 100% (3,704 of 3,704 decided outcomes); the 32
timeouts remain inconclusive and are **not** counted as killed. This confirms
the four earlier `RecognitionService.fit` label mutants no longer survive.
The retained Actions log reports aggregate counts only, and the run has no
per-mutant artifact, so the identities and causes of these 32 timeouts are not
established. The earlier function list is triage context for the 36-timeout
run, not a disposition of the newer 32. The zero-no-tests count applies to
this full run and does not change the established `--max-no-tests 69`
allowance.

## Scope, and why

Mutation testing uses `tests/unit`; its source scope is narrower than coverage.
The full `geoparser/` package, including the annotator, remains in coverage
measurement. The mutmut exclusions are explicit in `pyproject.toml`:

- **Annotator web/API wiring and `geoparser/cli/*`** — the excluded annotator
  files are app setup, routes, server wiring, constants, dependencies,
  exceptions, metadata, and API schemas. The database repositories, database
  models, and database helpers under `geoparser/annotator/db/` remain in
  mutation scope and have unit tests.
- **`geoparser/gazetteer/build/*`** — the build pipeline assembles SQL and
  drives duckdb, which the unit suite mocks away. Judging its mutants honestly
  needs the integration suite, and that rebuilds a real gazetteer per mutant:
  about 23 seconds each, some thirteen hours for the package. Excluding it also
  removed every segfault, since those all came from mutating native-extension
  code that mutmut runs in-process. It is covered by integration and e2e tests
  and the coverage gate; it is outside the mutation result counts above.

Mutants are judged by `tests/unit` only. Adding `tests/integration` was tried:
it removes every "no tests" mutant and cuts survival from 29% to about 11%, but
at the cost above.

## Tests or pragma?

A surviving mutant is one of two things, and the distinction is worth keeping
honest:

1. **A real gap.** The suite runs the line but never checks what it did. Write
   the assertion. This is the valuable half.
2. **An equivalent mutant.** Nothing can distinguish it from the original, so
   no test can ever kill it. Mark it `# pragma: no mutate` *with a comment
   saying why*, and prefer evidence over intuition — the two soundex ones were
   confirmed identical over 44,000 random inputs before being marked.

Message wording counts as the second kind. Pinning the prose of an error
word-for-word breaks on every copy-edit while verifying nothing; assert the
part that matters (that the message names the offending file, say) and use an
explicit `start`/`end` pragma region around the `raise`.

A blanket regex over "lines that look like message text" was considered and
rejected: a regex should not be the thing deciding what counts as behaviour.

## Previously verified mutation fixes

The entries below were reported at zero survivors in their respective runs;
that is historical evidence, not a claim about the current campaign.

- [x] `SentenceTransformerResolver._expand_window` — 32 (31 tests, 1 pragma)
- [x] `_check_database_compatibility` — 25 (7 tests, 18 pragma: message prose,
      plus SQL keywords and a SQLite identifier, all case-insensitive)
- [x] `SpacyRecognizer` — 20 (19 tests, 1 pragma: download progress message)
- [x] `ResolutionService.fit` / `_annotated_pairs` — 15 (tests)
- [x] `_search_tier`, `_all_resolved`, `_unresolved_candidate_lists`,
      `_merge_candidates`, `_evaluate_document`, `_gather_candidates`,
      `predict` tiers, `_search_once` — tests
- [x] `_encode` — 9 (1 test, 2 pragma: batch size and progress bar are
      throughput and display, not behaviour)
- [x] `soundex` — 9 (7 tests against published reference codes, 2 pragma,
      both verified equivalent empirically)
- [x] `Project._normalize_document_ids`, `create_documents`,
      `_ensure_project_record` — tests, plus 1 pragma for guidance wording
- [x] `gazetteers_dir`, `artifact_path`, `list_artifacts`,
      `register_functions` — tests, plus 1 pragma for the Windows-only
      appauthor argument
- [x] `Project.load_annotations` — 14 (tests: the import's data flow)
- [x] `Project.run_recognizer` / `run_resolver` / `get_documents` — 18 (tests:
      default and explicit tags, and the documents each service receives)
- [x] `_best_referent`, `_token_limit`, `_extract_contexts` — 12 (tests, plus
      1 pragma for the no-maximum-length message wording)
- [x] `RecognitionService._record_reference_predictions` and
      `ResolutionService._record_referent_predictions` — tests for the lenient
      handling of short prediction lists and of None predictions
- [x] `SentenceTransformerResolver._prepare_training_data` — 15 (tests for the
      contrastive pairs: labels, contexts and the gazetteer lookups)
- [x] `ReferenceRepository.update` / `get_by_document_and_span` — 8 (tests for
      the span fallbacks; a redundant `hasattr` guard was deleted rather than
      tested, since the foreign key makes a dangling document impossible)
- [x] `Project._normalize_document_ids` — 5 (1 pragma: guidance wording)

## Last recorded no-tests inventory

The 69 no-test mutants in the `f92c594` snapshot were grouped under six
functions. The counts below sum to 69. Since that run, focused unit tests have
been added or confirmed for each function. That improves the test evidence but
does not establish how many mutants the current tree kills: a fresh mutmut run
is required to refresh the inventory and survivor count. The tested source
revision for the historical campaign was not recorded, so these values must
not be presented as current-tree results.

| Function in the historical inventory | No-test mutants | Focused unit-test evidence now in the tree |
| --- | ---: | --- |
| `Context.update_recognizer_context` | 18 | `tests/unit/test_db/test_models/test_context.py::TestProjectContext.test_updates_recognizer_context_for_a_tag` |
| `Context.update_resolver_context` | 18 | `tests/unit/test_db/test_models/test_context.py::TestProjectContext.test_updates_resolver_context_for_a_tag` |
| `RecognitionService.fit` | 17 | `tests/unit/test_services/test_recognition.py::TestRecognitionServiceFit.test_fits_only_annotated_documents_and_forwards_spans_and_options` |
| `Project.train_recognizer` | 7 | `tests/unit/test_project/test_project.py::TestProjectTrainRecognizer.test_trains_recognizer_with_tag_and_options` |
| `Project.train_resolver` | 7 | `tests/unit/test_project/test_project.py::TestProjectTrainResolver.test_trains_resolver_with_tag_and_options` |
| `GazetteerArtifact.count_names` | 2 | `tests/unit/test_gazetteer/test_artifact.py::TestGazetteerArtifactCounts.test_counts_names` |
| **Historical total** | **69** | |

No new mutation exclusions or pragmas were added for these entries. A no-test
mutant means the selected unit suite did not reach that code in that run; it is
not evidence that the mutant is equivalent or that integration coverage has
killed it. Review equivalent mutants only after a run has produced a concrete
survivor and a test or other reproducible evidence demonstrates equivalence.

## A gate that could pass without checking anything

mutmut writes a stats file and exits zero even when its **own** baseline test
run fails. Every mutant is then left "not checked", and the exported stats read
`killed: 0, survived: 0` — which a gate that looks only at the survivor count
happily reports as a clean sheet. This was hit twice in one session: once when
an architecture test that reads the real package source failed inside the
rewritten `mutants/` copy, and once when a filtered run was started after
`mutants/` had been deleted, discarding the test-to-function mapping.

`scripts/mutation_gate.py` therefore fails when the outcome categories do not
add up to the number of mutants generated. A run that verified nothing now
reports exactly that instead of a 100% score.

Two practical consequences:

- A test that inspects the real source tree must skip itself inside the mutant
  copy. `tests/unit/test_meta/test_pragma_placement.py` and the pure-module
  check in `tests/unit/test_quality/test_architecture.py` both do this via an
  `IN_MUTANT_TREE` guard.
- A filtered run (`mutmut run <pattern>`) needs the mapping a full run builds.
  Do not delete `mutants/` before one.

The `7cdd960` full sweep above is the newest recorded mutation evidence. It
passes the zero-survivor gate, while its 32 timeouts remain unresolved
outcomes. The earlier `69` no-tests count and its unchanged ceiling remain a
separate, historical fit-coverage baseline.
