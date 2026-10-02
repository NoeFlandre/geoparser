
This file is a living record of the campaign to leave no surviving mutant. Update the numbers and the checklist below each time that you work on it.

## How to run it

```bash
uv run mutmut run                 # full sweep, regenerates mutants/
uv run mutmut export-cicd-stats
uv run python scripts/mutation_gate.py --max-survivors 0 --max-no-tests 69
```

To inspect the survivors of one function, use `uv run mutmut results` and `uv run mutmut show <mutant>`. To check again after you write a test, use `uv run mutmut run <mutant-name>`. This takes seconds and not minutes.

Pragmas take effect only when the tool regenerates the mutant tree. Delete `mutants/` before a run that must use new pragmas.

## CI evidence and timeout follow-up

The pull-request mutation job uploads an artifact with the name `mutation-evidence-<run-id>-<commit>`. It does this also when the mutation gate fails. The artifact contains these items: the run log, the stats-export log, and the gate log; the raw output of `mutmut results --all true`; the exported stats, when available; and `report.json`. The report records the tested commit and the run metadata. It records the changed-module patterns and every parsed mutant ID and outcome. It gives a replay command for one mutant. It also gives the `mutmut show` output for each unresolved result. The report identifies the mutants that the job did not check after a baseline failure. It does not claim that the tests killed them. The job also compares their count with the exported total.

To replay one result, copy its command from `report.json`. This is an example:

```bash
uv run --no-sync mutmut run --max-children 1 geoparser.services.recognition.fit__mutmut_3
```

The campaigns `81fabc7` and `7cdd960` kept aggregate logs only. Their timeouts are inconclusive. The identities of the 36 timeouts and the 32 timeouts in those runs are not available. The later run below kept a report for each mutant. It gives the exact allowlist for bounded replays.

## Latest completed hosted sweep (1 October 2026)

The newest completed full mutation evidence is [run 36901916301](https://github.com/NoeFlandre/geoparser/actions/runs/36901916301), for PR head `a532d5eba284ca5eb333cae6515513499ab02812`. The tests killed all **3,724 mutants**. There were **zero survivors, zero timeouts, zero no-test mutants, zero skipped mutants, and zero suspicious outcomes**. [Artifact 11183505541](https://github.com/NoeFlandre/geoparser/actions/runs/36901916301/artifacts/11183505541) contains the evidence for each mutant. The sweep used one mutation child and one OpenMP, MKL, and OpenBLAS thread. This result replaces the earlier inconclusive sweeps for this exact source revision. Their historical results stay below.

## Local recovery of the 36 timeouts (1 October 2026)

The previous completed full CI sweep is [run 36874286263](https://github.com/NoeFlandre/geoparser/actions/runs/36874286263), for PR head `060b2851c6ac80f0c97fd6cd231e7d330b6506ba`. Its [artifact 11170019901](https://github.com/NoeFlandre/geoparser/actions/runs/36874286263/artifacts/11170019901) has SHA-256 `2f08712e1dc6737b3a43697053aa4c75471a64f6ad936203f936b2fa0a008d83`. It reports 3,700 killed and 36 inconclusive timeouts, not 3,736 kills.

We compared the generated diff of each timed-out mutation with that artifact. Then we replayed it in a new Python process. The unmutated baseline passed. The first replay killed 14 mutants and showed 22 survivors. Then stronger tests killed 16 of those survivors. Mismatched document lists, reference lists, and candidate lists must now raise an error. They must not truncate silently. Missing precomputed scores must produce the correct diagnostic. The former document-length test accepted a different `ValueError` by accident. It now isolates the intended shape error.

The other six mutations changed redundant defaults or repeated validation. They changed `zip(strict=False)` against its default or `None`. They changed the default axis of the cosine similarity. They changed a second document-alignment check, after `_pending_pairs` already validated the same lists. Small cleanups of the implementation removed that redundant work. These six are **not counted as killed**. We added no mutation exclusions. We did not lower any quality threshold.

On the updated local code, a targeted mutation run on the four changed functions killed all 80 selected mutants. There were zero survivors and zero timeouts. It used one mutation child and one OpenMP, MKL, and OpenBLAS thread. This validation has a limited scope. It is not a claim of a new full 3,724-mutant sweep. All 2,340 tests passed locally. These tests include the benchmark tests and the real GLiNER and Jina integration tests. The repository-wide CRAP check covered 3,637 functions. The maximum score was 5.67, which is below the exclusive threshold of 6. Full hosted CI must still validate the resulting commit before the merge.

## Recorded run history

The counts below are historical snapshots. Each snapshot was recorded in the named documentation revision. They are not measurements of the current working tree. The older count of `212` no-test mutants and the later count of `69` came from different campaigns. They do not contradict each other.

| Snapshot recorded in | Run | Mutants | Killed | Survived | No tests | Timeout | Segfault | Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2d993fd` (2026-09-11) | Clean sweep after context extraction | 2028 | 1816 | **0** | 212 | 0 | 0 | — |
| `8d07b9b` (2026-09-12) | Clean sweep after model pass | 1999 | 1786 | **0** | 212 | 1 | 0 | 31.2/s |
| `f92c594` (2026-09-28) | Fit coverage, before latest assertions | 3568 | 3424 | **62** | 69 | 13 | 0 | — |
| `81fabc7` (2026-10-01) | Full sweep after quality refactor | 3736 | 3696 | **4** | 0 | 36 | 0 | 99.9% of decided outcomes |
| `7cdd960` (2026-10-01) | Full sweep after fit assertions and PAN-X span scoring | 3736 | 3704 | **0** | 0 | 32 | 0 | 100% of decided outcomes |
| PR head `fb896a2`; run `36818760703` | Full sweep with per-mutant artifact | 3736 | 3705 | **0** | 0 | 31 | 0 | 100% of decided outcomes |
| PR head `6fc75e4`; run `36832597467` | Full sweep after fit assertions | 3736 | 3709 | **0** | 0 | 27 | 0 | 100% of decided outcomes |
| PR head `ff0f369`; run `36860416064` | Full sweep after PAN-X integration | 3736 | 3710 | **0** | 0 | 26 | 0 | 100% of decided outcomes |
| PR head `060b285`; run `36874286263` | Full sweep after Transformers 5 update | 3736 | 3700 | **0** | 0 | 36 | 0 | 100% of decided outcomes |

The historical `f92c594` campaign has 3,568 outcomes in total. The tests killed 3,424 mutants. 62 survived. 69 had no covering unit test. 13 timed out. It did **not** pass the gate of zero survivors. The quality gauntlet and CI still enforce `--max-survivors 0 --max-no-tests 69`. We did not increase either limit. `69` is the historical ceiling for no-test mutants. It does not mean that the latest run had that number of no-test mutants.

The preceding full mutation run was [quality workflow run 36860416064](https://github.com/NoeFlandre/geoparser/actions/runs/36860416064), for PR #111 head `ff0f369f37000d56620251c5376d84eca5ac2a28` (merge checkout `9d260404ac78663558f75b323e449a53ed509944`). Its artifact is [mutation-evidence-36860416064](https://github.com/NoeFlandre/geoparser/actions/runs/36860416064/artifacts/11163287916), SHA-256 `3419f776a4f85540ed0b35656992248e001fe4adca20d0ea211957d1f172aab9`.

It recorded 3,710 killed mutants, zero survivors, zero mutants with no covering tests, and 26 timeouts (3,736 in total). The rate of 100% is for the 3,710 decided outcomes only. Timeouts are inconclusive. We do not count them as kills. The tests killed the four earlier `RecognitionService.fit` survivors (`mutmut_3`, `mutmut_6`, `mutmut_7`, and `mutmut_8`).

That run recorded 26 timeouts in the code for resolver similarity and ranking:

| Function | Timeouts |
| --- | ---: |
| `SimilarityMixin._calculate_similarity_batches` | 2 |
| `SimilarityMixin._non_empty` | 3 |
| `SimilarityMixin._flat_similarities` | 4 |
| `SentenceTransformerResolver._evaluate_candidates` | 3 |
| `SentenceTransformerResolver._pending_pairs` | 7 |
| `SentenceTransformerResolver._document_similarities` | 3 |
| `SentenceTransformerResolver._evaluate_document` | 4 |

The pull-request event skipped the exact-mutant-replay job. Therefore these 26 outcomes were unresolved in that snapshot. All 26 exact IDs are in the manual replay allowlist from the preceding run with 27 timeouts. Nobody replayed them on this head. The replay workflow accepts at most eight allowlisted IDs for each dispatch. It checks the expected full commit SHA. A replay must report a terminal killed outcome before you can count a timeout as killed.

The completed sweep on `81fabc76532002dede430534b60a7110c12db279` generated 3,736 mutants. The tests killed 3,696. Four survived. 36 timed out. No mutant was left without tests. The kill rate of the resolved outcomes was 99.9% (3,696 of 3,700 decided outcomes). The 36 timeouts are inconclusive. We do **not** count them as killed. The gate failed on four `RecognitionService.fit` survivors. The mutations changed the recognizer-kind label that goes to `require_fit`. A focused test now checks that the error for a missing fit names a recognizer and its name. The later run on the exact head below confirms that these four mutants do not survive now. All 36 timeouts were in `SimilarityMixin._calculate_similarity_batches`, `_non_empty`, `_flat_similarities`, or in `SentenceTransformerResolver._search_tier`, `_gather_candidates`, `_embed_candidates`, `_evaluate_candidates`, `_pending_pairs`, `_document_similarities`, and `_evaluate_document`. They need their own runtime and evidence triage. The percentage of resolved outcomes does not settle them.

The full sweep on the exact head `7cdd960dc559089aacb1614019a0f6dc0c3a409e` generated 3,736 mutants. The tests killed 3,704. Zero survived. Zero had no covering tests. 32 timed out. The 32 timeouts are inconclusive. We do **not** count them as killed. This confirms that the four earlier `RecognitionService.fit` label mutants do not survive now. That run kept aggregate logs only. Therefore the identities of its timeouts are not known. Its count of zero no-test mutants did not change the established allowance `--max-no-tests 69`.

The newer full sweep in workflow run `36818760703` used PR head `fb896a2b181fd03416653610b2f54f63c3e759e3`. (The Actions merge checkout was `2ce099a0bc175e3727fdd990089523574461aca6`.) It generated 3,736 mutants. The tests killed 3,705. Zero survived. Zero had no tests. 31 timed out. The resolved kill rate was 100% of the 3,705 decided outcomes. All 31 timeouts are still inconclusive. They are not kills. The SHA-256 of the artifact report is `9f163a8ce146003e9094323e2d9a2eb77e25b21c3c63b56560e7e86f0ecb2c1e`. A comparison with the preceding `c820ade` inventory found four additional timeouts. Three were `_embed_candidates` role-value mutations. One was an `_extract_context` end-boundary mutation. The tests killed all 17 `RecognitionService.fit` mutants in this sweep. This includes the four former survivors.

The current `scripts/mutation_replay_allowlist.json` records the later inventory of 27 timeouts from run `36842110918`. It is bound to the source checkout `da46b4841404493012e07f2e8df6165ffe8f02ec`. The manual `targeted-mutant-replay` mode of the Quality workflow accepts only those literal IDs. It checks out the retained source head `f0f07a630fda3be4b758d27690bd12a2c280897c`. It requires that the actual checkout matches it. That head and the historical merge checkout have the same tree `fb77bc0a9cb2516c16343c6c39d971560f6a9878`. (We verified this through the GitHub Git commit metadata.) The current controller is kept separate. The source and the locked dependencies use the evidence revision. The workflow runs at most eight IDs in series for each dispatch. You cannot reuse ordinal mutant IDs on later code revisions. Generate a new inventory for new code. The workflow uses the configured timeout policy of mutmut with no timeout-factor override. It uploads the logs and the report for each mutant. This is a diagnostic workflow. It is separate from the normal PR gates. A timeout in a replay stays inconclusive. We never count it as killed.

## Scope, and why

Mutation testing uses `tests/unit`. Its source scope is narrower than the coverage scope. The full `geoparser/` package, including the annotator, stays in the coverage measurement. The mutmut exclusions are explicit in `pyproject.toml`:

- **Annotator web/API wiring and `geoparser/cli/*`**. The excluded annotator files are the app setup, the routes, the server wiring, the constants, the dependencies, the exceptions, the metadata, and the API schemas. The database repositories, the database models, and the database helpers under `geoparser/annotator/db/` stay in the mutation scope. They have unit tests.
- **`geoparser/gazetteer/build/*`**. The build pipeline assembles SQL and drives duckdb. The unit suite mocks duckdb. To judge its mutants correctly, you need the integration suite. That suite builds a real gazetteer for each mutant. This takes about 23 seconds for each mutant and about thirteen hours for the package. The exclusion also removed every segfault. All segfaults came from mutation of native-extension code that mutmut runs in-process. The integration tests, the e2e tests, and the coverage gate cover this code. It is outside the mutation result counts above.

The unit tests in `tests/unit` judge the mutants. We tried to add `tests/integration`. It removes every "no tests" mutant. It reduces the survival from 29% to about 11%. But the cost is the cost above.

## Tests or pragma?

A surviving mutant is one of two things. Keep the distinction exact:

1. **A real gap.** The suite runs the line, but it does not check what the line did. Write the assertion. This is the valuable half.
2. **An equivalent mutant.** Nothing can distinguish it from the original. No test can kill it. Mark it `# pragma: no mutate` *with a comment that gives the reason*. Prefer evidence to intuition. We confirmed that the two soundex mutants were identical for 44,000 random inputs before we marked them.

The wording of a message is the second kind. If you pin the prose of an error word for word, the test breaks at each copy-edit. It verifies nothing. Assert the part that matters. For example, assert that the message names the offending file. Use an explicit `start`/`end` pragma region around the `raise`.

We considered a blanket regex for "lines that look like message text" and we rejected it. A regex must not decide what counts as behavior.

## Previously verified mutation fixes

The runs of the entries below reported zero survivors. This is historical evidence. It does not describe the current campaign.

- [x] `SentenceTransformerResolver._expand_window`: 32 (31 tests, 1 pragma)
- [x] `_check_database_compatibility`: 25 (7 tests, 18 pragma: message prose, SQL keywords, and a SQLite identifier, all case-insensitive)
- [x] `SpacyRecognizer`: 20 (19 tests, 1 pragma: download progress message)
- [x] `ResolutionService.fit` / `_annotated_pairs`: 15 (tests)
- [x] `_search_tier`, `_all_resolved`, `_unresolved_candidate_lists`, `_merge_candidates`, `_evaluate_document`, `_gather_candidates`, `predict` tiers, `_search_once`: tests
- [x] `_encode`: 9 (1 test, 2 pragma: the batch size and the progress bar affect throughput and display, not behavior)
- [x] `soundex`: 9 (7 tests against published reference codes, 2 pragma, both verified as equivalent by experiment)
- [x] `Project._normalize_document_ids`, `create_documents`, `_ensure_project_record`: tests, and 1 pragma for guidance wording
- [x] `gazetteers_dir`, `artifact_path`, `list_artifacts`, `register_functions`: tests, and 1 pragma for the appauthor argument that only Windows uses
- [x] `Project.load_annotations`: 14 (tests: the data flow of the import)
- [x] `Project.run_recognizer` / `run_resolver` / `get_documents`: 18 (tests: default and explicit tags, and the documents that each service receives)
- [x] `_best_referent`, `_token_limit`, `_extract_contexts`: 12 (tests, and 1 pragma for the wording of the no-maximum-length message)
- [x] `RecognitionService._record_reference_predictions` and `ResolutionService._record_referent_predictions`: tests for the lenient handling of short prediction lists and of None predictions
- [x] `SentenceTransformerResolver._prepare_training_data`: 15 (tests for the contrastive pairs: labels, contexts, and the gazetteer lookups)
- [x] `ReferenceRepository.update` / `get_by_document_and_span`: 8 (tests for the span fallbacks. We deleted a redundant `hasattr` guard and did not test it, because the foreign key makes a dangling document impossible)
- [x] `Project._normalize_document_ids`: 5 (1 pragma: guidance wording)

## Last recorded no-tests inventory

The 69 no-test mutants in the `f92c594` snapshot were in six functions. The counts below add up to 69. They are a historical inventory. The latest full CI run classified zero mutants as mutants with no covering tests. But it does not give a current allocation for each function for this old inventory. The campaign did not record the tested source revision. Do not present these values as results for the current tree.

| Function in the historical inventory | No-test mutants | Focused unit-test evidence now in the tree |
| --- | ---: | --- |
| `Context.update_recognizer_context` | 18 | `tests/unit/test_db/test_models/test_context.py::TestProjectContext.test_updates_recognizer_context_for_a_tag` |
| `Context.update_resolver_context` | 18 | `tests/unit/test_db/test_models/test_context.py::TestProjectContext.test_updates_resolver_context_for_a_tag` |
| `RecognitionService.fit` | 17 | `tests/unit/test_services/test_recognition.py::TestRecognitionServiceFit.test_fits_only_annotated_documents_and_forwards_spans_and_options` |
| `Project.train_recognizer` | 7 | `tests/unit/test_project/test_project.py::TestProjectTrainRecognizer.test_trains_recognizer_with_tag_and_options` |
| `Project.train_resolver` | 7 | `tests/unit/test_project/test_project.py::TestProjectTrainResolver.test_trains_resolver_with_tag_and_options` |
| `GazetteerArtifact.count_names` | 2 | `tests/unit/test_gazetteer/test_artifact.py::TestGazetteerArtifactCounts.test_counts_names` |
| **Historical total** | **69** | |

We added no new mutation exclusions or pragmas for these entries. A no-test mutant means that the selected unit suite did not reach that code in that run. It is not evidence that the mutant is equivalent. It is not evidence that the integration coverage killed it. Review equivalent mutants only after a run produced a concrete survivor and a test or other reproducible evidence shows the equivalence.

## A gate that could pass without checking anything

mutmut writes a stats file and exits with zero when its **own** baseline test run fails. Then every mutant stays "not checked". The exported stats show `killed: 0, survived: 0`. A gate that looks only at the survivor count reports this as a clean result. This occurred twice in one session. The first time, an architecture test that reads the real package source failed inside the rewritten `mutants/` copy. The second time, we started a filtered run after we deleted `mutants/`. This discarded the mapping from tests to functions.

Therefore `scripts/mutation_gate.py` fails when the outcome categories do not add up to the number of mutants generated. A run that verified nothing now reports exactly that. It does not report a score of 100%.

There are two practical consequences:

- A test that inspects the real source tree must skip itself inside the mutant copy. `tests/unit/test_meta/test_pragma_placement.py` and the pure-module check in `tests/unit/test_quality/test_architecture.py` both do this with an `IN_MUTANT_TREE` guard.
- A filtered run (`mutmut run <pattern>`) needs the mapping that a full run builds. Do not delete `mutants/` before a filtered run.

The `fb896a2` sweep recorded 31 unresolved timeouts. Run `36832597467` is an earlier full mutation snapshot. It recorded 27 timeouts before the later campaigns above. Neither run counts a timeout as killed. The historical `--max-no-tests 69` allowance is unchanged, although the latest run reported zero no-test mutants.
