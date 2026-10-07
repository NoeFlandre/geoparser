# Public benchmark protocol

Version 1.0 defines the comparison contract for issues #160 through #170. It does not run models, train models, download weights, or schedule compute. Use public benchmarks for this phase. Do not use the owner's OSM text data.

## Validate a plan

```bash
uv run python -m scripts.benchmark_protocol --dry-run docs/examples/benchmark-protocol-v1.json
uv run python -m scripts.benchmark_protocol --schema
```

The example is synthetic. Its hashes and hardware describe a test fixture, not a measured run. Replace them with verified values before an experiment. The dry-run command reads one local JSON file and prints the inventory, missing target languages, status counts, and provenance digests. It does not open the artifact locations, calculate bootstrap intervals, or write output files. Exit status 2 means validation failed.

The generated JSON Schema lists every field and allowed value. Import `Experiment` from `scripts.benchmark_protocol.schema`. Import `inventory_summary` and `aggregate` from `scripts.benchmark_protocol.summary`. Use `Experiment.provenance_digest(configuration)` to bind a result to its inputs. Call `aggregate(experiment)` only when you want to calculate confidence intervals from retained per-example counts.

## Freeze the experiment

Use the existing language inventory in `scripts/panx_benchmark/target_languages.json`. It contains 85 codes from the upstream sentence-splitting guide at revision `c6b503908b4687517f546cecce40c621d4ee56ac`. Keep script and variant distinctions. Dataset adapters must make their source-configuration mappings explicit. Do not derive target languages from advertised model coverage.

Each manifest contains one dataset snapshot, annotation quality, task, split, selection stage, source-code commit, dependency-lock digest, and hardware description. Each configuration records a pipeline, source configuration, language, source example digest, sample counts, model pins, label mapping, actual thresholds, batch size, and seed. Resolution also requires a pinned gazetteer. Every artifact has an identifier, revision, and SHA-256 digest. A digest identifies the exact bytes; the schema cannot verify remote bytes or prove that a declared pin is truthful.

Mark annotation quality as `human_gold` or `silver`. Mark model language support as `documented`, `transfer`, `unspecified`, or `out_of_language_control`. Mark training overlap as `known`, `unknown`, or `verified_absent`, and cite the audit or pinned model-card evidence in `provenance_note`. Absence of a training-data disclosure means unknown overlap. A multilingual marketing claim is not a verified language list.

Record adapter-specific tokenizer, prompt, and decoding settings in `parameters`. They enter the provenance digest. Use `custom_code` to pair each custom-code artifact with its retained review artifact. Review and pin that exact code before executing it. A schema-valid plan does not authorize execution, downloads, training, publication, or a paid service.

## Select on development data

Use `screening` with the `development` split. Select thresholds and finalists there. If a threshold is prescribed externally, use `fixed_before_evaluation` and document its origin. Never use test outcomes to select thresholds, labels, examples, prompts, or checkpoints.

Use `final` only with the `test` split and a `selection_sha256`. This digest identifies the saved development decision with its finalist list and frozen settings. Archive the decision before opening the test results. The validator checks the structure; it cannot prove the chronology or detect an experimenter who has already inspected test data. Treat a later change as a new protocol and report its reason.

## Keep the tasks separate

- `recognition` scores exact, deduplicated, half-open Python character spans. The adapter preserves original text, offsets, label mapping, and annotation ambiguity. Invalid outputs remain false positives and appear in raw predictions.
- `gold_span_resolution` supplies gold spans to the resolver. Every gold span is resolved, abstained, or invalid. Exact ID accuracy uses gold spans with eligible canonical IDs. Distance accuracy uses spans with gold coordinates, at inclusive 1 km, 10 km, and 50 km thresholds. Keep those eligible counts visible. Coverage, abstention, and invalid rates use all gold spans. Candidate recall uses gold spans with a known evaluable candidate target, before ranking. Missing targets count as misses, not exclusions merely because retrieval failed.
- `end_to_end` counts a true positive only when both the character span and canonical gazetteer ID match. Wrong IDs contribute both a false positive and a false negative. Use a source slice with eligible IDs for all gold spans. A distance-tolerant end-to-end experiment needs a new explicit contract; do not label a span-only score as end-to-end.

The existing `scripts.benchmark` runner's `resolution` phase uses gold annotations. Map it to `gold_span_resolution`. It is not end-to-end evidence. Its CLI default `min_similarity=0.0` differs from resolver constructors' `0.6` default. Record the actual setting, including 0.0. Do not reinterpret historical reports as if they used 0.6.

## Account for every configuration

Every pipeline and seed must declare every source slice in the experiment. Keep unsupported languages as explicit `unsupported` outcomes with a reason. Each declared configuration has exactly one result: `planned`, `complete`, `failed`, or `unsupported`. A failure also needs a reason. Missing results and duplicate cells fail validation. Planned and unsupported outcomes never become zero scores.

A complete result contains a matching provenance digest, measurements, aggregate counts, a raw-predictions artifact reference, and one count record per source document or sentence. Keep failed examples in that unit inventory and the gold denominators. Count failed predictions as missing predictions. Failed recognition and end-to-end units cannot retain valid predicted spans. They can retain invalid-output false positives. Failed resolution units cannot retain resolved outputs; candidate retrieval can still have succeeded before ranking failed. Retain invalid raw outputs and their false-positive penalty. The sum of unit counts must equal the reported aggregate counts. Example IDs, gold counts, and eligible resolution denominators must agree across compared pipelines.

Each complete result separately records checkpoint fetch, model load, warmup, and steady inference in seconds, plus the warmup example count. Record peak process RSS and peak device allocation in bytes. Use zero device bytes for CPU runs. The protocol requires the same hardware description; it does not claim that different batch sizes or seeds are interchangeable. Record any acquisition time outside steady inference. Do not combine warm and cold measurements.

## Aggregate and quantify uncertainty

`aggregate` keeps each pipeline and seed separate. It pools source configurations within a language, then computes precision, recall, and F1 from pooled TP, FP, and FN counts. Micro scores pool counts over all included languages. Macro scores give each included language equal weight. Resolution uses the same count-pooling and equal-language rules for its own metrics. A zero denominator produces zero and the corresponding count remains visible.

Each summary lists its languages and whether its declared inventory is complete. Its configuration inventory retains source slices, sample digests, statuses, reasons, counts, and whether each cell contributes to metrics. Compare pipeline rankings only on the same source/language membership. Do not compare a multilingual macro average with the English-only reference's average. Retain seed-level results instead of pooling repeated evaluations as independent samples.

The uncertainty method is a paired document bootstrap with replacement, stratified by language and source configuration. The unit is a complete document or the dataset's complete sentence, not an individual span. Sort units by stable example ID. A source-specific generator derived from the recorded bootstrap seed supplies the same draws to each pipeline on that source. Recompute micro and macro metrics on each draw. Intervals are the empirical inverse-CDF 2.5th and 97.5th percentiles. Store the resample count and seed. Use at least 1,000 resamples for published analysis.

The intervals condition on the selected dataset and pipeline seed. They do not measure annotation error, uncertain training overlap, or variation between training seeds. Small samples can produce degenerate intervals. The dry-run command deliberately does not calculate them.

## Reuse completed WikiANN evidence

Issue #98's 1 October partial-run status is historical. The [immutable published report](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results/blob/f44041f054baa221fdc02a9f8ee6c137517fb5b1/runs/panx/2026-10-02-cd0c750-transformers-5-18-cpu/report.md) records the completed 2 October run:

- Source commit `cd0c750309fa5aacd8a4f9ecb75b71f6f9230380` and dataset revision `f0a3be6dc5564c0cc4150bb660144800a1f539d4`
- 82 eligible languages and 423,100 test rows for GLiNER and XLM-R
- 10,000 English rows for spaCy, with other languages explicitly unsupported
- Canonical target languages without this test split: `ha`, `xh`, `zu`
- GLiNER threshold 0.5, seed 0, input group sizes 1 for GLiNER and 8 for spaCy and XLM-R

Do not rerun this evidence merely to fill the new schema. Keep its original protocol, raw predictions, timing definitions, and immutable report. Missing legacy fields stay unknown; do not manufacture model-memory measurements or bootstrap intervals. Do not merge the September feasibility sample or October partial checkpoints with the completed run. The September sample lacks a recoverable source revision and remains excluded from pipeline selection. Add a separately labeled reconciliation or derived analysis if a later protocol needs one.
