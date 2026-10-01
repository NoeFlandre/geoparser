# Benchmarking

The benchmark harness measures recognition and resolution over fixed corpora. It saves a checkpoint after each document chunk, so a stopped run can resume without repeating completed work. Reports record the input digest, commit, models, metrics, and elapsed time.

## Run locally

Install the benchmark extras and the `geonames` gazetteer, then inspect the CLI options:

```bash
python -m scripts.benchmark --help
```

By default the harness scores GeoVirus with all four main pipelines. A small run can be started with:

```bash
python -m scripts.benchmark \
  --corpus geovirus \
  --pipeline upstream \
  --pipeline hybrid \
  --limit 20 \
  --device auto \
  --output-dir benchmark-evidence/local-smoke
```

The command creates one folder per corpus, with a checkpoint and Markdown and JSON report for each selected pipeline. It also writes `summary.md` and `summary.json` at the output root. Repeating the same command resumes checkpoints whose run identity still matches.

### Options

| Option | Purpose |
| --- | --- |
| `--corpus NAME` | Select one or more registered corpora; use `all` for every registered split. |
| `--pipeline NAME` | Select a pipeline; repeat the flag to compare several. |
| `--phase recognition\|resolution` | Run one or both phases; by default both run. |
| `--limit N` | Keep the first N documents in deterministic corpus order. |
| `--device auto\|cpu\|cuda` | Choose model placement; `auto` uses CUDA when available. |
| `--chunk-size N` | Set how many documents are processed before saving a checkpoint. |
| `--min-similarity VALUE` | Set the shared resolver abstention threshold. The benchmark default is `0.0`. |
| `--output-dir PATH` | Choose where checkpoints and reports are stored. |

### Corpora

The registry includes GeoVirus; the HIPE-2022 test splits `hipe2020` (English, French, German), `newseye` (German, Finnish, French, Swedish), and `topres19th` (English); and NewsLi in Arabic, German, Spanish, Persian, Japanese, Polish, Romanian, Serbian, Tamil, Turkish, and Ukrainian. The harness uses unmasked test files. HIPE location entities are scored when their Wikidata item has coordinates; NewsLi is capped at 500 documents per language, selected in identifier order.

The downloadable corpus text stays in the local cache and is not published by the benchmark uploader. The benchmark dataset contains reports, checkpoints, logs, a results table, and charts. Source terms remain attached to their corpora:

| Corpus | Source and terms |
| --- | --- |
| GeoVirus | Included in the [UniTopRank data release](https://figshare.com/articles/software/UniTopRank/30445541), marked Apache 2.0. The harness downloads the XML from [the original GeoVirus repository](https://github.com/milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation). |
| NewsLi | [UniTopRank data release](https://figshare.com/articles/software/UniTopRank/30445541), marked Apache 2.0. |
| HIPE-2022 splits | [HIPE-2022 data release](https://github.com/hipe-eval/HIPE-2022-data), CC BY-NC-SA 4.0. |

The benchmark repository uses `other` as its Hub license marker because it contains evidence associated with sources under different terms. Consult the relevant corpus release before reusing source material.

## Pipelines

| Name | Recognition | Resolution |
| --- | --- | --- |
| `upstream` | spaCy `en_core_web_sm` | MiniLM `dguzh/geo-all-MiniLM-L6-v2` |
| `swapped` | GLiNER2.5 | Jina embeddings v5 and Jina reranker v3.5; retained as a historical comparison. |
| `hybrid` | GLiNER2.5 | MiniLM `dguzh/geo-all-MiniLM-L6-v2` |
| `prior` | GLiNER2.5 | MiniLM with population weight `0.3`; inflection fallback off by default. |

Seven additional names are available for ablations: `trim` enables inflection fallback alone, `population` uses the original low population weight, and `population-0.05`, `population-0.2`, `population-0.3`, `population-0.5`, and `population-1.0` sweep the population weight with fallback off. These are selectable but are not part of the default run.

Recognition scores each pipeline's own spans by exact match. Resolution supplies every pipeline with the same gold spans and scores the distance to the predicted location, so the resolution comparison does not depend on each recognizer's recall. The [published results](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) include the metrics, per-corpus charts, ablations, and evidence files.

## PAN-X location recognition

The separate PAN-X experiment compares the English upstream model,
`fastino/gliner2.5-multi-v1`, and `Davlan/xlm-roberta-base-ner-hrl` on the
existing WikiANN test splits. Its 85 target language codes and the WikiANN
intersection are pinned in `scripts/panx_benchmark/`; the upstream language
source and commit are recorded beside the list. The pinned dataset revision has
test splits for 82 target languages (`ha`, `xh`, and `zu` are absent), totaling
423,100 examples.

Run a small, real-inference feasibility sample across every available language
with:

```bash
python -m scripts.panx_benchmark --limit-per-language 8
```

This is a bounded sample from the pinned test splits, not a quality result. The
full run uses every test example and can take substantial CPU time:

```bash
python -m scripts.panx_benchmark
```

The report records per-language exact-span location precision, recall and F1,
macro and micro aggregates, test row counts, documented support versus
cross-lingual transfer, training-overlap knowledge, checkpoint download and
model-load times, and steady-state throughput. All models use the same joined
WikiANN text, character offsets, fixed batch size, and CPU host. The spaCy
control is scored on English only. No training, fine-tuning, resolution,
publication, or dataset-scale export occurs. Reports are written locally under
`benchmark-evidence/panx/` and are never uploaded by this command.

### Historical feasibility evidence

The September 30 sample in `benchmark-evidence/panx/feasibility-2026-09-30/`
has an unavailable source revision. Its JSON and Markdown flag it as unverified
and unsuitable for pipeline selection. It is retained as historical evidence,
not a reproducible benchmark. Legacy timings include metric bookkeeping; new
runs time prediction calls separately and must not combine timing estimates
with those older records. Publish the exact source revision before treating
any new run as remotely reproducible.

## Grid'5000

The submission wrapper checks for a duplicate active OAR job, requests one GPU and a bounded walltime, and submits `run_benchmark.sh` to the node. The node script builds GeoNames on local scratch, keeps model caches and the database off the home quota, and writes checkpoints to the selected results directory.

For example, run only the resolution phase of two choices over all corpora with:

```bash
CORPORA=all PIPELINES=prior,trim PHASES=resolution ./scripts/g5k/submit.sh
```

`CORPORA`, `PIPELINES`, and `PHASES` accept comma-separated names. `LIMIT`, `DEVICE`, `CHUNK_SIZE`, `MIN_SIMILARITY`, `WALLTIME`, `GPUS`, `QUEUE`, `CLUSTER`, `REPO_ROOT`, and `RESULTS` can also be set when submitting. Run `usagepolicycheck -t` before and after using Grid'5000.

## Publish the results

After preserving the evidence directory, publish it to the benchmark dataset:

```bash
python -m scripts.benchmark.publish \
  --repo-id NoeFlandre/geoparser-benchmark-results
```

The publisher uploads the evidence under `runs/`, regenerates `results.csv`, creates charts from the latest reports, and rebuilds the Hub dataset card. It does not upload the source corpus text. Keep the full output directory: its checkpoints and logs allow a run to be resumed or audited.
