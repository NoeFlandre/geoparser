# Benchmarking

The benchmark harness measures recognition and resolution on fixed corpora. It saves a checkpoint after each chunk of documents. Thus you can resume a stopped run without repeated work. The reports record the input digest, the commit, the models, the metrics, and the elapsed time.

## Run locally

Install the benchmark extras and the `geonames` gazetteer. Then examine the CLI options:

```bash
python -m scripts.benchmark --help
```

By default, the harness scores GeoVirus with all four main pipelines. Start a small run with this command:

```bash
python -m scripts.benchmark \
  --corpus geovirus \
  --pipeline upstream \
  --pipeline hybrid \
  --limit 20 \
  --device auto \
  --output-dir benchmark-evidence/local-smoke
```

The command creates one folder for each corpus. Each folder has a checkpoint and a Markdown report and a JSON report for each selected pipeline. The command also writes `summary.md` and `summary.json` at the output root. If you do the same command again, it resumes the checkpoints whose run identity is still the same.

### Options

| Option | Purpose |
| --- | --- |
| `--corpus NAME` | Selects one or more registered corpora. Use `all` for every registered split. |
| `--pipeline NAME` | Selects a pipeline. Repeat the flag to compare more than one. |
| `--phase recognition\|resolution` | Runs one phase or both phases. By default, both phases run. |
| `--limit N` | Keeps the first N documents in the fixed corpus order. |
| `--device auto\|cpu\|cuda` | Selects where the models run. `auto` uses CUDA when it is available. |
| `--chunk-size N` | Sets how many documents the harness processes before it saves a checkpoint. |
| `--min-similarity VALUE` | Sets the shared abstention threshold of the resolver. The benchmark default is `0.0`. |
| `--output-dir PATH` | Selects where the harness stores checkpoints and reports. |

### Corpora

The registry includes these corpora:

- GeoVirus.
- The HIPE-2022 test splits `hipe2020` (English, French, German), `newseye` (German, Finnish, French, Swedish), and `topres19th` (English).
- NewsLi in Arabic, German, Spanish, Persian, Japanese, Polish, Romanian, Serbian, Tamil, Turkish, and Ukrainian.

The harness uses unmasked test files. It scores a HIPE location entity when its Wikidata item has coordinates. NewsLi has a limit of 500 documents for each language. The harness selects them in identifier order.

Other corpora from the UniTopRank release and from TopoResolve are listed with their status and reasons in the [geographic corpora inventory](geographic-corpora-inventory.md). None of them is registered yet.

The text of the downloadable corpora stays in the local cache. The benchmark uploader does not publish it. The benchmark dataset contains reports, checkpoints, logs, a results table, and charts. The source terms stay with their corpora:

| Corpus | Source and terms |
| --- | --- |
| GeoVirus | Included in the [UniTopRank data release](https://figshare.com/articles/software/UniTopRank/30445541), marked Apache 2.0. The harness downloads the XML from [the original GeoVirus repository](https://github.com/milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation). |
| NewsLi | [UniTopRank data release](https://figshare.com/articles/software/UniTopRank/30445541), marked Apache 2.0. |
| HIPE-2022 splits | [HIPE-2022 data release](https://github.com/hipe-eval/HIPE-2022-data), CC BY-NC-SA 4.0. |

The benchmark repository uses `other` as its Hub license marker. It contains evidence that is associated with sources that have different terms. Read the release of the corpus before you reuse source material.

## Pipelines

| Name | Recognition | Resolution |
| --- | --- | --- |
| `upstream` | spaCy `en_core_web_sm` | MiniLM `dguzh/geo-all-MiniLM-L6-v2` |
| `swapped` | GLiNER2.5 | Jina embeddings v5 and Jina reranker v3.5. It is kept as a historical comparison. |
| `hybrid` | GLiNER2.5 | MiniLM `dguzh/geo-all-MiniLM-L6-v2` |
| `prior` | GLiNER2.5 | MiniLM with population weight `0.3`. The inflection fallback is off by default. |

Seven more names are available for ablations. `trim` enables the inflection fallback alone. `population` uses the original low population weight. `population-0.05`, `population-0.2`, `population-0.3`, `population-0.5`, and `population-1.0` sweep the population weight with the fallback off. You can select these names, but they are not part of the default run.

Recognition scores the spans of each pipeline by exact match. Resolution gives the same gold spans to each pipeline. It scores the distance to the predicted location. Thus the resolution comparison does not depend on the recall of each recognizer. The [published results](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) include the metrics, the charts for each corpus, the ablations, and the evidence files.

## PAN-X location recognition

The separate PAN-X experiment compares three models on the existing WikiANN test splits. The models are the English upstream model, `fastino/gliner2.5-multi-v1`, and `Davlan/xlm-roberta-base-ner-hrl`. The 85 target language codes and the WikiANN intersection are pinned in `scripts/panx_benchmark/`. The source and the commit of the upstream language list are recorded beside the list. The pinned dataset revision has test splits for 82 target languages (`ha`, `xh`, and `zu` are not present). The splits have 423,100 examples in total.

Use this command to run a small feasibility sample with real inference on every available language:

```bash
python -m scripts.panx_benchmark --limit-per-language 8
```

This is a bounded sample from the pinned test splits. It is not a quality result. The full run uses every test example. It can need much CPU time:

```bash
python -m scripts.panx_benchmark
```

The report records these items:

- The precision, the recall, and the F1 score of the exact-span location match for each language.
- The macro aggregates and the micro aggregates.
- The number of test rows.
- Whether the model documents support for the language or uses cross-lingual transfer.
- The knowledge of training overlap.
- The checkpoint download times and the model load times.
- The steady-state throughput.

All models use the same joined WikiANN text, the same character offsets, the same fixed batch size, and the same CPU host. The spaCy control is scored on English only. The command does no training, no fine-tuning, no resolution, no publication, and no dataset-scale export. It writes reports locally under `benchmark-evidence/panx/`. It never uploads them.

### Historical feasibility evidence

The sample of September 30 is in `benchmark-evidence/panx/feasibility-2026-09-30/`. Its source revision is not available. Its JSON file and Markdown file flag it as unverified and not suitable for pipeline selection. It is kept as historical evidence. It is not a reproducible benchmark. The legacy timings include the metric bookkeeping. New runs time the prediction calls separately. Do not combine the timing estimates of new runs with those older records. Publish the exact source revision before you treat a new run as reproducible by other people.

## Grid'5000

The submission wrapper checks for a duplicate active OAR job. It requests one GPU and a limited walltime, and it submits `run_benchmark.sh` to the node. The node script builds GeoNames on local scratch. It keeps the model caches and the database off the home quota. It writes the checkpoints to the results directory that you selected.

For example, use this command to run only the resolution phase of two pipelines on all corpora:

```bash
CORPORA=all PIPELINES=prior,trim PHASES=resolution ./scripts/g5k/submit.sh
```

`CORPORA`, `PIPELINES`, and `PHASES` accept names that are separated by commas. You can also set `LIMIT`, `DEVICE`, `CHUNK_SIZE`, `MIN_SIMILARITY`, `WALLTIME`, `GPUS`, `QUEUE`, `CLUSTER`, `REPO_ROOT`, and `RESULTS` when you submit. Run `usagepolicycheck -t` before and after you use Grid'5000.

## Publish the results

Keep the evidence directory. Then publish it to the benchmark dataset:

```bash
python -m scripts.benchmark.publish \
  --repo-id NoeFlandre/geoparser-benchmark-results
```

The publisher uploads the evidence under `runs/`. It regenerates `results.csv`, makes charts from the latest reports, and rebuilds the Hub dataset card. It does not upload the source corpus text. Keep the complete output directory. Its checkpoints and logs let you resume or audit a run.
