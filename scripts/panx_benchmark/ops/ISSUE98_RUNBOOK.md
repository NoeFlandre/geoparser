# Issue 98 checkpointed runbook

## Scope and identity

This run performs issue 98's WikiANN/PAN-X named-place recognition comparison. It does not run the geoparser resolver. It uses the source tree at `NoeFlandre/geoparser` commit `cd0c750309fa5aacd8a4f9ecb75b71f6f9230380`, the dataset revision and three model revisions pinned by `scripts/panx_benchmark`, and CPU inference. The immutable current-protocol snapshot is `62a90f96e0db98c1d881feb316bd8a2205c8a1665a152941b8b319abd4079653`.

The benchmark checkpoints completed model/language records under this snapshot. The uploader writes to the dataset repository `NoeFlandre/geoparser-benchmark-results`, under:

`runs/panx/2026-10-02-cd0c750-transformers-5-18-cpu/checkpoints/62a90f96e0db98c1d881feb316bd8a2205c8a1665a152941b8b319abd4079653/`

At the initial checkpoint sync, HF commit `0502a2f0b7f7ab2ae6717001ee0353ccf128ab94` held the snapshot manifest, 82 spaCy language/status records, and GLiNER `af` and `am` records. The active Arabic GLiNER record was still running at that point. The uploader records local SHA-256 values and downloads each file from its exact upload commit before marking that path verified.

The older incomplete archive stays separate. It is at results-dataset commit `270ec29b5692ae3792cef7ae76e40addaa586e04`, path `runs/incomplete/2026-10-01-panx-continuation-a9b1e86d-incomplete.zip`, SHA-256 `2c042e299087c59e4160aa644304b93f0a6e8b4af429e3926918c2d81d2eaff6`. Do not merge those records into this snapshot.

## Separate historical runs

The results dataset has another immutable incomplete run at commit `2d8403bca7edf01d903da10d08dd7e561b440a4a`. Its path is `runs/2026-10-01-panx-335eae9d25892ad70be42dac8855af8cf9e4fc23091658599e6527a26ecf3c1d/`. Its snapshot ID is `335eae9d25892ad70be42dac8855af8cf9e4fc23091658599e6527a26ecf3c1d`. Its source commit is `7cdd960dc559089aacb1614019a0f6dc0c3a409e`. It uses the same WikiANN revision and model revisions as this run. Its 39 completed GLiNER languages are `af`, `am`, `ar`, `az`, `be`, `bg`, `bn`, `ca`, `ceb`, `cs`, `cy`, `da`, `de`, `el`, `en`, `eo`, `es`, `et`, `eu`, `fa`, `fi`, `fr`, `fy`, `ga`, `gd`, `gl`, `gu`, `he`, `hi`, `hu`, `hy`, `id`, `ig`, `is`, `it`, `ja`, `jv`, and `kk`. It also evaluated spaCy English on 10,000 examples. It did not start XLM-R. The other 81 spaCy rows are status records, not evaluated examples. The run stopped after a GLiNER model load reached its 16 GiB cgroup limit. Its environment file records three cgroup OOM kill events.

We verified the historical archive at its exact HF commit. Its SHA-256 manifest lists 129 files. The archived path set has those 129 files plus the manifest itself. We checked the SHA-256 entries for its README, snapshot, run status, summary, environment, attempts, and lock file. The lock-file digest matches `environment.json`.

Keep this run separate from the current run. The historical source used Transformers 4.57.6 and batch size 8 for all models. Its timer included score aggregation. The current run uses Transformers 5.18.0, GLiNER batch size 1, and prediction-only timing. These source, dependency, batching, and timing differences mean that the runs do not form one comparable result matrix. Do not fill current-protocol language rows with historical scores or combine their timing totals.

The earlier continuation is a third record. It is at results-dataset commit `270ec29b5692ae3792cef7ae76e40addaa586e04`, in `runs/incomplete/2026-10-01-panx-continuation-a9b1e86d-incomplete.zip`. It has 13 saved GLiNER language records and 82 spaCy language/status records. Of the spaCy rows, English was evaluated and 81 were marked not evaluated. Its source is the separate `dbbc3ad31021799f253c59968be473afc47bd530` protocol. Keep it separate from both the 39-language archive and this current run.

## Resume in the same workspace

Keep the existing run root, checkpoint cache, HF cache, and `upload-state.json`. Install the repository-locked environment with uv 0.11.16 and `uv sync --locked`, then run:

```sh
cd /workspace/geoparser
HF_HUB_CACHE=/workspace/hf-cache/hub \
HF_XET_CACHE=/workspace/hf-cache/xet \
/workspace/geoparser/.venv/bin/python /workspace/panx-run/run_full_with_verified_sync.py
```

The helper runs model inference offline from the pinned local caches. Its uploader stays online. It resumes from atomic local checkpoints and repeats no completed model/language work whose identity matches this snapshot.

Start the readback gate as soon as the benchmark subprocess appears. Find its PID as the child of `run_full_with_verified_sync.py`. Start one gate for each model that still has languages to evaluate. Use these model keys:

```sh
/workspace/geoparser/.venv/bin/python \
  /workspace/panx-run/gate_next_language.py EVALUATOR_PID gliner2_multi

/workspace/geoparser/.venv/bin/python \
  /workspace/panx-run/gate_next_language.py EVALUATOR_PID xlmr_ner_hrl
```

Each gate watches for new atomic language checkpoints from its model. It pauses the evaluator until `upload-state.json` contains the same SHA-256 as a verified remote readback. If the remote upload fails, leave the evaluator paused and resolve the preservation error before continuing.

## Resume after workspace loss

1. Clone `NoeFlandre/geoparser` at source commit `cd0c750309fa5aacd8a4f9ecb75b71f6f9230380`. Install uv 0.11.16, then run `uv sync --locked`.
2. Set writable HF caches and use the existing authorized HF identity. Do not put access tokens in commands, files, or logs.
3. Restore only this snapshot from the results dataset. For this run, the invocation is:

   ```sh
   /workspace/geoparser/.venv/bin/python \
     /workspace/panx-run/restore_remote_checkpoints.py \
     --snapshot-id 62a90f96e0db98c1d881feb316bd8a2205c8a1665a152941b8b319abd4079653 \
     --checkpoint-dir /workspace/panx-run/full/checkpoints
   ```

   The helper reads the current dataset `main` revision unless `--revision` pins a known commit. It lists and downloads only files under this run's checkpoint prefix, and prints each restored SHA-256.
4. Run the same-workspace command above. The publisher may upload restored files again if the local upload state is absent. It verifies each new upload by exact-commit readback.
5. Start a gate for each model that still has languages to evaluate. Start each gate before the evaluator enters that model.

## Outputs and limits

The evaluator writes the final JSON and Markdown report to `/workspace/panx-run/full/report`. Upload those as individual files under a distinct `reports/` path and verify their SHA-256 values by reading them back from the HF results dataset. Do not use an archive upload; the earlier archive attempt was rejected by the Hub's Xet upload endpoint. Keep this run's outputs separate from the legacy incomplete archive.

The preflight estimated about 25.6 CPU hours for all three-model inference. This is a linear estimate from one example per eligible language, not a completion-time guarantee. The report must state measured hardware and batching, model and data revisions, per-language precision/recall/F1, macro scores, timing method, the English-only spaCy coverage, XLM-R's ten fine-tuned languages and transfer results, and all missing-language or dataset limitations. Use the results to decide whether to continue the issue 98 pilot. Do not expand to OSM datasets or add benchmarks.
