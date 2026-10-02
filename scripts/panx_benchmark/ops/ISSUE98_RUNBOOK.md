# Issue 98 checkpointed runbook

## Scope and identity

This run performs issue 98's WikiANN/PAN-X named-place recognition comparison. It does not run the geoparser resolver. It uses the source tree at `NoeFlandre/geoparser` commit `cd0c750309fa5aacd8a4f9ecb75b71f6f9230380`, the dataset revision and three model revisions pinned by `scripts/panx_benchmark`, and CPU inference. The immutable current-protocol snapshot is `62a90f96e0db98c1d881feb316bd8a2205c8a1665a152941b8b319abd4079653`.

The benchmark checkpoints completed model/language records under this snapshot. The uploader writes to the dataset repository `NoeFlandre/geoparser-benchmark-results`, under:

`runs/panx/2026-10-02-cd0c750-transformers-5-18-cpu/checkpoints/62a90f96e0db98c1d881feb316bd8a2205c8a1665a152941b8b319abd4079653/`

At the initial checkpoint sync, HF commit `0502a2f0b7f7ab2ae6717001ee0353ccf128ab94` held the snapshot manifest, 82 spaCy language/status records, and GLiNER `af` and `am` records. The active Arabic GLiNER record was still running at that point. The uploader records local SHA-256 values and downloads each file from its exact upload commit before marking that path verified.

The older incomplete archive stays separate. It is at results-dataset commit `270ec29b5692ae3792cef7ae76e40addaa586e04`, path `runs/incomplete/2026-10-01-panx-continuation-a9b1e86d-incomplete.zip`, SHA-256 `2c042e299087c59e4160aa644304b93f0a6e8b4af429e3926918c2d81d2eaff6`. Do not merge those records into this snapshot.

## Resume in the same workspace

Keep the existing run root, checkpoint cache, HF cache, and `upload-state.json`. Install the repository-locked environment with uv 0.11.16 and `uv sync --locked`, then run:

```sh
cd /workspace/geoparser
HF_HUB_CACHE=/workspace/hf-cache/hub \
HF_XET_CACHE=/workspace/hf-cache/xet \
/workspace/geoparser/.venv/bin/python /workspace/panx-run/run_full_with_verified_sync.py
```

The helper runs model inference offline from the pinned local caches. Its uploader stays online. It resumes from atomic local checkpoints and repeats no completed model/language work whose identity matches this snapshot.

Start the readback gate as soon as the benchmark subprocess appears. Find its PID as the child of `run_full_with_verified_sync.py`, then run:

```sh
/workspace/geoparser/.venv/bin/python \
  /workspace/panx-run/gate_next_language.py EVALUATOR_PID
```

The gate watches for each new atomic GLiNER language checkpoint. It pauses the evaluator until `upload-state.json` contains the same SHA-256 as a verified remote readback. If the remote upload fails, leave the evaluator paused and resolve the preservation error before continuing.

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
5. Start the gate before allowing the evaluator to begin another GLiNER language.

## Outputs and limits

The evaluator writes the final JSON and Markdown report to `/workspace/panx-run/full/report`. Upload those as individual files under a distinct `reports/` path and verify their SHA-256 values by reading them back from the HF results dataset. Do not use an archive upload; the earlier archive attempt was rejected by the Hub's Xet upload endpoint. Keep this run's outputs separate from the legacy incomplete archive.

The preflight estimated about 25.6 CPU hours for all three-model inference. This is a linear estimate from one example per eligible language, not a completion-time guarantee. The report must state measured hardware and batching, model and data revisions, per-language precision/recall/F1, macro scores, timing method, the English-only spaCy coverage, XLM-R's ten fine-tuned languages and transfer results, and all missing-language or dataset limitations. Use the results to decide whether to continue the issue 98 pilot. Do not expand to OSM datasets or add benchmarks.
