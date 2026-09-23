# Hybrid Resolver Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately named hybrid benchmark pipeline that keeps GLiNER2 recognition but replaces the failed Jina resolver with the measured upstream MiniLM resolver, then run and archive a fresh Grid’5000 comparison.

**Architecture:** The existing upstream and historical swapped pipelines remain selectable. A new `hybrid` factory branch composes `GLiNER2Recognizer` with `SentenceTransformerResolver`, so the runner, checkpoint identity, scoring, and report code remain shared and old Jina evidence is not overwritten.

**Tech Stack:** Python 3.13, pytest, Ruff, SentenceTransformers, GLiNER2, Grid’5000/OAR, checkpointed JSON benchmark reports.

---

### Task 1: Pin hybrid pipeline construction

**Files:**
- Create: `tests/unit/test_benchmark/test_pipelines.py`
- Modify: `scripts/benchmark/pipelines.py`

- [ ] **Step 1: Write the failing tests**

Create factory tests that patch the lazy-loaded model classes and assert the hybrid composition and explicit device placement:

```python
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.benchmark import pipelines


def test_hybrid_uses_gliner2_for_recognition(monkeypatch):
    recognizer = Mock()
    gliner = Mock(return_value=recognizer)
    monkeypatch.setattr("geoparser.modules.GLiNER2Recognizer", gliner)

    result = pipelines.build_recognizer(pipelines.HYBRID, device="cuda")

    assert result is recognizer
    gliner.assert_called_once_with()


def test_hybrid_uses_minilm_sentence_transformer_for_resolution(monkeypatch):
    resolver = SimpleNamespace(model_name="dguzh/geo-all-MiniLM-L6-v2")
    sentence_transformer = Mock(return_value=resolver)
    monkeypatch.setattr(
        "geoparser.modules.SentenceTransformerResolver", sentence_transformer
    )
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")

    result = pipelines.build_resolver(
        pipelines.HYBRID, device="cuda", min_similarity=0.0
    )

    assert result is resolver
    sentence_transformer.assert_called_once_with(
        gazetteer_name="geonames", min_similarity=0.0
    )


def test_hybrid_moves_both_models_to_requested_device(monkeypatch):
    transformer = Mock()
    resolver = SimpleNamespace(transformer=transformer, reranker=None)
    factory = Mock(return_value=resolver)
    monkeypatch.setattr("geoparser.modules.SentenceTransformerResolver", factory)

    pipelines.build_resolver(pipelines.HYBRID, device="cuda", min_similarity=0.0)

    transformer.to.assert_called_once_with("cuda")
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/geoparser-uv-cache uv run --no-sync pytest -q tests/unit/test_benchmark/test_pipelines.py --confcutdir=tests/unit/test_benchmark
```

Expected: collection or assertion failure because `HYBRID` and its factory branch do not yet exist.

- [ ] **Step 3: Implement the minimal factory branch**

Add `HYBRID = "hybrid"` and include it in `PIPELINES`. Keep `build_recognizer`’s GLiNER2 branch for every non-upstream pipeline. In `build_resolver`, use `SentenceTransformerResolver` for `UPSTREAM` and `HYBRID`, and keep `JinaResolver` only for historical `SWAPPED`. Update docstrings to name all three choices and explain the hybrid composition.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the same pytest command. Expected: all new pipeline tests pass.

- [ ] **Step 5: Commit the factory change**

```bash
git add scripts/benchmark/pipelines.py tests/unit/test_benchmark/test_pipelines.py
git commit -m "feat: add hybrid MiniLM benchmark pipeline"
```

### Task 2: Validate the benchmark wiring

**Files:**
- Modify: `tests/unit/test_benchmark/test_checkpoint.py`
- Modify: `tests/unit/test_benchmark/test_report.py`
- Modify: `scripts/g5k/run_benchmark.sh`
- Modify: `scripts/g5k/submit.sh`

- [ ] **Step 1: Add the identity and report coverage**

Add a checkpoint identity case using `pipeline="hybrid"` and extend the report fixture to render `upstream` and `hybrid` rows. Keep the historical `swapped` fixture test so the old report contract remains valid.

- [ ] **Step 2: Run the benchmark unit suite**

```bash
UV_CACHE_DIR=/tmp/geoparser-uv-cache uv run --no-sync pytest -q tests/unit/test_benchmark --confcutdir=tests/unit/test_benchmark
```

Expected: all benchmark unit tests pass.

- [ ] **Step 3: Add an explicit pipeline selection to the Grid’5000 wrapper**

Keep the current default of running every pipeline, but let a run select a comma-separated subset without changing checkpoint semantics. In `run_benchmark.sh`, append one `--pipeline` argument for each name in `PIPELINES` when that variable is non-empty. In `submit.sh`, forward `PIPELINES` to the job command. The fresh run will therefore use `PIPELINES='upstream,hybrid'` and will not recreate or overwrite the historical Jina result.

- [ ] **Step 4: Run formatting and static checks**

```bash
UV_CACHE_DIR=/tmp/geoparser-uv-cache uv run --no-sync ruff check scripts/benchmark tests/unit/test_benchmark
UV_CACHE_DIR=/tmp/geoparser-uv-cache uv run --no-sync ruff format --check scripts/benchmark tests/unit/test_benchmark
python -m compileall -q scripts/benchmark
git diff --check
```

Expected: each command exits zero.

- [ ] **Step 5: Validate shell syntax**

```bash
bash -n scripts/g5k/run_benchmark.sh scripts/g5k/submit.sh
```

Expected: exit zero.

- [ ] **Step 6: Commit benchmark wiring**

```bash
git add scripts/benchmark scripts/g5k tests/unit/test_benchmark
git commit -m "test: validate hybrid benchmark wiring"
```

### Task 3: Publish the runnable branch and preserve old evidence

**Files:**
- No source files; remote state only.

- [ ] **Step 1: Verify the branch and old evidence locally**

Run `git status --short -b`, confirm the new commits are on `feat/geovirus-benchmark`, and retain the prior report values from `/home/nflandre/geoparser-bench` as the Jina baseline.

- [ ] **Step 2: Push the runnable branch**

```bash
git push origin feat/geovirus-benchmark
```

- [ ] **Step 3: Synchronize Nancy to the pushed commit**

```bash
ssh nancy 'cd "$HOME/geoparser" && git fetch origin && git checkout feat/geovirus-benchmark && git pull --ff-only origin feat/geovirus-benchmark'
```

Verify `git rev-parse HEAD` remotely equals the local commit before submission.

### Task 4: Run the fresh Grid’5000 benchmark

**Files:**
- Remote results: `$HOME/geoparser-bench-hybrid/`

- [ ] **Step 1: Confirm no duplicate job exists**

```bash
ssh nancy 'oarstat -u'
```

Use a new job name, `geoparser-hybrid-bench`, and a new results directory so the old Jina report and checkpoints cannot be reused.

- [ ] **Step 2: Submit upstream and hybrid only**

```bash
ssh nancy 'cd "$HOME/geoparser" && JOB_NAME=geoparser-hybrid-bench WALLTIME=2:00 CHUNK_SIZE=10 CLUSTER=grue PIPELINES=upstream,hybrid RESULTS="$HOME/geoparser-bench-hybrid" bash scripts/g5k/submit.sh'
```

The job must run with the existing default `DEVICE=auto` and `MIN_SIMILARITY=0.0`; the wrapper adds `--pipeline upstream --pipeline hybrid` from `PIPELINES`.

- [ ] **Step 3: Poll without killing the job**

Read `oarstat -u` and the newest output log periodically. A preempted or walltime-ended job is resumable; resubmit the same job name and results directory only after `submit.sh` confirms no active duplicate.

- [ ] **Step 4: Verify the completed report**

Require 229 documents, 2167 gold toponyms, corpus digest `2f8a928b9af4a442`, the current source commit, both `upstream` and `hybrid` rows, and model names showing GLiNER2 plus `dguzh/geo-all-MiniLM-L6-v2` for hybrid.

### Task 5: Copy evidence off Nancy and report the result

**Files:**
- Create locally: `benchmark-evidence/2026-09-22-hybrid/benchmark-report.md`
- Create locally: `benchmark-evidence/2026-09-22-hybrid/benchmark-report.json`

- [ ] **Step 1: Copy the report and provenance**

```bash
mkdir -p benchmark-evidence/2026-09-22-hybrid
scp nancy:/home/nflandre/geoparser-bench-hybrid/benchmark-report.md benchmark-evidence/2026-09-22-hybrid/
scp nancy:/home/nflandre/geoparser-bench-hybrid/benchmark-report.json benchmark-evidence/2026-09-22-hybrid/
```

- [ ] **Step 2: Verify the copied JSON**

Parse it locally, compare the corpus digest and commit with the run identity, and ensure both pipeline entries are complete before claiming a result.

- [ ] **Step 3: Commit the evidence and final state**

```bash
git add benchmark-evidence/2026-09-22-hybrid
git commit -m "bench: record hybrid MiniLM Grid5000 results"
```

- [ ] **Step 4: Verify the final checkout**

```bash
git status --short -b
git log --oneline -6
```

Expected: a clean branch, with the report and all benchmark commits present.
