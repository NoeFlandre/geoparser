# Resolver and Persistence Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resolver caches, vectorized similarity scoring, bulk processing-status queries, and atomic batch persistence without changing public prediction results.

**Architecture:** Keep recognizers/resolvers database-free. Add resolver-local immutable caches and a batched similarity helper in `SentenceTransformerResolver`; pass precomputed score lists into the existing scalar selection hooks. Add ID-filtered bulk repository queries and service-owned `add_all`/single-commit write paths, preserving input order and making each service prediction call atomic.

**Tech Stack:** Python 3.10–3.14, PyTorch, SQLModel/SQLAlchemy, SQLite, pytest, Ruff, ty.

---

### Task 1: Add failing resolver cache and vectorization tests

**Files:**
- Modify: `tests/unit/test_modules/test_resolvers/test_sentencetransformer.py`
- Modify: `tests/unit/test_modules/test_resolvers/test_jina.py`

- [ ] **Step 1: Write cache tests**

Add tests that call the resolver with repeated identical reference text and assert the gazetteer search is called once, candidate descriptions are generated once, and measured sentences are computed once for a long document.

- [ ] **Step 2: Write the vectorization equivalence test**

Add a test with uneven candidate-list lengths, including an empty list, that compares the new batch score output with the existing scalar cosine scores and asserts that the batch path performs one cosine-similarity call.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -o addopts='' tests/unit/test_modules/test_resolvers/test_sentencetransformer.py tests/unit/test_modules/test_resolvers/test_jina.py -q
```

Expected: the new cache/vectorization tests fail because the caches and batch helper do not exist; existing tests continue to pass.

### Task 2: Implement resolver caches and vectorized scoring

**Files:**
- Modify: `geoparser/modules/resolvers/sentencetransformer.py`
- Modify: `geoparser/modules/resolvers/jina.py`

- [ ] **Step 1: Add cache state and cache helpers**

Add instance dictionaries for normalized search queries, candidate descriptions, and measured sentences. Route `_gather_candidates`, `_embed_candidates`, Jina reranking, and `_extract_context` through the helpers. Cache empty search results and return immutable tuples from the search cache.

- [ ] **Step 2: Add the batched score helper**

Implement a helper that receives aligned context strings and candidate lists, concatenates all candidate embedding tensors, repeats context embeddings according to candidate-list lengths, runs one `torch.nn.functional.cosine_similarity` call, and splits the result back into per-reference score lists. Return empty lists for empty candidates.

- [ ] **Step 3: Thread scores through evaluation**

Have `_evaluate_candidates` compute batch scores for unresolved references once per search pass and pass each score list to `_best_referent`. Keep the optional score argument backward-compatible so direct scalar callers and subclass tests still work. Preserve candidate order, threshold checks, and Jina shortlist ordering.

- [ ] **Step 4: Run the focused resolver tests and verify GREEN**

Run the command from Task 1. Expected: all resolver unit tests pass.

- [ ] **Step 5: Run formatting and lint for the changed modules**

Run:

```bash
uv run ruff check geoparser/modules/resolvers/sentencetransformer.py geoparser/modules/resolvers/jina.py tests/unit/test_modules/test_resolvers/test_sentencetransformer.py tests/unit/test_modules/test_resolvers/test_jina.py
uv run ruff format --check geoparser/modules/resolvers/sentencetransformer.py geoparser/modules/resolvers/jina.py tests/unit/test_modules/test_resolvers/test_sentencetransformer.py tests/unit/test_modules/test_resolvers/test_jina.py
```

### Task 3: Add failing bulk-query and atomic-write tests

**Files:**
- Modify: `tests/unit/test_services/test_recognition.py`
- Modify: `tests/unit/test_services/test_resolution.py`
- Modify: `tests/unit/test_db/test_crud/test_recognition.py`
- Modify: `tests/unit/test_db/test_crud/test_resolution.py`

- [ ] **Step 1: Write repository bulk-query tests**

Add tests for ID-filtered status queries that return only requested records and verify the service can reconstruct the original input order.

- [ ] **Step 2: Write recognition service batching tests**

Add a test that supplies multiple documents and predictions, asserts `Session.add_all` is used for the references/recognition records, and asserts one commit for the prediction batch. Add a failure test asserting the session rolls back and no later records are committed.

- [ ] **Step 3: Write resolution service batching tests**

Add equivalent tests for referents/resolution records. Include an invalid gazetteer identifier and assert the whole prediction batch is rejected atomically.

- [ ] **Step 4: Run the focused service/repository tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -o addopts='' tests/unit/test_services/test_recognition.py tests/unit/test_services/test_resolution.py tests/unit/test_db/test_crud/test_recognition.py tests/unit/test_db/test_crud/test_resolution.py -q
```

Expected: only the new bulk/atomicity tests fail; existing behavior tests pass.

### Task 4: Implement bulk status queries and atomic service writes

**Files:**
- Modify: `geoparser/db/crud/recognition.py`
- Modify: `geoparser/db/crud/resolution.py`
- Modify: `geoparser/services/recognition.py`
- Modify: `geoparser/services/resolution.py`

- [ ] **Step 1: Add ID-filtered repository queries**

Add methods that select processed document/reference IDs for the requested ID sequence and recognizer/resolver. Use one SQL statement per status check, then let the service filter its original objects in order.

- [ ] **Step 2: Batch recognition writes**

Build `Reference` model instances with the document text populated, build `Recognition` instances, add all objects to the session, flush once, and commit once after all predictions are validated. Roll back on any exception.

- [ ] **Step 3: Batch resolution writes**

Resolve and validate every predicted gazetteer feature first. Build `Referent` and `Resolution` instances, add them all, flush, and commit once. Roll back the whole prediction batch on validation or database failure.

- [ ] **Step 4: Run the focused service/repository tests and verify GREEN**

Run the command from Task 3. Expected: all focused tests pass, including the new query-count and rollback tests.

- [ ] **Step 5: Run service integration tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -o addopts='' tests/integration/test_services/test_recognition_service_integration.py tests/integration/test_services/test_resolution_service_integration.py -q
```

### Task 5: Add end-to-end regression coverage and run the full gate

**Files:**
- Modify: `tests/unit/test_benchmark/test_runner.py` if the optimized path needs benchmark-specific coverage
- Modify: `docs/technical-debt.md` only if the implementation leaves a documented limitation

- [ ] **Step 1: Run all unit, property, and acceptance tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -o addopts='' tests/unit tests/property tests/acceptance -q
```

- [ ] **Step 2: Run static checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check geoparser scripts tests
git diff --check
```

- [ ] **Step 3: Inspect the final diff and verify the requested checklist**

Confirm that the diff contains resolver caches, one batched similarity path, bulk status queries, atomic service writes, and tests for each. Confirm no benchmark model, threshold, scoring, or public interface changed.

- [ ] **Step 4: Commit the implementation**

```bash
git add geoparser tests
git commit -m "perf: batch resolver scoring and persistence"
```

- [ ] **Step 5: Verify clean state**

```bash
git status --porcelain=v1 -b
git log -2 --oneline --decorate
```

Expected: no uncommitted changes and the implementation commit is at `HEAD`.
