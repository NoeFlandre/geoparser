# Resolver and persistence performance design

## Goal

Reduce avoidable resolver and database overhead without changing public
prediction ordering, scores, identifiers, or the resumable benchmark contract.

## Scope

This change covers four related improvements:

1. Cache immutable gazetteer search results, candidate descriptions, and
   measured sentence/token data inside a resolver instance.
2. Compute embedding similarities for all candidate lists in one vectorized
   tensor operation per search pass instead of one operation and host
   conversion per reference.
3. Replace per-document and per-reference processing-status queries with
   bulk queries that preserve the caller's input order.
4. Batch database inserts and commit once per service invocation/batch. A
   failed batch is atomic and rolls back as one unit.

The model APIs and the public `Recognizer`, `Resolver`, `Project`, and service
interfaces remain unchanged.

## Design

### Resolver caches

`SentenceTransformerResolver` will keep three additional instance-local caches:

- `(normalized_name, method, tiers) -> tuple[Feature, ...]` for immutable
  gazetteer search results;
- `feature_id -> description` for the textual candidate description used by
  embedding and reranking;
- `document text -> tuple[Sentence, ...]` for sentence boundaries and token
  costs.

The existing embedding caches remain the source of truth for model outputs.
The new caches are safe to keep for the resolver lifetime because installed
gazetteer artifacts are read-only. Empty search results are cached too. Cache
keys use the same quote stripping and whitespace trimming as `Gazetteer.search`
so equivalent queries share work.

### Vectorized similarity scoring

The resolver will gather all unresolved `(context, candidate list)` pairs in a
search pass, flatten their candidate embeddings, repeat each context embedding
according to its candidate-list length, and calculate cosine similarities in a
single tensor call. The flat result is split back into one score list per
reference. Existing scalar `_best_referent` calls remain supported for direct
subclass/test use; the normal evaluation path receives the precomputed score
lists.

This reduces GPU launches and host synchronization while preserving candidate
order and the existing threshold/tie behavior. Empty candidate lists produce
empty score lists and are skipped exactly as before.

### Bulk status queries

Recognition and resolution services will query all requested document/reference
IDs in one database statement, then reconstruct the original caller order in
Python. Existing repository methods for whole-project bulk queries will be
reused or extended with ID-filtered variants so subset calls retain their
current semantics.

### Atomic batch writes

The services will construct model objects for a complete prediction batch,
`add_all` them, flush when generated IDs are needed, and commit once. A
database error rolls back the entire batch. Existing low-level single-record
repository methods remain available to callers that need them; the optimized
service path owns the batch boundary.

The resolution path will validate all predicted gazetteer referents before the
batch commit. This prevents a bad identifier from leaving a partially written
resolution result.

## Testing

RED tests will cover:

- duplicate gazetteer searches and descriptions are computed once;
- long-document sentence/token measurement is reused;
- vectorized scores equal the scalar scores, including empty and uneven lists;
- recognition and resolution status checks use one bulk query and preserve
  requested order;
- recognition and resolution batches commit once and roll back on failure;
- existing service behavior, prediction ordering, and resolver thresholds stay
  unchanged.

The focused unit tests will run after each RED/GREEN cycle, followed by the
full repository quality suite and a final diff/status check.

## Non-goals

- no change to model checkpoints, thresholds, gazetteer ranking, or benchmark
  scoring;
- no persistent cross-process embedding cache;
- no change to the public database schema;
- no bypass of the project/service layer in the benchmark.
