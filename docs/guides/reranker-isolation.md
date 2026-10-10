# Reranker isolation on frozen candidates

This guide describes the offline contract for issue #164. It compares no reranker, Qwen3-Reranker-0.6B and jinaai/jina-reranker-v3.5 on identical frozen candidate lists. The code lives in `scripts/reranker_isolation`. It does not download a checkpoint, load a model, run inference or read a dataset. Its tests use synthetic inputs and hand-counted expected values.

Run the tests with:

```bash
uv run pytest tests/unit/test_reranker_isolation
```

## Pinned checkpoints

| Checkpoint | Revision | Licence | Scoring | Custom code |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | apache-2.0 | pointwise | none |
| `jinaai/jina-reranker-v3.5` | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | cc-by-nc-4.0 | listwise | `modeling.py`, SHA-256 `ec6612461b4307eb3bab089e6916c00881b7e6b2e8edfbd56a9ace4560244837` |

The revisions and digests were read from the Hugging Face metadata on 2026-10-09. The Jina digest was recomputed from the pinned file. Both pins are constants in `scripts/reranker_isolation/pins.py`.

The Jina licence is non-commercial. Check it before publishing results or redistributing code or weights.

## Review gate for custom code

`remote_code_permitted` decides whether custom code may run. A checkpoint without custom code returns False. For a checkpoint with custom code, the caller must fetch the bytes of the pinned file and pass them in. The function refuses a missing or altered file with `UnreviewedCodeError`. An adapter that loads a checkpoint with `trust_remote_code=True` must call this gate first.

The `modeling.py` file at the pinned revision was read in full for this issue. Findings:

- It imports numpy, torch and transformers. It makes no network call, starts no subprocess and writes no file.
- `rerank` splits one call into blocks. A block closes after 125 documents, or when the remaining token capacity falls to 8192 or below. The blocks are scored separately, and the query embeddings are averaged across blocks. Shortlists that fit in one block are scored in one shared context. `jina_blocks` predicts the split from document token lengths, and the report should flag any reference that is split.
- `rerank` orders results with `np.argsort`, which is not stable. The harness therefore ranks from the returned `relevance_score` values with its own tie rule.
- `rerank` has no handling for an empty document list, so the harness never calls it with one.
- The tokenizer is loaded again with `trust_remote_code=True` from `name_or_path`. This is a second remote-code path. Its files sit at the pinned commit, but their digests are neither recorded nor verified.
- Each forward pass sets `lm_head` to `Identity`. Scoring is unaffected, but a model object should not be reused for other tasks without reloading.

This review was done by the implementing agent. It is not an independent review.

## Scoring semantics

Higher is better for both rerankers.

- **Qwen3-Reranker.** Each (query, document) pair is formatted with the documented template and scored on its own. The score is P(yes), the softmax over the "no" and "yes" logits at the last position. `qwen_yes_probability` computes it in a form that does not overflow. `assemble_qwen_ids` keeps the prefix and suffix whole and cuts the pair from the end to the budget of 8192 tokens minus the prefix and suffix.
- **Jina reranker v3.5.** The whole shortlist goes into one listwise call. Each result has an `index` and a `relevance_score`, which is a cosine similarity. `listwise_scores` aligns results to shortlist positions, and it rejects a missing, repeated, out-of-range or non-finite value. A query is truncated to 1024 tokens and each document to 8192 tokens.

A malformed output is never turned into a score. `resolve` classifies it as `invalid`. An empty shortlist abstains without calling the scorer.

## Candidates, ties and identifiers

Each reference is a `FrozenItem` with a context, its candidates in first-stage order and a canonical gold ID. The shortlist takes the first K candidates. Its size is a parameter of the policy, not a value chosen here. The repository's existing Jina resolver uses 20 by default. Choosing the value for this study is a decision for the owner.

`rank_positions` sorts by score, highest first. Equal scores keep the shortlist order. Each position maps to one candidate identifier. The recall ceiling counts the eligible references, meaning those with a canonical gold ID, whose gold appears in the shortlist. A gold absent from the frozen list is a miss.

## Decisions and abstention

`decide` returns one of three statuses:

- `resolved` with the top candidate's identifier, when the top score reaches the threshold. The threshold is inclusive.
- `abstained`, when the shortlist is empty or the top score is below the threshold.
- `invalid`, when the scorer produced unusable output or failed at runtime.

Abstention and invalid output both count as misses. Invalid output is reported separately so that it can be inspected.

## Thresholds and freezing

Thresholds are chosen on development data only. `select_threshold` maximises a utility that adds one per correct resolution and subtracts a penalty per wrong one. Abstentions cost nothing. Ties go to the lowest threshold. The penalty is an explicit argument with no default. It must be chosen before any test outcome is inspected, because with a penalty of zero the best threshold is always the lowest one.

`canonical_digest` hashes a decision as sorted, compact JSON. Store the digest with the development decision. Before the final evaluation, call `verify_digest` on the stored decision, so that a changed decision is refused.

## Reporting

`summarize` compares each system with the baseline on the same example IDs. The baseline is no reranker, which is the first-stage top candidate. The report contains:

- Counts of resolved, abstained, invalid and correct outcomes, and accuracy.
- Gain, which is the accuracy difference from the baseline.
- A 95 percent interval for the gain from a paired document bootstrap. Each draw uses the same example indices for both systems, and the interval is taken from the per-draw differences.
- Fixes, which are examples the baseline got wrong and the system got right, and regressions, which are the reverse.
- Mean steady inference seconds per example.
- Load, fetch, warmup and steady times, kept apart in `Timing`.
- Peak memory relative to the baseline. An unavailable reading stays unavailable and is never reported as zero.

The bootstrap interval is a percentile interval from `percentile_interval`. It uses the same inverse-CDF rule as the public benchmark protocol, with at least 1000 resamples in a real run.

## Not included

This change does not include the following. Each one is needed before any result can be reported.

- Model adapters that load the checkpoints and call them. Adapters must pass the review gate first.
- Frozen candidate lists built from a public dataset, and the gazetteer snapshot they refer to.
- Runs of any kind, including timing and memory measurements.
- The shortlist size, the threshold grid and the wrong-penalty value for the study. These need owner decisions.
- Pins for the Jina tokenizer and any other files loaded with remote code.
- An independent code review, and the mutation and CRAP gates on a full run.
- A command-line runner.
