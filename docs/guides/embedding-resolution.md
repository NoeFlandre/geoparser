# Embedding resolution comparison

This guide describes the offline half of issue #163. It compares the retained MiniLM and Jina v5 embeddings with Qwen3-Embedding-0.6B, Qwen3-Embedding-4B and BGE-M3 on gold-span resolution, under one frozen retrieval setup. The code pins, adapts and scores; it does not download a model, run inference or publish a result. Those steps wait for a separate execution approval.

Run the offline commands from the repository root:

```bash
uv run python -m scripts.embedding_resolution registry
uv run python -m scripts.embedding_resolution freeze PLAN.json --output EXPERIMENT.json
```

`registry` prints every pin, prompt, historical setting and scoring constant. `freeze` validates a plan and writes the planned inventory only when it is valid. Exit status 2 means the plan was refused, and nothing is written.

## Pinned models

Each revision is a full commit hash that the Hugging Face model API reported on 2026-10-09. Prompts, pooling, normalization and dimensions come from the sentence-transformers configuration at that commit. A pin names an identity; it does not check the bytes of a cached copy.

| Key | Repository | Status | Dimension | Pooling | Prompts (query / document) | Custom code | Licence |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| `geo-minilm` | `dguzh/geo-all-MiniLM-L6-v2` | retained | 384 | mean | none / none | no | not declared |
| `jina-v5-text-small` | `jinaai/jina-embeddings-v5-text-small` | retained | 1024 | last token | `Query: ` / `Document: ` with task `retrieval` | yes | CC BY-NC 4.0 |
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | added | 1024 | last token | instruction / none | no | Apache-2.0 |
| `qwen3-embedding-4b` | `Qwen/Qwen3-Embedding-4B` | added | 2560 | last token | instruction / none | no | Apache-2.0 |
| `bge-m3` | `BAAI/bge-m3` | added | 1024 | CLS | none / none | no | MIT |

The Qwen instruction is `Instruct: Given a web search query, retrieve relevant passages that answer the query` followed by a newline and `Query:`. Only the query side receives it. All five models declare L2 normalization of their outputs. The adapter rescales each row to unit length. It does not check the raw output for normalization.

Jina v5 loads its modelling code from the repository (`trust_remote_code`). The freeze refuses to name that model until a reviewed code artifact is supplied for it. The review itself is outstanding; see the gaps below.

## Adapters

`EmbeddingAdapter` wraps an encoder and applies the model's prompt to each role. It splits inputs into batches of the configured size and checks each batch: the output must have one row per string and the pinned dimension, every value must be finite, and no row may be zero. Rows are then normalized when the model declares it. An empty input returns zero rows without calling the encoder. Non-string inputs are refused before any encoding.

`SentenceTransformerEncoder` is the only code that loads a checkpoint. It passes the pinned revision, the remote-code flag, the maximum sequence length, the documented prompt and the documented task. Importing this module does not import torch.

## Scoring

Two policies are compared, and both use the same candidates and the same contexts:

- `similarity`: the candidate with the highest cosine similarity wins.
- `population`: the candidate with the highest similarity plus `0.3 * log10(1 + population) / 10` wins. The weight is the one `PriorResolver` uses.

The threshold applies to the raw cosine similarity of the chosen candidate, as it does in the library. A candidate below the threshold causes an abstention. Ties go to the earlier candidate in gazetteer order.

The population-only baseline ignores the context. It chooses the most populous retrieved candidate and abstains only when retrieval returned nothing.

Each gold span is counted once as resolved, abstained or invalid, so the three always sum to the number of gold spans. Invalid output is a chosen candidate that was not retrieved. Exact-ID accuracy, distance bands at 1, 10 and 50 km, and candidate recall each have their own eligible denominator, and each is recorded beside its count. A missing gold target counts as a miss, even when retrieval failed. Distances use the library's great-circle function.

## Thresholds

Thresholds are calibrated per model and per policy on development spans only. Each span gives one observation: the chosen candidate's similarity and whether it is the gold canonical ID. For each threshold in a grid of 201 values from -1 to 1 in steps of 0.01, the objective is the number of correct resolutions minus the number of incorrect ones. An abstention scores zero. The lowest threshold among the maxima is kept, which preserves coverage.

Cosine scales differ between models, so a threshold is never copied between them. The 0.6 constructor default of the resolver and the 0.0 default of the benchmark CLI were never calibrated. They are kept only as labelled historical settings for MiniLM, the encoder the 2026-09-23 evidence runs used. A historical value for any other model is refused by the freeze.

## Freeze plan

A freeze plan holds only what an approved run can supply:

- the shared protocol block, with task `gold_span_resolution`, split `development` and stage `screening`;
- the gazetteer artifact, whose identifier must have an attribute map for candidate descriptions;
- the context budget: a token limit and the tokenizer artifact that counts it;
- one weight artifact per embedding model, whose identifier and revision must match the registry pin;
- reviewed custom code for each model that loads it, at the registry pin;
- source slices, each with an example count, a gold-span count and a sample digest;
- threshold records: a calibrated value with its calibration digest, a labelled historical value, or the structural baseline;
- the batch size, the seed, and the provenance note.

A synthetic example with fixture digests is in `docs/examples/embedding-resolution-plan.json`. `freeze` accepts it and writes nothing unless `--output` is given. Its digests are not real, so do not use it as an experiment.

Every threshold record becomes one pipeline, and every pipeline covers every source slice. The shared validator in `scripts.benchmark_protocol` then checks the whole inventory. Every configuration starts as `planned`, with no provenance digest, measurements, scores or raw predictions.

## What is not done

- No model is downloaded or loaded, and no inference runs.
- No dataset, gazetteer snapshot or context is built. The plan's sample digests and gazetteer digest are placeholders until an approved run supplies them.
- No threshold has been calibrated. Calibration needs development predictions from an approved run.
- No reranking is added. This comparison is retrieval plus the optional population prior.
- The Jina v5 modelling code (`modeling_jina_embeddings_v5.py` and `configuration_jina_embeddings_v5.py`) has not been reviewed. `custom_st.py` was read: it normalizes its output and applies the adapter named by the task. The review must cover the other two files and be pinned before execution.
- The weight digests (SHA-256 of the exact bytes) are not known. The plan requires them, so the freeze cannot run until they are computed from the downloaded files.
- The GeoNames gazetteer is a live download without a pinned snapshot. It needs a snapshot identifier and digest before the freeze.
- Language support and training overlap are unknown for every model. The plan defaults them to `unspecified` and `unknown`.
- The context budget is a decision for the owner. It must be the same for every model, either by counting one shared tokenizer or by giving each model its own limit. The plan records whichever is chosen, and nothing in the code chooses for it.
- Latency and peak memory are measured by `scripts.embedding_resolution.measurement`, which is not yet called by any run. A zero reading is never recorded as a measurement.
