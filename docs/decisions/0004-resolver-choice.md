# ADR 0004: Resolver choice for the default pipeline

## Status

Accepted — 2026-09-24

## Context

The historical GeoVirus comparison used the same gold spans for every resolver. MiniLM achieved 0.834 accuracy within 161 km in about 82 seconds; the Jina embedding and reranking pipeline achieved 0.657 in about 2,440 seconds. The broader multilingual sweep then compared the MiniLM baseline with a population prior. With weight `0.3` and inflection fallback off, GeoVirus accuracy within 161 km was 0.846, compared with 0.834 for the same MiniLM resolver without the prior.

The sweep evidence is stored in `benchmark-evidence/2026-09-23-sweep/` and `benchmark-evidence/2026-09-23-ablation/`. The [benchmark results dataset](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) publishes the reports and score tables.

## Decision

Use `PriorResolver` in the README and quickstart examples. Its default population weight is `0.3`, and inflection fallback is disabled. The resolver keeps the upstream MiniLM encoder and uses population only to rank candidates whose raw context similarity already passes the configured threshold.

Keep `SentenceTransformerResolver` as the simpler baseline and preserve `JinaResolver` as a selectable option for comparison. The benchmark guide explains the pipelines, ablations, and corpus-specific results.

## Consequences

The recommended resolver uses the same model family as the existing MiniLM baseline and adds a small ranking adjustment without another model. Population data may be incomplete or uneven across gazetteers, so users should validate the resolver on their corpus and can set `population_weight=0` to disable the prior. The benchmark is evidence for the tested models, corpora, and GeoNames artifact; it does not guarantee the same ranking for every domain.
