# ADR 0004: Resolver choice for the default pipeline

## Status

Accepted — 2026-09-24

## Context

The historical GeoVirus comparison used the same gold spans for every resolver. MiniLM achieved 0.834 accuracy within 161 km in about 82 seconds. The Jina embedding and reranking pipeline achieved 0.657 in about 2,440 seconds. Then the broader multilingual sweep compared the MiniLM baseline with a population prior. The weight was `0.3` and the inflection fallback was off. The GeoVirus accuracy within 161 km was 0.846. The same MiniLM resolver without the prior had 0.834.

The sweep evidence is in `benchmark-evidence/2026-09-23-sweep/` and `benchmark-evidence/2026-09-23-ablation/`. The [benchmark results dataset](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) publishes the reports and the score tables.

## Decision

Use `PriorResolver` in the README examples and the quickstart examples. Its default population weight is `0.3`. The inflection fallback is disabled. The resolver keeps the upstream MiniLM encoder. It uses the population only to rank candidates. These candidates already have a raw context similarity that passes the configured threshold.

Keep `SentenceTransformerResolver` as the simpler baseline. Keep `JinaResolver` as an option that you can select for comparison. The benchmark guide explains the pipelines, the ablations, and the results for each corpus.

## Consequences

The recommended resolver uses the same model family as the existing MiniLM baseline. It adds a small ranking adjustment and no other model. The population data can be incomplete or uneven in different gazetteers. Therefore users must validate the resolver on their corpus. They can set `population_weight=0` to disable the prior. The benchmark is evidence for the tested models, the tested corpora, and the tested GeoNames artifact. It does not guarantee the same ranking for every domain.
