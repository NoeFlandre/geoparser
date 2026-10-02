# ADR 0003: Resource-bounded real-model pilot

## Status

Accepted — 2026-09-14

## Context

The project needs executable evidence from the configured recognizer checkpoints and resolver checkpoints. The checkpoints are large. The Mac has a small system volume. Therefore a pilot must be reproducible and inspectable. It must be safe to run again. It must not copy models into the repository. It must not download them silently.

## Decision

Use a fixed corpus of 13 Andorra documents with hand-written spans. Use the real checkpoints `fastino/gliner2.5-multi-v1`, `jinaai/jina-embeddings-v5-text-small`, and `jinaai/jina-reranker-v3.5`. Use the `andorranames` fixture with 3,268 features. Run the project recognition service and resolution service in two phases. Release GLiNER before you construct Jina. Batch the short inputs. Limit the resolver context to 128 tokens. Convert only the reranker to float32 on CPU. Save the JSON evidence and the Markdown evidence next to a generated gazetteer on the output volume that the caller selects.

## Tradeoffs

The two-phase run takes more time than a run that keeps both model graphs alive. The timing report separates two items. The first item is the recognition and resolution decisions for each document. The second item is the shared batch embedding work. The fixed corpus is not a benchmark of the global geoparser quality. It is a deterministic smoke test. It shows real model behavior and resolution errors.

## Consequences

Model caches and generated evidence stay outside source control. In offline mode, a missing checkpoint fails immediately. It does not start an unbounded download. When you expand the pilot, update its gold spans and its acceptance evidence. Do not change the corpus silently.
