# Hybrid Resolver Benchmark Design

## Context

The completed GeoVirus run used the same gold spans for both resolvers and therefore isolates resolution quality. The upstream spaCy + `dguzh/geo-all-MiniLM-L6-v2` pipeline scored 0.834 accuracy within 161 km, 262.7 km mean error, and 82.3 seconds. The swapped GLiNER2 + Jina embedding/reranker pipeline scored 0.657, 1138.0 km, and 2439.8 seconds. Recognition improved slightly in the swapped pipeline, so the resolver/reranker is the failing replacement, not GLiNER2 recognition.

## Decision

Keep the existing `upstream` pipeline unchanged and add a distinct `hybrid` pipeline:

- recognition: `fastino/gliner2.5-multi-v1` through `GLiNER2Recognizer`;
- resolution: `SentenceTransformerResolver` with `dguzh/geo-all-MiniLM-L6-v2`;
- gazetteer: `geonames`;
- benchmark threshold: `0.0`, matching the previous comparison;
- device placement: the same explicit CUDA/CPU placement used by the current harness.

The old `swapped` pipeline and its report remain untouched as historical evidence. A fresh run will request only `upstream` and `hybrid` and write to a new results directory, so no checkpoint or report can mix Jina and MiniLM predictions.

## Data flow and implementation

The benchmark pipeline factory gains the `hybrid` name and returns GLiNER2 for recognition and the existing SentenceTransformer resolver for resolution. The shared runner, checkpoint identity, scoring, and report format remain unchanged; model provenance in the report identifies the replacement explicitly.

Unit tests will pin the new factory selection and device movement without loading real models. Existing benchmark tests and the resolver/service tests remain the regression suite. The remote command will use the same GeoVirus digest, GeoNames artifact, CUDA device, chunked checkpoints, and Grid’5000 safeguards as the completed run.

## Success criteria

1. The local test and static checks pass.
2. The branch is clean and the runnable commit is available to Nancy.
3. The new Grid’5000 run completes or leaves valid resumable checkpoints.
4. The final report compares upstream and hybrid on all 229 documents and 2167 gold toponyms, with model names, elapsed time, and resolution metrics preserved.
5. The remote report and checkpoints are copied off Nancy before the handoff.
