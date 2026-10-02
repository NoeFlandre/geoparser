# Offline model pilot

The repository includes a small real-model pilot for Andorra. It checks the full recognition and resolution path. It does not bundle checkpoints. It uses the multilingual recognizer `fastino/gliner2.5-multi-v1`, the embeddings `jinaai/jina-embeddings-v5-text-small` of Jina, and the reranker `jinaai/jina-reranker-v3.5`. It runs against the Andorra gazetteer fixture with 3,268 features.

Put the checkpoints in an existing Hugging Face cache. Then run this command:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/pilot.py \
  --config tests/fixtures/gazetteer/andorranames.yaml \
  --output-dir pilot-results \
  --hf-home /path/to/hf-cache --offline
```

The command writes `pilot-report.json` and `pilot-report.md` to the output directory. It also writes the generated gazetteer and a temporary SQLite database there. The JSON file keeps every observed span, identifier, model name, and timing. The Markdown file is a compact summary that a person can read.

If the system volume is small, put the Hugging Face cache, the output directory, and the temporary directory on a larger volume. The pilot recognizes all documents in one batch. It releases GLiNER before it loads Jina. It limits the context window of the short documents to 128 tokens. It converts only the CPU reranker to float32. These choices decrease the peak local resource pressure. They do not replace real inference with mocks.

## How the aggregate is counted

The aggregate metrics compare annotations that carry the document that they came from. Thus the same offsets in two sentences stay two annotations. The reported gold counts and predicted counts are the denominators that the metrics use. In the committed run, the resolution accuracy of 0.067 is one correct identifier out of the fifteen gold annotations. It is not out of a smaller set that collisions between documents merged.

## Reading the resolution number

The pilot reports the resolution accuracy with exact identifiers. In the committed run this number is low (1 of 15 gold annotations). Read it with the table for each document. Do not read it alone. A choice of granularity causes this number. The resolver does not find the wrong place.

GeoNames stores an Andorran parish twice. One record is the populated place. The other record is the first-order administrative division with the same name. The committed run has seven resolved mismatches. Five of them are exactly this pair:

| Gold | Predicted |
| --- | --- |
| `3041563` Andorra la Vella (PPLC) | `3041566` Andorra la Vella (ADM1) |
| `3041204` Canillo (PPLA) | `3041203` Canillo (ADM1) |
| `3039163` Sant Julià de Lòria (PPLA) | `3039162` Sant Julià de Lòria (ADM1) |
| `3040686` Encamp (PPLA) | `3040684` Encamp (ADM1) |
| `3338529` Escaldes-Engordany (ADM1) | `3040051` les Escaldes (PPLA) |

A hit for the same place with a different granularity can count as correct. Then the accuracy of this run increases from 0.067 to 0.400. Only one prediction is a really different place. In the Spanish sentence, `Andorra` resolves to `3039328` Radio Andorra (a radio station). The correct place is `3041565` Principality of Andorra.

Keep both numbers. The exact accuracy is the strict metric that is honest. The report computes it. The breakdown above prevents the wrong conclusion that the resolver does not work. The remaining gap in this run is four spans that the recognizer found and the resolver did not resolve. Three spans were never recognized.

Neither figure is a benchmark. Fifteen annotations in thirteen sentences are a smoke test for the wiring. They are not a measurement of the model quality.
