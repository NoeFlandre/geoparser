# GeoVirus benchmark

- Documents: 229
- Gold toponyms: 2167
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| upstream | cuda | 0.837 | 0.784 | 0.810 |
| swapped | cuda | 0.826 | 0.832 | 0.829 |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| upstream | cuda | 0.834 | 262.7 | 30.5 | 0.300 | 82.3 |
| swapped | cuda | 0.657 | 1138.0 | 75.9 | 0.403 | 2439.8 |

## Models

- **upstream**: recognizer=en_core_web_sm, resolver=dguzh/geo-all-MiniLM-L6-v2
- **swapped**: recognizer=fastino/gliner2.5-multi-v1, resolver=jinaai/jina-embeddings-v5-text-small, reranker=jinaai/jina-reranker-v3.5

Resolution is scored on the gold spans, supplied to both pipelines
through ManualRecognizer, so the resolvers are judged on the same
toponyms and a difference cannot be an artefact of recognition.

A gold toponym a pipeline did not place is charged the maximum
possible error rather than dropped, so resolving less cannot improve
a score. The abstention threshold is shared and stated above: the
library default of 0.6 is calibrated for the upstream model's
similarity scale, and at that value Jina abstains on every toponym.

AUC is a log-scaled summary of the whole error distribution, lower
being better. Its normalization is this harness's own, so compare
runs of this harness rather than published AUC figures.
