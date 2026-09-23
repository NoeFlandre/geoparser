# newsli-sr benchmark

- Documents: 500
- Gold toponyms: 985
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| upstream | cuda | 0.018 | 0.032 | 0.023 |
| hybrid | cuda | 0.192 | 0.784 | 0.309 |
| prior | cuda | 0.192 | 0.784 | 0.309 |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| upstream | cuda | 0.854 | 422.2 | 16.9 | 0.302 | 45.3 |
| hybrid | cuda | 0.854 | 422.2 | 16.9 | 0.302 | 75.0 |
| prior | cuda | 0.832 | 573.7 | 20.7 | 0.311 | 63.3 |

## Models

- **upstream**: recognizer=en_core_web_sm, resolver=dguzh/geo-all-MiniLM-L6-v2
- **hybrid**: recognizer=fastino/gliner2.5-multi-v1, resolver=dguzh/geo-all-MiniLM-L6-v2
- **prior**: recognizer=fastino/gliner2.5-multi-v1, resolver=dguzh/geo-all-MiniLM-L6-v2

Resolution is scored on the gold spans, supplied to both pipelines
through ManualRecognizer, so the resolvers are judged on the same
toponyms and a difference cannot be an artefact of recognition.

A gold toponym a pipeline did not place is charged the maximum
possible error rather than dropped, so resolving less cannot improve
a score. The shared abstention threshold is stated above. Models
can have different similarity scales, so the threshold is part of
the experiment and should be considered when comparing resolvers.

AUC is a log-scaled summary of the whole error distribution, lower
being better. Its normalization is this harness's own, so compare
runs of this harness rather than published AUC figures.
