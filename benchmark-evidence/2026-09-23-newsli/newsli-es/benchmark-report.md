# newsli-es benchmark

- Documents: 500
- Gold toponyms: 1270
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| upstream | cuda | 0.105 | 0.355 | 0.162 |
| hybrid | cuda | 0.204 | 0.860 | 0.330 |
| prior | cuda | 0.204 | 0.860 | 0.330 |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| upstream | cuda | 0.802 | 493.6 | 10.2 | 0.296 | 68.9 |
| hybrid | cuda | 0.802 | 493.6 | 10.2 | 0.296 | 96.1 |
| prior | cuda | 0.820 | 403.0 | 10.2 | 0.283 | 95.1 |

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
