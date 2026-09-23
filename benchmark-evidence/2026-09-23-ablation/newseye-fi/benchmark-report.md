# newseye-fi benchmark

- Documents: 18
- Gold toponyms: 214
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | cuda | 0.313 | 3599.4 | 1288.4 | 0.625 | 10.7 |
| trim | cuda | 0.388 | 2980.3 | 768.8 | 0.568 | 8.4 |
| population | cuda | 0.346 | 3512.6 | 1114.5 | 0.611 | 11.1 |
| prior | cuda | 0.458 | 2707.9 | 299.7 | 0.517 | 8.4 |
| population-0.05 | cuda | 0.341 | 3442.8 | 1087.7 | 0.612 | 10.7 |
| population-0.2 | cuda | 0.360 | 3355.9 | 1011.9 | 0.599 | 10.8 |

## Models

- **hybrid**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **trim**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **population**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **prior**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **population-0.05**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **population-0.2**: resolver=dguzh/geo-all-MiniLM-L6-v2

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
