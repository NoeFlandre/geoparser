# newsli-uk benchmark

- Documents: 164
- Gold toponyms: 258
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | cuda | 0.643 | 670.4 | 42.7 | 0.360 | 33.5 |
| trim | cuda | 0.589 | 1217.0 | 51.7 | 0.393 | 20.2 |
| population | cuda | 0.643 | 698.5 | 42.7 | 0.361 | 33.3 |
| prior | cuda | 0.589 | 1234.4 | 51.7 | 0.393 | 20.0 |
| population-0.05 | cuda | 0.643 | 675.0 | 42.7 | 0.361 | 33.2 |
| population-0.2 | cuda | 0.647 | 704.3 | 38.7 | 0.358 | 33.4 |

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
