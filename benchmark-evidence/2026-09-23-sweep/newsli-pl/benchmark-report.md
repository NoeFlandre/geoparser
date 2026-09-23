# newsli-pl benchmark

- Documents: 186
- Gold toponyms: 196
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | cuda | 0.648 | 700.3 | 8.7 | 0.322 | 13.5 |
| population-0.3 | cuda | 0.679 | 706.4 | 2.9 | 0.301 | 13.5 |
| population-0.5 | cuda | 0.679 | 701.6 | 2.9 | 0.300 | 13.6 |
| population-1.0 | cuda | 0.668 | 729.0 | 6.1 | 0.310 | 13.5 |

## Models

- **hybrid**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **population-0.3**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **population-0.5**: resolver=dguzh/geo-all-MiniLM-L6-v2
- **population-1.0**: resolver=dguzh/geo-all-MiniLM-L6-v2

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
