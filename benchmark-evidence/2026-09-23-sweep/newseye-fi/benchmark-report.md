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
| hybrid | cuda | 0.313 | 3599.4 | 1288.4 | 0.625 | 11.0 |
| population-0.3 | cuda | 0.360 | 3333.4 | 1011.9 | 0.597 | 10.9 |
| population-0.5 | cuda | 0.364 | 3340.7 | 1054.2 | 0.601 | 10.9 |
| population-1.0 | cuda | 0.369 | 3321.9 | 1011.9 | 0.599 | 10.9 |

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
