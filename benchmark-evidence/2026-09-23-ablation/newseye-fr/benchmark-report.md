# newseye-fr benchmark

- Documents: 33
- Gold toponyms: 665
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | cuda | 0.686 | 990.2 | 17.5 | 0.332 | 31.2 |
| trim | cuda | 0.669 | 960.6 | 22.6 | 0.339 | 30.5 |
| population | cuda | 0.705 | 917.4 | 10.2 | 0.321 | 31.3 |
| prior | cuda | 0.689 | 888.1 | 16.9 | 0.329 | 30.4 |
| population-0.05 | cuda | 0.701 | 918.4 | 10.2 | 0.322 | 31.2 |
| population-0.2 | cuda | 0.717 | 914.2 | 5.9 | 0.316 | 31.1 |

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
