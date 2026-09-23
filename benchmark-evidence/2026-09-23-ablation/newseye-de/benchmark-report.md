# newseye-de benchmark

- Documents: 8
- Gold toponyms: 628
- Gazetteer: geonames
- Shared abstention threshold: 0.0

## Recognition (each pipeline's own spans, exact match)

| Pipeline | Device | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |

## Resolution (gold spans supplied, distance scored)

| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | cuda | 0.646 | 705.2 | 5.6 | 0.321 | 14.2 |
| trim | cuda | 0.643 | 686.5 | 5.6 | 0.319 | 13.3 |
| population | cuda | 0.651 | 685.7 | 5.1 | 0.318 | 14.2 |
| prior | cuda | 0.650 | 669.6 | 4.5 | 0.315 | 13.4 |
| population-0.05 | cuda | 0.651 | 689.6 | 5.6 | 0.319 | 14.5 |
| population-0.2 | cuda | 0.664 | 655.1 | 3.2 | 0.310 | 14.2 |

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
