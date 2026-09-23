# Benchmark summary

Recognition is exact-match F1 on each pipeline's own spans; resolution
is scored on the gold spans. Each corpus's own report has the detail.

| Corpus | Lang | Docs | Gold | Pipeline | Rec F1 | Acc@161km | AUC | Seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| geovirus | en | 229 | 2167 | upstream | 0.810 | 0.834 | 0.300 | 41.5 |
| geovirus | en | 229 | 2167 | hybrid | 0.829 | 0.834 | 0.300 | 55.2 |
| hipe2020-de | de | 48 | 558 | upstream | 0.247 | 0.640 | 0.371 | 20.2 |
| hipe2020-de | de | 48 | 558 | hybrid | 0.607 | 0.640 | 0.371 | 37.6 |
| hipe2020-fr | fr | 43 | 800 | upstream | 0.198 | 0.766 | 0.294 | 42.2 |
| hipe2020-fr | fr | 43 | 800 | hybrid | 0.668 | 0.766 | 0.294 | 63.6 |
| hipe2020-en | en | 38 | 158 | upstream | 0.497 | 0.620 | 0.400 | 10.5 |
| hipe2020-en | en | 38 | 158 | hybrid | 0.632 | 0.620 | 0.400 | 20.0 |
