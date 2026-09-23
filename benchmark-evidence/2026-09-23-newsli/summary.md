# Benchmark summary

Recognition is exact-match F1 on each pipeline's own spans; resolution
is scored on the gold spans. Each corpus's own report has the detail.

| Corpus | Lang | Docs | Gold | Pipeline | Rec F1 | Acc@161km | AUC | Seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| newsli-ar | ar | 500 | 3319 | upstream | 0.005 | 0.852 | 0.251 | 100.8 |
| newsli-ar | ar | 500 | 3319 | hybrid | 0.358 | 0.852 | 0.251 | 136.3 |
| newsli-ar | ar | 500 | 3319 | prior | 0.358 | 0.856 | 0.249 | 127.0 |
| newsli-de | de | 500 | 1019 | upstream | 0.133 | 0.904 | 0.192 | 31.7 |
| newsli-de | de | 500 | 1019 | hybrid | 0.406 | 0.904 | 0.192 | 56.5 |
| newsli-de | de | 500 | 1019 | prior | 0.406 | 0.909 | 0.189 | 56.6 |
| newsli-es | es | 500 | 1270 | upstream | 0.162 | 0.802 | 0.296 | 68.9 |
| newsli-es | es | 500 | 1270 | hybrid | 0.330 | 0.802 | 0.296 | 96.1 |
| newsli-es | es | 500 | 1270 | prior | 0.330 | 0.820 | 0.283 | 95.1 |
| newsli-fa | fa | 71 | 326 | upstream | 0.015 | 0.732 | 0.376 | 9.2 |
| newsli-fa | fa | 71 | 326 | hybrid | 0.321 | 0.732 | 0.376 | 19.0 |
| newsli-fa | fa | 71 | 326 | prior | 0.321 | 0.709 | 0.387 | 18.3 |
| newsli-ja | ja | 500 | 1577 | upstream | 0.000 | 0.847 | 0.226 | 22.4 |
| newsli-ja | ja | 500 | 1577 | hybrid | 0.081 | 0.847 | 0.226 | 55.3 |
| newsli-ja | ja | 500 | 1577 | prior | 0.081 | 0.849 | 0.225 | 54.6 |
| newsli-pl | pl | 186 | 196 | upstream | 0.057 | 0.648 | 0.322 | 19.6 |
| newsli-pl | pl | 186 | 196 | hybrid | 0.132 | 0.648 | 0.322 | 40.6 |
| newsli-pl | pl | 186 | 196 | prior | 0.132 | 0.673 | 0.309 | 37.9 |
| newsli-ro | ro | 226 | 385 | upstream | 0.128 | 0.810 | 0.278 | 19.6 |
| newsli-ro | ro | 226 | 385 | hybrid | 0.352 | 0.810 | 0.278 | 36.0 |
| newsli-ro | ro | 226 | 385 | prior | 0.352 | 0.826 | 0.273 | 34.9 |
| newsli-sr | sr | 500 | 985 | upstream | 0.023 | 0.854 | 0.302 | 45.3 |
| newsli-sr | sr | 500 | 985 | hybrid | 0.309 | 0.854 | 0.302 | 75.0 |
| newsli-sr | sr | 500 | 985 | prior | 0.309 | 0.832 | 0.311 | 63.3 |
| newsli-ta | ta | 500 | 1578 | upstream | 0.010 | 0.507 | 0.515 | 27.5 |
| newsli-ta | ta | 500 | 1578 | hybrid | 0.016 | 0.507 | 0.515 | 238.4 |
| newsli-ta | ta | 500 | 1578 | prior | 0.016 | 0.512 | 0.508 | 235.0 |
| newsli-tr | tr | 500 | 1578 | upstream | 0.121 | 0.709 | 0.364 | 40.0 |
| newsli-tr | tr | 500 | 1578 | hybrid | 0.328 | 0.709 | 0.364 | 67.8 |
| newsli-tr | tr | 500 | 1578 | prior | 0.328 | 0.776 | 0.323 | 65.2 |
| newsli-uk | uk | 164 | 258 | upstream | 0.055 | 0.643 | 0.360 | 39.7 |
| newsli-uk | uk | 164 | 258 | hybrid | 0.160 | 0.643 | 0.360 | 59.7 |
| newsli-uk | uk | 164 | 258 | prior | 0.160 | 0.589 | 0.393 | 46.3 |
