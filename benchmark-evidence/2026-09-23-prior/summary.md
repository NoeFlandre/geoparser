# Benchmark summary

Recognition is exact-match F1 on each pipeline's own spans; resolution
is scored on the gold spans. Each corpus's own report has the detail.

| Corpus | Lang | Docs | Gold | Pipeline | Rec F1 | Acc@161km | AUC | Seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| geovirus | en | 229 | 2167 | upstream | 0.810 | 0.834 | 0.300 | 73.6 |
| geovirus | en | 229 | 2167 | hybrid | 0.829 | 0.834 | 0.300 | 51.3 |
| geovirus | en | 229 | 2167 | prior | 0.829 | 0.837 | 0.298 | 43.2 |
| hipe2020-de | de | 48 | 558 | upstream | 0.247 | 0.640 | 0.371 | 20.4 |
| hipe2020-de | de | 48 | 558 | hybrid | 0.599 | 0.640 | 0.371 | 35.2 |
| hipe2020-de | de | 48 | 558 | prior | 0.599 | 0.661 | 0.364 | 33.6 |
| hipe2020-fr | fr | 43 | 800 | upstream | 0.198 | 0.766 | 0.294 | 42.6 |
| hipe2020-fr | fr | 43 | 800 | hybrid | 0.695 | 0.766 | 0.294 | 60.6 |
| hipe2020-fr | fr | 43 | 800 | prior | 0.695 | 0.765 | 0.299 | 59.3 |
| hipe2020-en | en | 38 | 158 | upstream | 0.497 | 0.620 | 0.400 | 10.6 |
| hipe2020-en | en | 38 | 158 | hybrid | 0.632 | 0.620 | 0.400 | 20.0 |
| hipe2020-en | en | 38 | 158 | prior | 0.632 | 0.633 | 0.395 | 20.0 |
| newseye-de | de | 8 | 628 | upstream | 0.105 | 0.646 | 0.321 | 22.1 |
| newseye-de | de | 8 | 628 | hybrid | 0.283 | 0.646 | 0.321 | 45.6 |
| newseye-de | de | 8 | 628 | prior | 0.283 | 0.650 | 0.315 | 45.1 |
| newseye-fr | fr | 33 | 665 | upstream | 0.183 | 0.686 | 0.332 | 37.2 |
| newseye-fr | fr | 33 | 665 | hybrid | 0.467 | 0.686 | 0.332 | 63.9 |
| newseye-fr | fr | 33 | 665 | prior | 0.467 | 0.689 | 0.329 | 63.1 |
| newseye-fi | fi | 18 | 214 | upstream | 0.036 | 0.313 | 0.625 | 12.7 |
| newseye-fi | fi | 18 | 214 | hybrid | 0.463 | 0.313 | 0.625 | 25.4 |
| newseye-fi | fi | 18 | 214 | prior | 0.463 | 0.458 | 0.517 | 22.8 |
| newseye-sv | sv | 18 | 265 | upstream | 0.197 | 0.555 | 0.386 | 11.2 |
| newseye-sv | sv | 18 | 265 | hybrid | 0.520 | 0.555 | 0.386 | 23.3 |
| newseye-sv | sv | 18 | 265 | prior | 0.520 | 0.619 | 0.336 | 22.2 |
| topres19th-en | en | 110 | 880 | upstream | 0.492 | 0.724 | 0.303 | 26.5 |
| topres19th-en | en | 110 | 880 | hybrid | 0.624 | 0.724 | 0.303 | 39.6 |
| topres19th-en | en | 110 | 880 | prior | 0.624 | 0.719 | 0.305 | 38.5 |
