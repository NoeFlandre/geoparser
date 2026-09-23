# Benchmark summary

Recognition is exact-match F1 on each pipeline's own spans; resolution
is scored on the gold spans. Each corpus's own report has the detail.

| Corpus | Lang | Docs | Gold | Pipeline | Rec F1 | Acc@161km | AUC | Seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| geovirus | en | 229 | 2167 | upstream | 0.810 | 0.834 | 0.300 | 60.7 |
| geovirus | en | 229 | 2167 | hybrid | 0.829 | 0.834 | 0.300 | 57.5 |
| hipe2020-de | de | 48 | 558 | upstream | 0.247 | 0.640 | 0.371 | 20.3 |
| hipe2020-de | de | 48 | 558 | hybrid | 0.599 | 0.640 | 0.371 | 34.8 |
| hipe2020-fr | fr | 43 | 800 | upstream | 0.198 | 0.766 | 0.294 | 42.4 |
| hipe2020-fr | fr | 43 | 800 | hybrid | 0.695 | 0.766 | 0.294 | 60.3 |
| hipe2020-en | en | 38 | 158 | upstream | 0.497 | 0.620 | 0.400 | 10.8 |
| hipe2020-en | en | 38 | 158 | hybrid | 0.632 | 0.620 | 0.400 | 20.1 |
| newseye-de | de | 8 | 628 | upstream | 0.105 | 0.646 | 0.321 | 22.5 |
| newseye-de | de | 8 | 628 | hybrid | 0.283 | 0.646 | 0.321 | 46.0 |
| newseye-fr | fr | 33 | 665 | upstream | 0.183 | 0.686 | 0.332 | 37.9 |
| newseye-fr | fr | 33 | 665 | hybrid | 0.467 | 0.686 | 0.332 | 63.6 |
| newseye-fi | fi | 18 | 214 | upstream | 0.036 | 0.313 | 0.625 | 12.6 |
| newseye-fi | fi | 18 | 214 | hybrid | 0.463 | 0.313 | 0.625 | 25.2 |
| newseye-sv | sv | 18 | 265 | upstream | 0.197 | 0.555 | 0.386 | 11.2 |
| newseye-sv | sv | 18 | 265 | hybrid | 0.520 | 0.555 | 0.386 | 23.4 |
| topres19th-en | en | 110 | 880 | upstream | 0.492 | 0.724 | 0.303 | 26.6 |
| topres19th-en | en | 110 | 880 | hybrid | 0.624 | 0.724 | 0.303 | 39.5 |
