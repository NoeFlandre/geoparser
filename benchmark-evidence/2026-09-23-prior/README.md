# Prior-aware resolver run (OAR job 6937696, commit 43967dc)

The run used all 9 corpora with the upstream pipeline, the hybrid pipeline, and the prior pipeline at one commit. The threshold was 0.0. It ran on one Tesla T4 on grue (Nancy).

`prior` is `hybrid` with `PriorResolver`. It uses the same GLiNER2 recognizer and the same MiniLM encoder. It adds an inflection fallback for exact gazetteer misses and a small log-population prior. Thus its recognition is the same as the recognition of hybrid. Only the resolution is different.

Later commits on the branch refactor `PriorResolver` for the CRAP gate and the type gate. They do not change its behavior.
