# Prior-aware resolver run (OAR job 6937696, commit 43967dc)

All 9 corpora with upstream, hybrid and prior at one commit, threshold 0.0,
on one Tesla T4 on grue (Nancy). `prior` is `hybrid` with `PriorResolver`:
the same GLiNER2 recognizer and MiniLM encoder, plus an inflection fallback
for exact gazetteer misses and a small log-population prior. Its recognition
is therefore identical to hybrid's; only resolution differs.

Later commits on the branch refactor PriorResolver for the CRAP and type
gates without changing its behaviour.
