# Multilingual run, partial (OAR job 6937476, commit 119f5ad)

This job crashed with CUDA out of memory on newseye-de, when GLiNER2 received
a 133k-character page whole. The four corpora below completed before the
crash and are kept as they were written. The windowing fix is `213883d`, and
the full rerun is job 6937505.

Completed: geovirus, hipe2020-de, hipe2020-fr, hipe2020-en.
Not completed: newseye-de (crashed), newseye-fr, newseye-fi, newseye-sv, topres19th-en.
