# Multilingual run (OAR job 6937547, commit 1115f8f)

The run is complete. It used all 9 corpora with the upstream pipeline and the hybrid pipeline. The threshold was 0.0. It ran on one Tesla T4 on grue-4 (Nancy).

The logs of two earlier attempts are in `logs/`. Job 6937505 used windows of 20k characters. Job 6937525 used windows of 10k characters. Both jobs ran out of GPU memory on NewsEye OCR text. This led to commit `1115f8f`. In that commit, the long-document mode of GLiNER2 handles texts of more than 10k characters.
