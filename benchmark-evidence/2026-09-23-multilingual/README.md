# Multilingual run (OAR job 6937547, commit 1115f8f)

Complete: all 9 corpora, upstream and hybrid, threshold 0.0, one Tesla T4 on
grue-4 (Nancy). The logs of the two earlier attempts are kept in `logs/`:
job 6937505 (20k-character windows) and job 6937525 (10k-character windows)
both ran out of GPU memory on NewsEye OCR, which led to `1115f8f`, where
GLiNER2's long-document mode handles texts over 10k characters.
