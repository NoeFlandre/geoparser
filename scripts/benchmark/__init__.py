"""
The GeoVirus benchmark harness.

Split so that the parts which must be correct -- corpus parsing, checkpoint
handling and scoring -- can be tested without models, a gazetteer or a GPU.
Only ``pipelines`` and ``runner`` need any of those.
"""
