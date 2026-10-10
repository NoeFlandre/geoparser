"""Offline contracts for comparing rerankers on identical frozen candidate lists.

Nothing in this package loads a checkpoint, downloads a file or runs inference.
Model adapters belong in a separate step that must pass the review gate in
:mod:`scripts.reranker_isolation.pins` first.
"""
