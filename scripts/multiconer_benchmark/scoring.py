"""Exact-span recognition counts per sentence, on the shared protocol contract.

The span arithmetic is the PAN-X ``Counts`` accumulator. Invalid outputs are
added to false positives, as the protocol requires, and each distinct raw
malformed output is counted once.
"""

from __future__ import annotations

from collections.abc import Iterable

from scripts.benchmark_protocol.schema import RecognitionCounts, require
from scripts.panx_benchmark.metrics import Counts

Span = tuple[int, int]
_COUNT_FIELDS = (
    "gold_spans",
    "predicted_spans",
    "true_positive",
    "false_positive",
    "false_negative",
    "invalid_outputs",
)


def _as_span(value: object, text_length: int) -> Span | None:
    """Return a valid half-open span, or None for any malformed output."""
    pair = _int_pair(value)
    if pair is None:
        return None
    start, end = pair
    if not 0 <= start < end <= text_length:
        return None
    return (start, end)


def _int_pair(value: object) -> Span | None:
    """Return a two-element tuple or list of exact ints, or None."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    start, end = value
    if type(start) is not int or type(end) is not int:
        return None
    return (start, end)


def score_sentence(
    text: str, gold: Iterable[Span], predictions: Iterable[object]
) -> RecognitionCounts:
    """Score one sentence's exact, deduplicated place spans."""
    valid: set[Span] = set()
    invalid: set[str] = set()
    for item in predictions:
        span = _as_span(item, len(text))
        if span is None:
            invalid.add(repr(item))
        else:
            valid.add(span)
    gold_spans = set(gold)
    cell = Counts()
    cell.add(gold_spans, valid)
    false_positive = cell.false_positive + len(invalid)
    return RecognitionCounts(
        task="recognition",
        gold_spans=len(gold_spans),
        predicted_spans=cell.true_positive + false_positive,
        true_positive=cell.true_positive,
        false_positive=false_positive,
        false_negative=cell.false_negative,
        invalid_outputs=len(invalid),
    )


def sum_counts(rows: Iterable[RecognitionCounts]) -> RecognitionCounts:
    """Pool compatible sentence counts before any ratio is calculated."""
    materialized = list(rows)
    require(bool(materialized), "Cannot pool an empty count inventory")
    totals = {
        name: sum(getattr(row, name) for row in materialized) for name in _COUNT_FIELDS
    }
    return RecognitionCounts(task="recognition", **totals)
