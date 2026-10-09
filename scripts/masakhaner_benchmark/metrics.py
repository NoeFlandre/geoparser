"""Exact-span scores and a seeded sentence bootstrap for MasakhaNER 2.0.

A span is correct only when its start and end match a gold span exactly. The
bootstrap resamples sentences with replacement. It reports a percentile interval
for micro F1, so one language's uncertainty is not hidden by its point estimate.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from scripts.masakhaner_benchmark.data import Span

SentenceCounts = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MicroScores:
    """Pooled exact-span counts with zero-division-safe precision and recall."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the scores as a plain mapping for JSON reports."""
        return asdict(self)


def ratio(numerator: float, denominator: float) -> float:
    """Return zero when a metric denominator is empty."""
    return numerator / denominator if denominator else 0.0


def sentence_counts(gold: Iterable[Span], predicted: Iterable[Span]) -> SentenceCounts:
    """Return true positive, false positive and false negative span counts."""
    gold_spans, predicted_spans = set(gold), set(predicted)
    return (
        len(gold_spans & predicted_spans),
        len(predicted_spans - gold_spans),
        len(gold_spans - predicted_spans),
    )


def micro_scores(counts: Iterable[SentenceCounts]) -> MicroScores:
    """Pool sentence counts and score them once, as micro averages."""
    true_positive = false_positive = false_negative = 0
    for tp, fp, fn in counts:
        true_positive += tp
        false_positive += fp
        false_negative += fn
    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    f1 = ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    return MicroScores(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def bootstrap_f1_interval(
    counts: Sequence[SentenceCounts],
    *,
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a seeded percentile interval for micro F1 over sentences.

    The same counts, seed and resample count always give the same interval.
    """
    if not counts:
        message = "A bootstrap interval needs at least one scored sentence"
        raise ValueError(message)
    if resamples < 1:
        message = "The bootstrap needs at least one resample"
        raise ValueError(message)
    if not 0 < confidence < 1:
        message = "The confidence level must lie strictly between 0 and 1"
        raise ValueError(message)
    rng = random.Random(seed)  # noqa: S311
    size = len(counts)
    values = sorted(
        micro_scores(counts[rng.randrange(size)] for _ in range(size)).f1
        for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    low = values[int(tail * (resamples - 1))]
    high = values[int((1 - tail) * (resamples - 1))]
    return low, high
