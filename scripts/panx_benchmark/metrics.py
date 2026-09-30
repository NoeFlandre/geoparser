"""Exact-span location scores for the PAN-X recognition comparison."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from scripts.panx_benchmark.data import Span


@dataclass(slots=True)
class Counts:
    """Accumulated exact-span outcomes for one language or model."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    sentences: int = 0
    malformed_gold_tags: int = 0
    elapsed_seconds: float = 0.0

    def add(self, gold: Iterable[Span], predicted: Iterable[Span]) -> None:
        """Accumulate one sentence using exact, half-open character spans."""
        gold_spans, predicted_spans = set(gold), set(predicted)
        self.true_positive += len(gold_spans & predicted_spans)
        self.false_positive += len(predicted_spans - gold_spans)
        self.false_negative += len(gold_spans - predicted_spans)
        self.sentences += 1

    def scores(self) -> dict[str, float | int]:
        """Return counts and zero-division-safe precision, recall, and F1."""
        precision = ratio(self.true_positive, self.true_positive + self.false_positive)
        recall = ratio(self.true_positive, self.true_positive + self.false_negative)
        f1 = ratio(2 * precision * recall, precision + recall)
        return {
            "sentences": self.sentences,
            "gold_spans": self.true_positive + self.false_negative,
            "predicted_spans": self.true_positive + self.false_positive,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "malformed_gold_tags": self.malformed_gold_tags,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "elapsed_seconds": self.elapsed_seconds,
            "sentences_per_second": ratio(self.sentences, self.elapsed_seconds),
        }


def ratio(numerator: float, denominator: float) -> float:
    """Return zero when a metric denominator is empty."""
    return numerator / denominator if denominator else 0.0


def macro_scores(per_language: dict[str, dict[str, float | int]]) -> dict[str, float]:
    """Average language-level scores with equal weight per evaluated language."""
    rows = list(per_language.values())
    if not rows:
        return dict.fromkeys(("precision", "recall", "f1"), 0.0)
    return {
        metric: fmean(float(row[metric]) for row in rows)
        for metric in ("precision", "recall", "f1")
    }
