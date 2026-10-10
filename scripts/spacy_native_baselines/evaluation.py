"""Score native baselines on shared examples with matched and transfer kept apart.

Three groups are reported and never merged:

* ``matched``: a native pipeline scored on examples in its own language.
* ``transfer``: the English cross-language control scored on every
  non-English language that has a native pipeline. These are labelled transfer
  scores, not native results.
* ``unsupported``: languages with no native pipeline. They are counted and
  never predicted by any model, including the English control in either group.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from scripts.panx_benchmark.data import Example
from scripts.panx_benchmark.metrics import Counts, macro_scores
from scripts.spacy_native_baselines.roster import ENGLISH_CONTROL_LANGUAGE, Roster


class BatchPredictor(Protocol):
    """Predict place spans for an ordered batch of texts."""

    def predict_batch(self, texts: list[str]) -> list[set[tuple[int, int]]]:
        """Return one span set per input text."""


def _score(
    predictor: BatchPredictor,
    examples: Sequence[Example],
    batch_size: int,
) -> dict[str, Any]:
    counts = Counts()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        started = time.perf_counter()
        predictions = predictor.predict_batch([example.text for example in batch])
        counts.elapsed_seconds += time.perf_counter() - started
        if len(predictions) != len(batch):
            message = "A predictor must return one span set per input text."
            raise ValueError(message)
        for example, spans in zip(batch, predictions, strict=True):
            counts.add(example.gold_spans, spans, text_length=len(example.text))
            counts.malformed_gold_tags += example.malformed_location_tags
    return counts.scores()


def _positive_batch_size(value: Any) -> int:
    if type(value) is not int or value < 1:
        message = "batch_size must be a positive integer."
        raise ValueError(message)
    return value


def _has_native_pipeline(roster: Roster, language: str) -> bool:
    return roster.route(language).pipeline is not None


def _check_native_recognizers(
    recognizers: Mapping[str, BatchPredictor],
    roster: Roster,
) -> None:
    for language in recognizers:
        route = roster.route(language)
        if route.pipeline is None:
            message = (
                f"Refusing to attach a recognizer to unsupported language "
                f"{language!r}: {route.reason}"
            )
            raise ValueError(message)


def _matched_scores(
    recognizers: Mapping[str, BatchPredictor],
    examples_by_language: Mapping[str, Sequence[Example]],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    return {
        language: _score(
            recognizers[language], examples_by_language[language], batch_size
        )
        for language in examples_by_language
        if language in recognizers
    }


def _transfer_scores(
    english_control: BatchPredictor | None,
    examples_by_language: Mapping[str, Sequence[Example]],
    roster: Roster,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    if english_control is None:
        return {}
    return {
        language: _score(english_control, examples, batch_size)
        for language, examples in examples_by_language.items()
        if language != ENGLISH_CONTROL_LANGUAGE
        and _has_native_pipeline(roster, language)
    }


def _unsupported_counts(
    examples_by_language: Mapping[str, Sequence[Example]],
    roster: Roster,
) -> dict[str, dict[str, Any]]:
    unsupported: dict[str, dict[str, Any]] = {}
    for language, examples in examples_by_language.items():
        route = roster.route(language)
        if route.pipeline is None:
            unsupported[language] = {
                "sentences": len(examples),
                "reason": route.reason,
            }
    return unsupported


def evaluate_baselines(
    examples_by_language: Mapping[str, Sequence[Example]],
    recognizers: Mapping[str, BatchPredictor],
    english_control: BatchPredictor | None,
    roster: Roster,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score every arm on the same examples and return the three groups."""
    batch_size = _positive_batch_size(batch_size)
    _check_native_recognizers(recognizers, roster)
    matched = _matched_scores(recognizers, examples_by_language, batch_size)
    transfer = _transfer_scores(
        english_control, examples_by_language, roster, batch_size
    )
    return {
        "matched": matched,
        "matched_macro": macro_scores(matched),
        "transfer": transfer,
        "transfer_macro": macro_scores(transfer),
        "unsupported": _unsupported_counts(examples_by_language, roster),
    }
