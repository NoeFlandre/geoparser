"""Score native baselines on shared examples with matched and transfer kept apart.

Three groups are reported and never merged:

* ``matched``: a native pipeline scored on examples in its own language.
* ``transfer``: the English cross-language control scored on every
  non-English language. These are labelled transfer scores, not native results.
* ``unsupported``: languages with no native pipeline. They are counted and
  never predicted by any model, including the English control in ``matched``.
"""

from __future__ import annotations

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
        predictions = predictor.predict_batch([example.text for example in batch])
        if len(predictions) != len(batch):
            message = "A predictor must return one span set per input text."
            raise ValueError(message)
        for example, spans in zip(batch, predictions, strict=True):
            counts.add(example.gold_spans, spans, text_length=len(example.text))
    return counts.scores()


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


def evaluate_baselines(
    examples_by_language: Mapping[str, Sequence[Example]],
    recognizers: Mapping[str, BatchPredictor],
    english_control: BatchPredictor | None,
    roster: Roster,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score every arm on the same examples and return the three groups."""
    _check_native_recognizers(recognizers, roster)

    matched = {
        language: _score(
            recognizers[language], examples_by_language[language], batch_size
        )
        for language in examples_by_language
        if language in recognizers
    }

    transfer: dict[str, dict[str, Any]] = {}
    if english_control is not None:
        transfer = {
            language: _score(english_control, examples, batch_size)
            for language, examples in examples_by_language.items()
            if language != ENGLISH_CONTROL_LANGUAGE
        }

    unsupported = {}
    for language, examples in examples_by_language.items():
        route = roster.route(language)
        if route.pipeline is None:
            unsupported[language] = {
                "sentences": len(examples),
                "reason": route.reason,
            }

    return {
        "matched": matched,
        "matched_macro": macro_scores(matched),
        "transfer": transfer,
        "transfer_macro": macro_scores(transfer),
        "unsupported": unsupported,
    }
