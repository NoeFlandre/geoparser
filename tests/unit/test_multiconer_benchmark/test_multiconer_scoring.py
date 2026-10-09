"""Exact-span recognition counts that reuse the shared protocol contract."""

import pytest

from scripts.benchmark_protocol.schema import RecognitionCounts
from scripts.multiconer_benchmark.scoring import score_sentence

TEXT = "我 在 北京 的 故宫 参观"  # 14 code points


def test_invalid_and_duplicate_predictions_follow_the_protocol_penalty():
    # Gold: 北京 (4,6) and 故宫 (9,11).
    # Valid distinct: (4,6) TP, (9,11) TP, (0,2) FP.
    # Invalid distinct: (5,5) empty, (3,99) out of range, "4-6" malformed.
    predictions = [(4, 6), (9, 11), (0, 2), (5, 5), (3, 99), (4, 6), "4-6"]
    counts = score_sentence(TEXT, {(4, 6), (9, 11)}, predictions)
    assert isinstance(counts, RecognitionCounts)
    assert counts.gold_spans == 2
    assert counts.true_positive == 2
    assert counts.false_positive == 4
    assert counts.false_negative == 0
    assert counts.predicted_spans == 6
    assert counts.invalid_outputs == 3


def test_missed_gold_span_is_a_false_negative_and_empty_prediction_is_zero():
    counts = score_sentence(TEXT, {(4, 6), (9, 11)}, [(4, 6)])
    assert (counts.true_positive, counts.false_positive, counts.false_negative) == (
        1,
        0,
        1,
    )
    assert counts.predicted_spans == 1

    empty = score_sentence(TEXT, set(), [])
    assert empty.gold_spans == 0
    assert empty.predicted_spans == 0


@pytest.mark.parametrize("bad", [(True, 2), (1.0, 3), (2, None), [1, 2, 3]])
def test_malformed_prediction_types_are_invalid_outputs_not_crashes(bad):
    counts = score_sentence(TEXT, {(4, 6)}, [bad])
    assert counts.invalid_outputs == 1
    assert counts.false_positive == 1
    assert counts.true_positive == 0
