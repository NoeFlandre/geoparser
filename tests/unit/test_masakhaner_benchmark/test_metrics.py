import random

import pytest

from scripts.masakhaner_benchmark.metrics import (
    bootstrap_f1_interval,
    micro_scores,
    sentence_counts,
)


def test_exact_span_counts_need_both_boundaries_to_match():
    counts = sentence_counts(
        gold={(0, 4), (6, 9)},
        predicted={(0, 4), (10, 12)},
    )

    assert counts == (1, 1, 1)


def test_micro_scores_pool_the_counts_before_dividing():
    scores = micro_scores([(1, 1, 1), (2, 0, 1)])

    assert (scores.true_positive, scores.false_positive, scores.false_negative) == (
        3,
        1,
        2,
    )
    assert scores.precision == pytest.approx(0.75)
    assert scores.recall == pytest.approx(0.6)
    assert scores.f1 == pytest.approx(6 / 9)


def test_empty_counts_give_zero_scores_without_dividing_by_zero():
    scores = micro_scores([])

    assert (scores.precision, scores.recall, scores.f1) == (0.0, 0.0, 0.0)


def test_identical_perfect_sentences_give_a_degenerate_interval_at_one():
    assert bootstrap_f1_interval([(1, 0, 0)] * 5, seed=3) == (1.0, 1.0)


def test_sentences_without_spans_give_a_degenerate_interval_at_zero():
    assert bootstrap_f1_interval([(0, 0, 0)] * 3, seed=3) == (0.0, 0.0)


def test_a_perfect_and_an_empty_sentence_bracket_the_bootstrap_range():
    # Resampled pairs give F1 of 1 (both perfect), 0.5 (mixed) or 0 (both
    # wrong). With 2000 resamples the 2.5th and 97.5th percentiles are 0 and 1.
    low, high = bootstrap_f1_interval([(1, 0, 0), (0, 1, 1)], seed=7, resamples=2000)

    assert (low, high) == (0.0, 1.0)


def test_the_same_seed_and_counts_give_the_same_interval():
    counts = [(2, 1, 0), (0, 2, 1), (3, 0, 1), (1, 1, 1)]

    first = bootstrap_f1_interval(counts, seed=11, resamples=200)
    second = bootstrap_f1_interval(counts, seed=11, resamples=200)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


@pytest.mark.parametrize(
    ("counts", "resamples", "confidence", "message"),
    [
        ([], 10, 0.95, "at least one scored sentence"),
        ([(1, 0, 0)], 0, 0.95, "at least one resample"),
        ([(1, 0, 0)], 10, 1.0, "strictly between 0 and 1"),
    ],
)
def test_invalid_bootstrap_settings_are_rejected(
    counts, resamples, confidence, message
):
    with pytest.raises(ValueError, match=message):
        bootstrap_f1_interval(
            counts, seed=0, resamples=resamples, confidence=confidence
        )


def test_percentile_bounds_are_the_expected_order_statistics():
    # Replays the same seeded draws to check the index arithmetic: with 1000
    # resamples the 2.5th and 97.5th percentiles are the 25th and 975th values.
    counts = [(2, 1, 0), (0, 2, 1), (3, 0, 1), (1, 1, 1)]
    rng = random.Random(5)
    size = len(counts)
    values = sorted(
        micro_scores(counts[rng.randrange(size)] for _ in range(size)).f1
        for _ in range(1000)
    )

    assert bootstrap_f1_interval(counts, seed=5, resamples=1000) == (
        values[24],
        values[974],
    )
