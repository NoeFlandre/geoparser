"""Resolution gain, regressions, timing separation and paired intervals.

The scenario is four eligible examples with gold IDs Q1 to Q4. Its counts are
counted by hand in the comments of ``SCENARIO``. The bootstrap is checked with
degenerate inputs and with properties, because its draws are seeded random
numbers rather than values that can be counted by hand.
"""

import math

import pytest

from scripts.reranker_isolation.ranking import Decision, Status
from scripts.reranker_isolation.report import (
    Outcome,
    Timing,
    paired_gain_interval,
    percentile_interval,
    summarize,
)

RESOLVED: Status = "resolved"


def outcome(
    example_id: str,
    system: str,
    gold: str,
    status: Status,
    identifier: str | None,
    seconds: float,
) -> Outcome:
    return Outcome(
        example_id=example_id,
        system=system,
        gold_id=gold,
        decision=Decision(status=status, identifier=identifier),
        seconds=seconds,
    )


# Gold IDs are Q1 to Q4 for e1 to e4.
#   none  resolves e1 Q1 (right), e2 Q9 (wrong), e3 Q3 (right), e4 Q8 (wrong): 2 right
#   qwen  resolves e1 Q1 (right), e2 Q2 (right), e3 is invalid, e4 abstains: 2 right
#         fixes: e2. regressions: e3 (right under none, not under qwen). gain 0.
#   jina  resolves all four correctly: 4 right. fixes: e2 and e4. regressions 0.
SCENARIO = [
    outcome("e1", "none", "Q1", RESOLVED, "Q1", 0.0),
    outcome("e2", "none", "Q2", RESOLVED, "Q9", 0.0),
    outcome("e3", "none", "Q3", RESOLVED, "Q3", 0.0),
    outcome("e4", "none", "Q4", RESOLVED, "Q8", 0.0),
    outcome("e1", "qwen3", "Q1", RESOLVED, "Q1", 0.25),
    outcome("e2", "qwen3", "Q2", RESOLVED, "Q2", 0.25),
    outcome("e3", "qwen3", "Q3", "invalid", None, 0.25),
    outcome("e4", "qwen3", "Q4", "abstained", None, 0.25),
    outcome("e1", "jina", "Q1", RESOLVED, "Q1", 0.5),
    outcome("e2", "jina", "Q2", RESOLVED, "Q2", 0.5),
    outcome("e3", "jina", "Q3", RESOLVED, "Q3", 0.5),
    outcome("e4", "jina", "Q4", RESOLVED, "Q4", 0.5),
]

TIMINGS = {
    "none": Timing(
        fetch_seconds=0.0,
        load_seconds=0.0,
        warmup_seconds=0.0,
        steady_seconds=0.0,
        peak_rss_bytes=1000,
    ),
    "qwen3": Timing(
        fetch_seconds=10.0,
        load_seconds=3.0,
        warmup_seconds=1.0,
        steady_seconds=1.0,
        peak_rss_bytes=1500,
    ),
    "jina": Timing(
        fetch_seconds=5.0,
        load_seconds=2.0,
        warmup_seconds=0.5,
        steady_seconds=2.0,
        peak_rss_bytes=None,
    ),
}


def summary() -> dict:
    return summarize(SCENARIO, TIMINGS, baseline="none", seed=7, resamples=200)


# --- counts, gain and regressions ---------------------------------------------------


def test_baseline_counts_its_own_outcomes() -> None:
    baseline = summary()["none"]

    assert (baseline.eligible, baseline.correct) == (4, 2)
    assert (baseline.resolved, baseline.abstained, baseline.invalid) == (4, 0, 0)
    assert baseline.accuracy == pytest.approx(0.5)


def test_baseline_has_no_gain_of_its_own() -> None:
    baseline = summary()["none"]

    assert baseline.gain is None
    assert baseline.gain_interval is None
    assert (baseline.fixes, baseline.regressions) == (0, 0)


def test_qwen_counts_its_invalid_and_abstained_outcomes() -> None:
    qwen = summary()["qwen3"]

    assert (qwen.eligible, qwen.correct) == (4, 2)
    assert (qwen.resolved, qwen.abstained, qwen.invalid) == (2, 1, 1)
    assert qwen.accuracy == pytest.approx(0.5)


def test_qwen_gain_is_zero_with_one_fix_and_one_regression() -> None:
    qwen = summary()["qwen3"]

    assert qwen.gain == pytest.approx(0.0)
    assert (qwen.fixes, qwen.regressions) == (1, 1)


def test_jina_gain_is_half_with_two_fixes_and_no_regressions() -> None:
    jina = summary()["jina"]

    assert (jina.eligible, jina.correct) == (4, 4)
    assert jina.accuracy == pytest.approx(1.0)
    assert jina.gain == pytest.approx(0.5)
    assert (jina.fixes, jina.regressions) == (2, 0)


def test_gain_intervals_are_ordered_and_bounded_by_the_metric_range() -> None:
    results = summary()

    for name in ("qwen3", "jina"):
        low, high = results[name].gain_interval
        assert -1.0 <= low <= high <= 1.0


def test_summaries_are_reproducible_for_the_same_seed() -> None:
    assert summary() == summary()


def test_pairing_ignores_the_order_of_outcomes() -> None:
    shuffled = list(reversed(SCENARIO))

    assert summarize(shuffled, TIMINGS, baseline="none", seed=7, resamples=200) == (
        summary()
    )


# --- timing and memory --------------------------------------------------------------


def test_loading_time_is_reported_apart_from_steady_inference() -> None:
    qwen = summary()["qwen3"]

    assert qwen.timing.load_seconds == pytest.approx(3.0)
    assert qwen.mean_steady_seconds == pytest.approx(0.25)


def test_memory_increase_is_relative_to_no_reranker() -> None:
    assert summary()["qwen3"].extra_peak_rss_bytes == 500
    assert summary()["none"].extra_peak_rss_bytes == 0


def test_unavailable_peak_memory_stays_unavailable_not_zero() -> None:
    assert summary()["jina"].extra_peak_rss_bytes is None


@pytest.mark.parametrize(
    "field",
    ["fetch_seconds", "load_seconds", "warmup_seconds", "steady_seconds"],
)
def test_timing_rejects_negative_or_non_finite_seconds(field: str) -> None:
    values = {
        "fetch_seconds": 0.0,
        "load_seconds": 0.0,
        "warmup_seconds": 0.0,
        "steady_seconds": 0.0,
    }
    values[field] = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        Timing(peak_rss_bytes=1, **values)
    values[field] = math.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        Timing(peak_rss_bytes=1, **values)


def test_timing_rejects_a_zero_peak_memory_reading() -> None:
    with pytest.raises(ValueError, match="positive or unavailable"):
        Timing(
            fetch_seconds=0.0,
            load_seconds=0.0,
            warmup_seconds=0.0,
            steady_seconds=0.0,
            peak_rss_bytes=0,
        )


def test_outcome_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        outcome("e1", "none", "Q1", RESOLVED, "Q1", -0.1)


# --- pairing validation -------------------------------------------------------------


def test_summaries_need_at_least_one_outcome() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize([], TIMINGS, baseline="none", seed=1, resamples=10)


def test_the_baseline_must_be_present() -> None:
    without_baseline = [row for row in SCENARIO if row.system != "none"]

    with pytest.raises(ValueError, match="baseline"):
        summarize(without_baseline, TIMINGS, baseline="none", seed=1, resamples=10)


def test_every_system_must_cover_the_same_examples() -> None:
    missing_one = [
        row for row in SCENARIO if not (row.system == "jina" and row.example_id == "e4")
    ]

    with pytest.raises(ValueError, match="paired"):
        summarize(missing_one, TIMINGS, baseline="none", seed=1, resamples=10)


def test_systems_must_agree_on_the_gold_id_of_each_example() -> None:
    relabelled = [
        outcome("e1", "jina", "Q7", RESOLVED, "Q1", 0.5)
        if row.system == "jina" and row.example_id == "e1"
        else row
        for row in SCENARIO
    ]

    with pytest.raises(ValueError, match="gold IDs differ"):
        summarize(relabelled, TIMINGS, baseline="none", seed=1, resamples=10)


def test_duplicate_outcomes_are_rejected() -> None:
    duplicated = [*SCENARIO, SCENARIO[0]]

    with pytest.raises(ValueError, match="duplicate"):
        summarize(duplicated, TIMINGS, baseline="none", seed=1, resamples=10)


def test_every_system_needs_a_timing_record() -> None:
    timings = {name: timing for name, timing in TIMINGS.items() if name != "jina"}

    with pytest.raises(ValueError, match="timing"):
        summarize(SCENARIO, timings, baseline="none", seed=1, resamples=10)


def test_bootstrap_needs_at_least_one_resample() -> None:
    with pytest.raises(ValueError, match="resamples"):
        summarize(SCENARIO, TIMINGS, baseline="none", seed=1, resamples=0)


# --- intervals ---------------------------------------------------------------------


def test_percentile_interval_uses_the_inverse_cdf_positions() -> None:
    # Rank 25 of 1000 values and rank 975 of 1000 values, zero-based.
    assert percentile_interval(list(range(1000))) == (24, 974)


def test_percentile_interval_of_one_value_is_that_value() -> None:
    assert percentile_interval([0.3]) == (0.3, 0.3)


def test_percentile_interval_needs_values() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile_interval([])


def test_identical_systems_have_a_zero_gain_interval() -> None:
    same = [True, False, True]

    assert paired_gain_interval(same, same, seed=1, resamples=50) == (0.0, 0.0)


def test_always_right_against_always_wrong_has_a_gain_of_one() -> None:
    assert paired_gain_interval([True] * 4, [False] * 4, seed=1, resamples=50) == (
        1.0,
        1.0,
    )


def test_paired_gain_interval_of_no_examples_is_zero() -> None:
    assert paired_gain_interval([], [], seed=1, resamples=50) == (0.0, 0.0)


def test_paired_gain_interval_needs_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_gain_interval([True], [True, False], seed=1, resamples=5)


def test_paired_gain_interval_needs_at_least_one_resample() -> None:
    with pytest.raises(ValueError, match="resamples"):
        paired_gain_interval([True], [False], seed=1, resamples=0)
