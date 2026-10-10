"""Stable ranking, candidate-ID mapping, thresholds and failure classification."""

from collections.abc import Sequence

import pytest

from scripts.reranker_isolation.candidates import Candidate
from scripts.reranker_isolation.ranking import (
    Decision,
    decide,
    rank_positions,
    resolve,
)


def candidates(*identifiers: str) -> list[Candidate]:
    return [
        Candidate(identifier=identifier, description=identifier)
        for identifier in identifiers
    ]


# --- score direction and stable ties ----------------------------------------------


def test_higher_scores_rank_first() -> None:
    assert rank_positions([0.2, 0.9, 0.5]) == [1, 2, 0]


def test_equal_scores_keep_the_shortlist_order() -> None:
    assert rank_positions([0.5, 0.9, 0.5, 0.9, 0.1]) == [1, 3, 0, 2, 4]


def test_negative_zero_ties_with_zero_and_keeps_order() -> None:
    assert rank_positions([0.0, -0.0, -1.0]) == [0, 1, 2]


def test_no_scores_rank_nothing() -> None:
    assert rank_positions([]) == []


# --- candidate-ID mapping and thresholds --------------------------------------------


def test_the_best_position_maps_to_its_candidate_identifier() -> None:
    decision = decide(candidates("A", "B", "C"), [0.1, 0.7, 0.7], threshold=None)

    assert decision == Decision(status="resolved", identifier="B")


def test_a_tied_top_score_resolves_to_the_earlier_candidate() -> None:
    decision = decide(candidates("A", "B"), [0.7, 0.7], threshold=None)

    assert decision == Decision(status="resolved", identifier="A")


def test_threshold_is_inclusive_at_the_top_score() -> None:
    at_threshold = decide(candidates("A", "B"), [0.5, 0.2], threshold=0.5)
    below_threshold = decide(candidates("A", "B"), [0.49, 0.2], threshold=0.5)

    assert at_threshold == Decision(status="resolved", identifier="A")
    assert below_threshold == Decision(status="abstained", identifier=None)


def test_without_a_threshold_any_nonempty_shortlist_resolves() -> None:
    assert decide(candidates("A"), [-0.9], threshold=None) == Decision(
        status="resolved", identifier="A"
    )


# --- empty and malformed outputs ---------------------------------------------------


def test_an_empty_shortlist_abstains_even_without_scores() -> None:
    assert decide([], None, None) == Decision(status="abstained", identifier=None)


def test_missing_scores_for_a_nonempty_shortlist_are_invalid_not_abstained() -> None:
    assert decide(candidates("A"), None, None) == Decision(
        status="invalid", identifier=None
    )


def test_score_count_must_match_the_shortlist() -> None:
    with pytest.raises(ValueError, match="one score per candidate"):
        decide(candidates("A", "B"), [0.5], None)


def test_resolve_does_not_call_the_scorer_for_an_empty_shortlist() -> None:
    calls: list[int] = []

    def scorer(shortlist: Sequence[Candidate]) -> list[float]:
        calls.append(len(shortlist))
        return []

    assert resolve([], scorer, None) == Decision(status="abstained", identifier=None)
    assert calls == []


def test_resolve_marks_a_scorer_returning_too_few_scores_invalid() -> None:
    def scorer(shortlist: Sequence[Candidate]) -> list[float]:
        return [0.5]

    assert resolve(candidates("A", "B"), scorer, None) == Decision(
        status="invalid", identifier=None
    )


def test_resolve_marks_non_finite_scores_invalid() -> None:
    def scorer(shortlist: Sequence[Candidate]) -> list[float]:
        return [float("nan"), 0.1]

    assert resolve(candidates("A", "B"), scorer, None) == Decision(
        status="invalid", identifier=None
    )


def test_resolve_marks_runtime_failures_invalid() -> None:
    def scorer(shortlist: Sequence[Candidate]) -> list[float]:
        message = "out of memory"
        raise RuntimeError(message)

    assert resolve(candidates("A"), scorer, None) == Decision(
        status="invalid", identifier=None
    )


def test_resolve_does_not_hide_programming_errors() -> None:
    def scorer(shortlist: Sequence[Candidate]) -> list[float]:
        message = "wrong keyword"
        raise TypeError(message)

    with pytest.raises(TypeError, match="wrong keyword"):
        resolve(candidates("A"), scorer, None)


def test_resolve_applies_the_threshold_to_the_scorer_output() -> None:
    def high(shortlist: Sequence[Candidate]) -> list[float]:
        return [0.3, 0.8]

    def low(shortlist: Sequence[Candidate]) -> list[float]:
        return [0.3, 0.4]

    assert resolve(candidates("A", "B"), high, 0.5) == Decision(
        status="resolved", identifier="B"
    )
    assert resolve(candidates("A", "B"), low, 0.5) == Decision(
        status="abstained", identifier=None
    )
