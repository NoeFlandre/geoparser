"""Development-only threshold selection and digests of the frozen decision."""

import pytest

from scripts.reranker_isolation.thresholds import (
    DevExample,
    canonical_digest,
    select_threshold,
    utility,
    verify_digest,
)

# Hand-counted development examples. The last one abstained or was invalid, so
# no threshold can resolve it; it shifts every utility equally.
DEV = [
    DevExample("a", top_score=0.9, top_correct=True),
    DevExample("b", top_score=0.8, top_correct=False),
    DevExample("c", top_score=0.6, top_correct=True),
    DevExample("d", top_score=0.4, top_correct=True),
    DevExample("e", top_score=None, top_correct=False),
]


def test_utility_counts_correct_resolutions_and_penalises_wrong_ones() -> None:
    # Threshold 0.5 resolves a, b and c: two correct, one wrong.
    assert utility(DEV, 0.5, wrong_penalty=0.5) == pytest.approx(2 - 0.5)
    # Threshold 0.85 resolves only a.
    assert utility(DEV, 0.85, wrong_penalty=0.5) == pytest.approx(1.0)
    # Threshold 0.0 resolves a, b, c and d: three correct, one wrong.
    assert utility(DEV, 0.0, wrong_penalty=0.5) == pytest.approx(3 - 0.5)


def test_selection_prefers_the_highest_utility() -> None:
    assert select_threshold(DEV, [0.0, 0.5, 0.85], wrong_penalty=0.5) == 0.0


def test_selection_can_choose_a_higher_threshold_when_errors_are_costly() -> None:
    # With penalty 2: 0.0 gives 3 - 2 = 1, 0.5 gives 2 - 2 = 0, 0.85 gives 1.
    # 0.95 resolves nothing, so it gives 0 and does not win.
    assert select_threshold(DEV, [0.5, 0.85, 0.95], wrong_penalty=2.0) == 0.85


def test_ties_go_to_the_lowest_threshold() -> None:
    # With penalty 2: 0.0 gives 3 - 2 = 1 and 0.9 gives 1 (only a is resolved).
    assert select_threshold(DEV, [0.9, 0.0], wrong_penalty=2.0) == 0.0


def test_selection_needs_a_grid_and_a_non_negative_penalty() -> None:
    with pytest.raises(ValueError, match="grid"):
        select_threshold(DEV, [], wrong_penalty=0.5)
    with pytest.raises(ValueError, match="penalty"):
        select_threshold(DEV, [0.5], wrong_penalty=-1.0)


def test_canonical_digest_ignores_key_order_and_matches_a_known_value() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert canonical_digest({"a": 1, "b": 2}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )


def test_canonical_digest_refuses_non_finite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_digest({"threshold": float("nan")})


def test_verify_digest_accepts_the_frozen_decision_and_refuses_a_change() -> None:
    frozen = {"threshold": 0.5}
    digest = canonical_digest(frozen)

    verify_digest(frozen, digest)
    with pytest.raises(ValueError, match="does not match"):
        verify_digest({"threshold": 0.6}, digest)


def test_selection_rejects_a_non_finite_grid_value() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        select_threshold(DEV, [0.5, float("nan")], wrong_penalty=0.5)
