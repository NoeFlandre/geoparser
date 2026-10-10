"""Shortlist policy and candidate-recall ceiling on frozen first-stage lists."""

import pytest

from scripts.reranker_isolation.candidates import (
    Candidate,
    FrozenItem,
    recall_ceiling,
    shortlist,
)


def item(example_id: str, gold: str | None, identifiers: list[str]) -> FrozenItem:
    candidates = tuple(
        Candidate(identifier=identifier, description=f"place {identifier}")
        for identifier in identifiers
    )
    return FrozenItem(
        example_id=example_id,
        context="Travelled from Lyon to Paris.",
        candidates=candidates,
        gold_id=gold,
    )


def identifiers(candidates: tuple[Candidate, ...]) -> list[str]:
    return [candidate.identifier for candidate in candidates]


def test_shortlist_keeps_the_frozen_first_stage_order() -> None:
    frozen = item("a", "Q1", ["Q2", "Q1", "Q3"])

    assert identifiers(shortlist(frozen, 2)) == ["Q2", "Q1"]


def test_shortlist_larger_than_the_list_returns_every_candidate() -> None:
    frozen = item("a", "Q1", ["Q2", "Q1", "Q3"])

    assert identifiers(shortlist(frozen, 10)) == ["Q2", "Q1", "Q3"]


def test_shortlist_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        shortlist(item("a", "Q1", ["Q1"]), 0)


def test_candidate_identifiers_must_be_unique_within_an_item() -> None:
    with pytest.raises(ValueError, match="unique"):
        item("a", "Q1", ["Q1", "Q1"])


RECALL_ITEMS = [
    item("a", "Q1", ["Q1", "Q2", "Q3"]),  # gold at rank 1
    item("b", "Q4", ["Q2", "Q4", "Q1"]),  # gold at rank 2
    item("c", "Q5", ["Q1", "Q2"]),  # gold missing from the frozen list
    item("d", None, ["Q1"]),  # no canonical ID, so not eligible
]


def test_recall_ceiling_excludes_items_without_a_canonical_gold_id() -> None:
    assert recall_ceiling(RECALL_ITEMS, 1).eligible == 3
    assert recall_ceiling(RECALL_ITEMS, 3).eligible == 3


def test_recall_ceiling_finds_golds_inside_the_shortlist_only() -> None:
    at_one = recall_ceiling(RECALL_ITEMS, 1)
    at_two = recall_ceiling(RECALL_ITEMS, 2)

    assert (at_one.eligible, at_one.found) == (3, 1)
    assert at_one.ratio == pytest.approx(1 / 3)
    assert (at_two.eligible, at_two.found) == (3, 2)
    assert at_two.ratio == pytest.approx(2 / 3)


def test_a_gold_missing_from_the_frozen_list_stays_a_miss_at_any_size() -> None:
    assert recall_ceiling(RECALL_ITEMS, 3).found == 2


def test_recall_ceiling_with_no_eligible_items_is_zero_and_visible() -> None:
    ceiling = recall_ceiling([item("d", None, ["Q1"])], 1)

    assert (ceiling.eligible, ceiling.found, ceiling.ratio) == (0, 0, 0.0)


def test_recall_ceiling_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        recall_ceiling([item("a", "Q1", ["Q1"])], 0)
