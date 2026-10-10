"""Frozen first-stage candidate lists, the fixed shortlist policy and recall.

A frozen item lists its candidates in first-stage order, best first, with ties
in gazetteer order. Each reranker only reorders a shortlist taken from that
frozen list, so every system sees the same candidates and the same context.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One gazetteer entry as the reranker sees it."""

    identifier: str
    description: str


@dataclass(frozen=True)
class FrozenItem:
    """One reference with its context, frozen candidates and canonical gold ID.

    ``gold_id`` is None when the gold referent has no canonical identifier. Such
    items are outside the recall and accuracy denominators.
    """

    example_id: str
    context: str
    candidates: tuple[Candidate, ...]
    gold_id: str | None

    def __post_init__(self) -> None:
        identifiers = [candidate.identifier for candidate in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            msg = f"candidate identifiers must be unique within {self.example_id}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecallCeiling:
    """How many eligible golds the shortlist can possibly hand to a reranker."""

    eligible: int
    found: int

    @property
    def ratio(self) -> float:
        """Found over eligible, or zero when nothing is eligible."""
        return self.found / self.eligible if self.eligible else 0.0


def _check_size(size: int) -> None:
    if size < 1:
        msg = f"shortlist size must be at least 1, got {size}"
        raise ValueError(msg)


def shortlist(item: FrozenItem, size: int) -> tuple[Candidate, ...]:
    """Return the first ``size`` frozen candidates, in first-stage order."""
    _check_size(size)
    return item.candidates[:size]


def recall_ceiling(items: Sequence[FrozenItem], size: int) -> RecallCeiling:
    """Count eligible golds present in each item's shortlist.

    Eligibility is a canonical gold ID. A gold that is absent from the frozen
    list is a miss, not an exclusion.
    """
    _check_size(size)
    eligible = [item for item in items if item.gold_id is not None]
    found = sum(
        item.gold_id in {candidate.identifier for candidate in shortlist(item, size)}
        for item in eligible
    )
    return RecallCeiling(eligible=len(eligible), found=found)
