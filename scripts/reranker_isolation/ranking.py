"""Choose one referent per reference from a frozen shortlist and its scores.

Ranking is by score, highest first. Equal scores keep the shortlist order, so a
tie never depends on how a library happens to sort its output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from scripts.reranker_isolation.candidates import Candidate
from scripts.reranker_isolation.scoring import checked_scores

Status = Literal["resolved", "abstained", "invalid"]


@dataclass(frozen=True)
class Decision:
    """The outcome for one reference.

    ``abstained`` means the system had candidates but chose none, or no candidate
    reached the threshold. ``invalid`` means the scorer's output was unusable.
    The two are reported separately, and both count as misses.
    """

    status: Status
    identifier: str | None


def rank_positions(scores: Sequence[float]) -> list[int]:
    """Return shortlist positions, best first; equal scores keep their order."""
    return sorted(
        range(len(scores)), key=lambda position: (-scores[position], position)
    )


def decide(
    shortlist: Sequence[Candidate],
    scores: Sequence[float] | None,
    threshold: float | None,
) -> Decision:
    """Pick the top candidate, abstaining below the threshold.

    The threshold is inclusive. ``scores`` is None when the scorer produced no
    usable output for a non-empty shortlist.

    Raises:
        ValueError: If there is not exactly one score per candidate.
    """
    if not shortlist:
        return Decision(status="abstained", identifier=None)
    if scores is None:
        return Decision(status="invalid", identifier=None)
    _check_one_score_per_candidate(shortlist, scores)
    best = rank_positions(scores)[0]
    if not _clears(scores[best], threshold):
        return Decision(status="abstained", identifier=None)
    return Decision(status="resolved", identifier=shortlist[best].identifier)


def _check_one_score_per_candidate(
    shortlist: Sequence[Candidate], scores: Sequence[float]
) -> None:
    if len(scores) != len(shortlist):
        msg = (
            "expected one score per candidate: "
            f"{len(scores)} scores for {len(shortlist)} candidates"
        )
        raise ValueError(msg)


def _clears(score: float, threshold: float | None) -> bool:
    """Whether a score reaches the threshold; no threshold means always."""
    return threshold is None or score >= threshold


def resolve(
    shortlist: Sequence[Candidate],
    scorer: Callable[[Sequence[Candidate]], Sequence[object]],
    threshold: float | None,
) -> Decision:
    """Score a shortlist and decide, classifying unusable output as invalid.

    An empty shortlist abstains without calling the scorer. A scorer that
    returns malformed output, or that fails at runtime, yields ``invalid``.
    Programming errors such as a ``TypeError`` are not caught.
    """
    if not shortlist:
        return Decision(status="abstained", identifier=None)
    try:
        scores = checked_scores(scorer(shortlist), len(shortlist))
    except (ArithmeticError, RuntimeError, ValueError):
        return Decision(status="invalid", identifier=None)
    return decide(shortlist, scores, threshold)
