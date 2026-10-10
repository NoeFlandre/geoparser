"""Development-only threshold selection, and digests that freeze the decision.

The utility counts correct resolutions and charges a fixed penalty for each
wrong one. Abstentions and invalid outputs earn nothing and cost nothing.
Their contribution does not depend on the threshold, so they cannot change
which threshold wins. The penalty is an explicit choice, not a default. Choose
it before looking at any test outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DevExample:
    """One development example's top candidate, as a threshold would see it.

    ``top_score`` is None when the example abstained or was invalid, so no
    threshold can resolve it.
    """

    example_id: str
    top_score: float | None
    top_correct: bool


def utility(
    examples: Sequence[DevExample], threshold: float, *, wrong_penalty: float
) -> float:
    """Correct resolutions minus the penalty times wrong resolutions."""
    correct = 0
    wrong = 0
    for example in examples:
        if example.top_score is None or example.top_score < threshold:
            continue
        if example.top_correct:
            correct += 1
        else:
            wrong += 1
    return correct - wrong_penalty * wrong


def select_threshold(
    examples: Sequence[DevExample], grid: Sequence[float], *, wrong_penalty: float
) -> float:
    """Return the grid threshold with the highest utility on development data.

    Ties go to the lowest threshold, because the candidates are scanned in
    ascending order and ``max`` keeps the first maximum it meets.

    Raises:
        ValueError: If the grid is empty or not finite, or the penalty is not a
            finite non-negative number.
    """
    _check_selection_inputs(grid, wrong_penalty)
    return max(
        sorted(grid),
        key=lambda threshold: utility(examples, threshold, wrong_penalty=wrong_penalty),
    )


def _check_selection_inputs(grid: Sequence[float], wrong_penalty: float) -> None:
    _check_grid(grid)
    _check_penalty(wrong_penalty)


def _check_grid(grid: Sequence[float]) -> None:
    if not grid:
        msg = "threshold grid must not be empty"
        raise ValueError(msg)
    if not all(math.isfinite(threshold) for threshold in grid):
        msg = "threshold grid values must be finite"
        raise ValueError(msg)


def _check_penalty(wrong_penalty: float) -> None:
    if not math.isfinite(wrong_penalty) or wrong_penalty < 0:
        msg = f"wrong penalty must be finite and non-negative, got {wrong_penalty}"
        raise ValueError(msg)


def canonical_digest(payload: Mapping[str, object]) -> str:
    """SHA-256 of the payload as sorted, compact JSON with no NaN or infinity."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_digest(payload: Mapping[str, object], expected: str) -> None:
    """Refuse a decision whose contents no longer match its frozen digest.

    Raises:
        ValueError: If the payload's digest differs from ``expected``.
    """
    if canonical_digest(payload) != expected:
        msg = "decision digest does not match the frozen value"
        raise ValueError(msg)
