"""
Ranking helpers for resolving toponyms in many languages.

Two failure modes of the upstream resolver on non-English text motivate this
module. An inflected form ("Saksan", Finnish for "of Germany", or German
"Chinas") has no exact gazetteer match, so the search falls through to fuzzy
matching and lands on an unrelated place; retrying exact search on the name
with its ending trimmed recovers most of them. And an exonym ("Bâle") matches
hamlets as readily as the city, which a context embedding alone does not
always separate; a small, logarithmic population prior breaks those near-ties
without overriding context when context is decisive.

Like :mod:`~geoparser.modules.resolvers.context`, this is plain arithmetic and
imports nothing from the rest of the package or the machine learning stack.
"""

from __future__ import annotations

import math
import typing as t
from collections.abc import Sequence

# Longest ending tried when an exact search misses. Three characters covers
# the common case endings of Finnish, Swedish, German and French plurals
# without trimming a name into a different word.
MAX_TRIM = 3
# Shortest name left after trimming; below this, stubs match almost anything.
MIN_STEM = 3
# log10 of the population is divided by this, so the most populous country
# (about 1.4 billion, log10 ~ 9.2) stays below 1.
PRIOR_SCALE = 10.0


def population_prior(population: t.Any) -> float:
    """
    Return a candidate's population prior, in [0, 1).

    Args:
        population: The gazetteer's population value, possibly missing

    Returns:
        log10(1 + population) / PRIOR_SCALE, or 0 when it is unknown
    """
    try:
        value = float(population)
    except (TypeError, ValueError):
        return 0.0
    if not value > 0:
        return 0.0
    return math.log10(1 + value) / PRIOR_SCALE


def combined_scores(
    similarities: Sequence[float], populations: Sequence[t.Any], weight: float
) -> list[float]:
    """
    Return each candidate's similarity plus its weighted population prior.

    Args:
        similarities: Context-to-candidate similarity per candidate
        populations: Population per candidate, in the same order
        weight: How much the prior counts; 0 leaves similarities unchanged

    Returns:
        One score per candidate
    """
    if not weight:
        return list(similarities)
    return [
        similarity + weight * population_prior(population)
        for similarity, population in zip(similarities, populations, strict=True)
    ]


def inflection_variants(name: str) -> list[str]:
    """
    Return shorter forms of a single-word name to retry an exact search with.

    Args:
        name: The toponym as written

    Returns:
        The name with 1 to MAX_TRIM trailing characters removed, shortest
        trim first, each at least MIN_STEM characters; empty for a phrase
    """
    stem = name.strip()
    if not _is_single_word(stem):
        return []
    return [
        stem[:-trim] for trim in range(1, MAX_TRIM + 1) if len(stem) - trim >= MIN_STEM
    ]


def _is_single_word(text: str) -> bool:
    """Whether the text is one non-empty word, with no whitespace inside."""
    return bool(text) and not any(character.isspace() for character in text)
