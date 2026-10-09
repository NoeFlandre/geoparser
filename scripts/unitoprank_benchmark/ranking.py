"""
Rank the toponyms of one document with UniTopRank and map the top choice back.

The ranker returns addresses, not identifiers, so every address is traced to
the gazetteer identifier it was built from. A mention is unresolved when it has
no candidate, or when the ranker returns none for its surface form. Those two
cases are kept apart, because they mean different things about the candidate
retrieval and about the ranking.

Mentions that share a surface form share one ranking, as they do in the ranker,
so their outcomes agree by construction.

The ranking function is passed in. Tests use a stub; the benchmark loads the
pinned ``rank_toponyms`` through :mod:`scripts.unitoprank_benchmark.pins`.
"""

from __future__ import annotations

import typing as t
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scripts.unitoprank_benchmark.candidates import CandidateSet, normalize_surface

RankFunction = Callable[..., t.Mapping[str, t.Any]]


@dataclass(frozen=True)
class Mention:
    """One toponym in the text, as character offsets and its surface form."""

    start: int
    end: int
    surface: str


@dataclass(frozen=True)
class Outcome:
    """What UniTopRank decided for one mention."""

    start: int
    end: int
    surface: str
    identifier: str | None
    latitude: float | None
    longitude: float | None
    score: float | None
    candidate_count: int
    tied: bool
    reason: str | None


def rank_document(
    text: str,
    mentions: Sequence[Mention],
    candidates: CandidateSet,
    *,
    rank_toponyms: RankFunction,
    ranker_config: t.Any,
) -> list[Outcome]:
    """
    Resolve every mention of one document.

    Args:
        text: The document text the offsets refer to
        mentions: Every toponym to resolve, in text order
        candidates: The candidate set built for this document's surfaces
        rank_toponyms: The ranking function, called with keyword arguments
        ranker_config: The ``RankerConfig`` to pass, frozen before evaluation

    Returns:
        One outcome per mention, in the order the mentions were given

    Raises:
        ValueError: If the ranker names an address no candidate was built with,
            which would make the result impossible to trace to an identifier
    """
    payload = [
        {"text": mention.surface, "start": mention.start, "end": mention.end}
        for mention in mentions
    ]
    result = rank_toponyms(
        text=text,
        toponyms=payload,
        candidates_by_toponym=candidates.by_surface,
        config=ranker_config,
    )
    ranked = result["ranked_candidates_by_toponym"]
    return [
        _outcome(
            mention, candidates, ranked.get(normalize_surface(mention.surface), [])
        )
        for mention in mentions
    ]


def _outcome(
    mention: Mention, candidates: CandidateSet, ranking: Sequence[t.Mapping[str, t.Any]]
) -> Outcome:
    """Build the outcome for one mention from its surface's ranking."""
    key = normalize_surface(mention.surface)
    offered = len(candidates.by_surface.get(key, []))
    if not ranking:
        reason = "no_candidates" if offered == 0 else "not_ranked"
        return _unresolved(mention, offered, reason)
    top = ranking[0]
    address = str(top["address"])
    identifiers = candidates.identifiers.get(key, {})
    if address not in identifiers:
        message = f"ranker returned an address that was not offered: {address!r}"
        raise ValueError(message)
    score = float(top["score"])
    tied = len(ranking) > 1 and float(ranking[1]["score"]) == score
    return Outcome(
        start=mention.start,
        end=mention.end,
        surface=mention.surface,
        identifier=identifiers[address],
        latitude=float(top["lat"]),
        longitude=float(top["lon"]),
        score=score,
        candidate_count=offered,
        tied=tied,
        reason=None,
    )


def _unresolved(mention: Mention, offered: int, reason: str) -> Outcome:
    """Return the outcome of a mention UniTopRank did not place."""
    return Outcome(
        start=mention.start,
        end=mention.end,
        surface=mention.surface,
        identifier=None,
        latitude=None,
        longitude=None,
        score=None,
        candidate_count=offered,
        tied=False,
        reason=reason,
    )
