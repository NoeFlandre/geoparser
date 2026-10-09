"""
Map gazetteer candidates onto the candidate sets UniTopRank ranks.

UniTopRank ranks the candidates it is given. It does not retrieve them, so this
module is the step where the choices that change a ranking are written down:
which candidates are kept, how a missing population or administrative level is
represented, and the order in which candidates are offered. Every drop and
every substitution is counted, so the report can say how much of the gazetteer
the ranker never saw.

Candidates are offered in identifier order, whatever order the gazetteer
returned them in. The ranker breaks exact score ties by input order, so this
makes a tie resolve to the lower identifier on every run.

UniTopRank merges candidates whose address string is equal, so two candidates
that would share an address are not both offered: the lower identifier is kept
and the other is counted as an address collision.

Like the other benchmark helpers, this module holds no models, no gazetteer and
no network access.
"""

from __future__ import annotations

import math
import typing as t
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """One gazetteer feature, with only the fields UniTopRank can use."""

    identifier: str
    name: str
    latitude: float | None
    longitude: float | None
    feature_code: str | None = None
    population: int | float | None = None
    # Narrowest administrative level first, broadest last, for example
    # ("Lamar County", "Texas", "United States", "North America"). A missing
    # level is None or empty; it is dropped from the address and counted.
    admin_path: tuple[str | None, ...] = ()
    alternate_names: tuple[str, ...] = ()


@dataclass
class CandidateSet:
    """Candidates per normalized surface form, ready for the ranker."""

    by_surface: dict[str, list[dict[str, t.Any]]] = field(default_factory=dict)
    # Surface, then address string, then the gazetteer identifier it stands for.
    identifiers: dict[str, dict[str, str]] = field(default_factory=dict)
    dropped: Counter[str] = field(default_factory=Counter)
    missing: Counter[str] = field(default_factory=Counter)


def normalize_surface(surface: str) -> str:
    """Return the key UniTopRank uses for a toponym: stripped and lower case."""
    return surface.strip().lower()


def build_candidate_set(
    candidates_by_surface: Mapping[str, Iterable[Candidate]],
) -> CandidateSet:
    """
    Turn gazetteer candidates into the ranker's input, deterministically.

    Surfaces that normalize to the same key share one candidate list, as they
    do in the ranker. Within a key, candidates are kept in identifier order.

    Args:
        candidates_by_surface: Surface form to the candidates retrieved for it

    Returns:
        The candidate set, with the counts of what was dropped or substituted
    """
    pooled: dict[str, list[Candidate]] = {}
    for surface, candidates in candidates_by_surface.items():
        pooled.setdefault(normalize_surface(surface), []).extend(candidates)

    result = CandidateSet()
    for key in sorted(pooled):
        _add_surface(result, key, pooled[key])
    return result


def _add_surface(result: CandidateSet, key: str, candidates: list[Candidate]) -> None:
    """Keep the usable candidates for one surface, in identifier order."""
    entries: list[dict[str, t.Any]] = []
    identifiers: dict[str, str] = {}
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.identifier):
        if candidate.identifier in seen:
            result.dropped["duplicate_identifier"] += 1
            continue
        seen.add(candidate.identifier)
        entry = _to_entry(candidate, result)
        if entry is None:
            continue
        if entry["address"] in identifiers:
            result.dropped["address_collision"] += 1
            continue
        identifiers[entry["address"]] = candidate.identifier
        entries.append(entry)
    result.by_surface[key] = entries
    result.identifiers[key] = identifiers


def _to_entry(candidate: Candidate, result: CandidateSet) -> dict[str, t.Any] | None:
    """Return the ranker's dictionary for one candidate, or None if unusable."""
    name = candidate.name.strip()
    if not name:
        result.dropped["no_name"] += 1
        return None
    coordinates = _coordinates(candidate.latitude, candidate.longitude)
    if coordinates is None:
        result.dropped["no_coordinates"] += 1
        return None
    levels = [
        level.strip() for level in candidate.admin_path if level and level.strip()
    ]
    if not candidate.admin_path:
        result.missing["admin_path"] += 1
    result.missing["admin_level"] += len(candidate.admin_path) - len(levels)
    if not candidate.feature_code or not candidate.feature_code.strip():
        result.missing["feature_code"] += 1
    return {
        "address": ", ".join([name, *levels]),
        "lat": coordinates[0],
        "lon": coordinates[1],
        "name": name,
        "alt_names": sorted(
            {alias.strip() for alias in candidate.alternate_names if alias.strip()}
        ),
        "population": _population(candidate.population, result),
        "admin_level": (candidate.feature_code or "").strip(),
    }


def _coordinates(
    latitude: float | None, longitude: float | None
) -> tuple[float, float] | None:
    """Return finite, in-range coordinates, or None."""
    if latitude is None or longitude is None:
        return None
    lat, lon = float(latitude), float(longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def _population(value: int | float | None, result: CandidateSet) -> int:
    """Return a population, counting a missing or negative one as zero."""
    if value is None or not math.isfinite(value) or value < 0:
        result.missing["population"] += 1
        return 0
    return int(value)
