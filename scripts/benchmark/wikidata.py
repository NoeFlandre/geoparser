"""
Look up the coordinates of Wikidata items, cache first.

HIPE links toponyms to Wikidata rather than giving coordinates, so the gold
locations come from each item's coordinate location (P625). The answers are
kept in a JSON file committed next to the harness: the gold then cannot drift
as Wikidata is edited, and a compute node needs no access to Wikidata.
"""

from __future__ import annotations

import json
import typing as t
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

API_URL = "https://www.wikidata.org/w/api.php"
BATCH_SIZE = 50
COORDINATE_PROPERTY = "P625"
USER_AGENT = "geoparser-benchmark/1.0 (https://github.com/NoeFlandre/geoparser)"

Located = tuple[float, float] | None
Fetch = Callable[[list[str]], Mapping[str, Located]]


def coordinates_from_entities(response: Mapping[str, t.Any]) -> dict[str, Located]:
    """Return each entity's first coordinate claim, or None when it has none."""
    found: dict[str, Located] = {}
    for qid, entity in response.get("entities", {}).items():
        found[qid] = None
        for claim in entity.get("claims", {}).get(COORDINATE_PROPERTY, []):
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            if value:
                found[qid] = (float(value["latitude"]), float(value["longitude"]))
                break
    return found


def fetch_from_wikidata(qids: list[str]) -> dict[str, Located]:
    """Ask the Wikidata API for one batch of items' coordinate claims."""
    query = urllib.parse.urlencode(
        {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": "claims",
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return coordinates_from_entities(json.load(response))


def load_coordinates(
    qids: Iterable[str], cache_path: Path, *, fetch: Fetch = fetch_from_wikidata
) -> dict[str, tuple[float, float]]:
    """
    Return the coordinates of the located items among ``qids``.

    Items missing from the cache are fetched in batches and written back,
    including those Wikidata knows no location for, so a later run asks
    for nothing it has already been told.

    Args:
        qids: The Wikidata items to place
        cache_path: The JSON cache of QID to [latitude, longitude] or null
        fetch: Looks up one batch of at most BATCH_SIZE items

    Returns:
        QID to (latitude, longitude), for the items that have coordinates
    """
    wanted = set(qids)
    cache: dict[str, t.Any] = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    missing = sorted(wanted - cache.keys())
    if missing:
        for offset in range(0, len(missing), BATCH_SIZE):
            batch = missing[offset : offset + BATCH_SIZE]
            fetched = fetch(batch)
            for qid in batch:
                value = fetched.get(qid)
                cache[qid] = list(value) if value else None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(dict(sorted(cache.items())), indent=0) + "\n", encoding="utf-8"
        )
    return {qid: (cache[qid][0], cache[qid][1]) for qid in wanted if cache.get(qid)}
