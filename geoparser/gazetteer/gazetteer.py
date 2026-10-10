"""
Public gazetteer query interface.

A Gazetteer wraps an installed artifact (a self-contained, read-only SQLite
file) and exposes name search and identifier lookup over its features.
"""

from __future__ import annotations

import typing as t

from geoparser.gazetteer.artifact import (
    DEFAULT_SEARCH_LIMIT,
    GazetteerArtifact,
    SearchMethod,
    artifact_path,
)
from geoparser.gazetteer.feature import Feature

# The artifact query that answers each search method.
_SEARCH_QUERIES: t.Final[dict[SearchMethod, str]] = {
    "exact": "search_exact",
    "phrase": "search_phrase",
    "partial": "search_partial",
    "fuzzy": "search_fuzzy",
}


def normalize_name(name: str) -> str:
    """Remove double quote characters and trim whitespace from a query."""
    return name.replace('"', "").strip()


class Gazetteer:
    """
    A gazetteer interface for querying geographic features.

    This class provides access to an installed gazetteer artifact, allowing
    retrieval of candidate features for name matching using different search
    strategies: exact, phrase, partial, and fuzzy matching.
    """

    def __init__(self, gazetteer_name: str):
        """
        Initialize the gazetteer interface.

        Args:
            gazetteer_name: Name of the gazetteer to query for candidates

        Raises:
            ValueError: If the gazetteer is not installed. Querying an
                uninstalled gazetteer would silently return no results, so we
                fail here instead of letting that happen unnoticed.
        """
        path = artifact_path(gazetteer_name)
        if not path.exists():
            # pragma: no mutate start - wording only; a test pins the type
            # and that the message names the missing gazetteer.
            msg = (
                f"Gazetteer '{gazetteer_name}' is not installed. Install it by running "
                f"'geoparser install {gazetteer_name}', or run "
                "'geoparser list' to see which gazetteers are installed."
            )
            raise ValueError(msg)
            # pragma: no mutate end
        self.gazetteer_name = gazetteer_name
        self._artifact = GazetteerArtifact(path)

    @property
    def crs(self) -> str:
        """Coordinate reference system of the gazetteer's geometries."""
        return self._artifact.crs

    def search(
        self,
        name: str,
        method: SearchMethod = "exact",
        limit: int = DEFAULT_SEARCH_LIMIT,
        tiers: int = 1,
    ) -> list[Feature]:
        """
        Search for features using the specified search method.

        Args:
            name: Name string to search for
            method: Search method to use ("exact", "phrase", "partial", "fuzzy")
            limit: Maximum number of results to return (default: ``DEFAULT_SEARCH_LIMIT``)
            tiers: Number of rank tiers to include in results (default: 1,
                ignored for exact method)

        Returns:
            List of Feature objects matching the search criteria

        Raises:
            ValueError: If an unknown search method is specified
        """
        # Remove quotes and trim whitespace
        normalized_name = normalize_name(name)
        if not normalized_name:
            return []

        # Callers may pass any string at runtime, so check it against the table.
        if method not in _SEARCH_QUERIES:
            msg = f"Unknown search method: {method}"
            raise ValueError(msg)

        query = getattr(self._artifact, _SEARCH_QUERIES[method])
        if method == "exact":
            return query(normalized_name, limit)
        return query(normalized_name, limit, tiers)

    def find(self, identifier: str) -> Feature | None:
        """
        Find a feature by its identifier.

        Args:
            identifier: The identifier value of the feature to find

        Returns:
            Feature object if found, None otherwise
        """
        return self._artifact.find(identifier)


# One Gazetteer per installed artifact, keyed by name and by the file's
# identity, so a reinstall (which replaces the file) is picked up rather than
# served from a connection to the old one.
_OPEN: dict[str, tuple[tuple[int, int], Gazetteer]] = {}


def get_gazetteer(gazetteer_name: str) -> Gazetteer:
    """
    Return a shared Gazetteer for an installed artifact.

    Opening a Gazetteer checks the file and opens a SQLite connection, which
    adds up when it is done per feature lookup. The artifact is read-only and
    keeps one connection per thread, so one instance can be shared.

    Args:
        gazetteer_name: Name of the gazetteer

    Returns:
        The shared Gazetteer

    Raises:
        ValueError: If the gazetteer is not installed
    """
    try:
        stat = artifact_path(gazetteer_name).stat()
    except FileNotFoundError:
        return Gazetteer(gazetteer_name)
    stamp = (stat.st_ino, stat.st_mtime_ns)
    cached = _OPEN.get(gazetteer_name)
    if cached is None or cached[0] != stamp:
        cached = (stamp, Gazetteer(gazetteer_name))
        _OPEN[gazetteer_name] = cached
    return cached[1]
