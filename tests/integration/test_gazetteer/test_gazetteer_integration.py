"""
Integration tests for geoparser/gazetteer/gazetteer.py

Tests Gazetteer with a real artifact built from the Andorra fixture data.
"""

import pytest
from shapely.geometry.base import BaseGeometry

from geoparser.gazetteer.feature import Feature
from geoparser.gazetteer.gazetteer import Gazetteer


def _by_identifier(results, identifier):
    """Pick one feature out of an unordered result list."""
    return next(f for f in results if f.identifier == identifier)


@pytest.mark.integration
class TestGazetteerIntegration:
    """Integration tests for Gazetteer with real Andorra data."""

    def test_creates_with_gazetteer_name(self, andorra_gazetteer):
        """Test that Gazetteer can be initialized with gazetteer name."""
        gazetteer = Gazetteer("andorranames")

        assert gazetteer.gazetteer_name == "andorranames"

    def test_exact_search_finds_the_expected_features(self, andorra_gazetteer):
        """
        Exact search returns exactly the expected features.

        It has no defined order, so identifiers are compared sorted.
        """
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search("Andorra la Vella", method="exact")

        assert sorted(f.identifier for f in results) == ["3041563", "3041566"]
        assert _by_identifier(results, "3041566").data["name"] == "Andorra la Vella"

    @pytest.mark.parametrize(
        ("query", "method", "expected_identifiers", "top_name"),
        [
            ("Escaldes", "phrase", ["3040051"], "les Escaldes"),
            ("Andorra", "partial", ["3041563", "3041565"], "Andorra la Vella"),
            ("Andorra la Vela", "partial", ["3041563"], "Andorra la Vella"),
            ("Andora", "fuzzy", ["3041563", "3041565"], "Andorra la Vella"),
            ("Andorrra", "fuzzy", ["3041563", "3041565"], "Andorra la Vella"),
            ("Escaldez", "fuzzy", ["3040051"], "les Escaldes"),
        ],
    )
    def test_scored_search_methods_find_the_expected_features(
        self, andorra_gazetteer, query, method, expected_identifiers, top_name
    ):
        """Scored methods order by score then identifier, so order is exact."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search(query, method=method)

        assert [f.identifier for f in results] == expected_identifiers
        assert results[0].data["name"] == top_name

    def test_search_respects_limit_parameter(self, andorra_gazetteer):
        """Test that search respects the limit parameter."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search("Andorra", method="partial", limit=1)

        # Tied scores may be cut either way, so only the count and candidates
        assert len(results) == 1
        assert results[0].identifier in {"3041563", "3041565"}

    def test_search_respects_tiers_parameter(self, andorra_gazetteer):
        """Test that search respects the tiers parameter for score-based tiering."""
        gazetteer = Gazetteer("andorranames")

        results_tier1 = gazetteer.search("Andorra", method="phrase", tiers=1)
        results_tier2 = gazetteer.search("Andorra", method="phrase", tiers=2)

        assert [f.identifier for f in results_tier1] == ["3041563", "3041565"]
        assert len(results_tier2) == 9
        assert [f.identifier for f in results_tier2[:2]] == ["3041563", "3041565"]

    def test_search_normalizes_quotes(self, andorra_gazetteer):
        """Test that search normalizes quotation marks in names."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search('"Andorra"', method="exact")

        assert sorted(f.identifier for f in results) == ["3041563", "3041565"]

    def test_search_strips_whitespace(self, andorra_gazetteer):
        """Test that search strips leading/trailing whitespace."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search("  Andorra  ", method="exact")

        assert sorted(f.identifier for f in results) == ["3041563", "3041565"]

    def test_find_returns_specific_feature(self, andorra_gazetteer):
        """Test that find returns a specific feature by identifier."""
        gazetteer = Gazetteer("andorranames")

        # Andorra la Vella has geonameid 3041563
        feature = gazetteer.find("3041563")

        assert feature is not None  # narrows the type for the checks below
        assert (feature.identifier, feature.data["name"]) == (
            "3041563",
            "Andorra la Vella",
        )

    def test_find_returns_none_for_nonexistent_feature(self, andorra_gazetteer):
        """Test that find returns None for non-existent identifier."""
        gazetteer = Gazetteer("andorranames")

        feature = gazetteer.find("9999999")

        assert feature is None

    def test_search_returns_feature_objects(self, andorra_gazetteer):
        """Test that search returns runtime Feature objects."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search("Andorra", method="exact")

        assert sorted(f.identifier for f in results) == ["3041563", "3041565"]
        assert all(isinstance(f, Feature) for f in results)

    def test_feature_has_geometry(self, andorra_gazetteer):
        """Test that returned features have geometry information."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search("Andorra la Vella", method="exact")

        feature = _by_identifier(results, "3041563")
        assert isinstance(feature.geometry, BaseGeometry)
        assert feature.geometry.wkt == "POINT (1.52109 42.50779)"
        assert feature.crs == "EPSG:4326"

    def test_feature_has_names(self, andorra_gazetteer):
        """Test that features have associated names."""
        gazetteer = Gazetteer("andorranames")

        results = gazetteer.search("Andorra la Vella", method="exact")

        feature = _by_identifier(results, "3041563")
        assert "Andorra la Vella" in feature.names
        assert "Andorre-la-Vieille" in feature.names

    def test_feature_has_lookup_attributes(self, andorra_gazetteer):
        """Test that features carry attributes enriched via lookups."""
        gazetteer = Gazetteer("andorranames")

        feature = gazetteer.find("3041563")
        assert feature is not None

        assert feature.data["country_name"] == "Andorra"
        assert feature.data["feature_name"] == "capital of a political entity"
        assert feature.gazetteer_name == "andorranames"

    def test_searches_multiple_parishes(self, andorra_gazetteer):
        """Test that gazetteer contains data from multiple Andorra parishes."""
        gazetteer = Gazetteer("andorranames")

        parishes = ["Canillo", "Encamp", "Ordino", "Massana"]
        results_per_parish = [
            gazetteer.search(parish, method="phrase") for parish in parishes
        ]

        assert [[f.identifier for f in results] for results in results_per_parish] == [
            ["3041203", "3041204"],
            ["3040684", "3040686", "6942578"],
            ["3039676", "3039678"],
            ["3040131"],
        ]

    def test_search_case_insensitive(self, andorra_gazetteer):
        """Test that search is case-insensitive."""
        gazetteer = Gazetteer("andorranames")

        results_lower = gazetteer.search("andorra", method="exact")
        results_upper = gazetteer.search("ANDORRA", method="exact")
        results_mixed = gazetteer.search("AnDoRRa", method="exact")

        expected = ["3041563", "3041565"]
        assert sorted(f.identifier for f in results_lower) == expected
        assert sorted(f.identifier for f in results_upper) == expected
        assert sorted(f.identifier for f in results_mixed) == expected

    def test_handles_special_characters(self, andorra_gazetteer):
        """Test that search handles diacritics in place names."""
        gazetteer = Gazetteer("andorranames")

        with_accent = gazetteer.search("Ansalonga", method="exact")

        assert [f.identifier for f in with_accent] == ["3041546"]

    def test_search_returns_consistent_results(self, andorra_gazetteer):
        """Test that multiple searches return consistent results."""
        gazetteer = Gazetteer("andorranames")

        results1 = gazetteer.search("Andorra la Vella", method="exact")
        results2 = gazetteer.search("Andorra la Vella", method="exact")

        assert sorted(f.identifier for f in results1) == ["3041563", "3041566"]
        assert sorted(f.identifier for f in results2) == ["3041563", "3041566"]

    def test_different_search_methods_return_result_lists(self, andorra_gazetteer):
        """Test that different search methods all return lists."""
        gazetteer = Gazetteer("andorranames")
        query = "Andorra"

        exact_results = gazetteer.search(query, method="exact")
        partial_results = gazetteer.search(query, method="partial")

        assert isinstance(exact_results, list)
        assert isinstance(partial_results, list)
        assert sorted(f.identifier for f in exact_results) == ["3041563", "3041565"]
        assert [f.identifier for f in partial_results] == ["3041563", "3041565"]
