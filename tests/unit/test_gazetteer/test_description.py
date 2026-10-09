"""
Tests for the text built from a gazetteer feature's attributes.

A candidate is compared to a reference's context as a sentence built from its
gazetteer attributes. The exact wording is what the encoder sees, so the
separators are behaviour rather than presentation.
"""

from unittest.mock import Mock

import pytest

from geoparser.gazetteer.description import (
    GAZETTEER_ATTRIBUTE_MAP,
    admin_levels,
    describe_feature,
)

GEONAMES = GAZETTEER_ATTRIBUTE_MAP["geonames"]


def _candidate(**data) -> Mock:
    """A gazetteer candidate carrying the given attributes."""
    candidate = Mock()
    candidate.data = data
    return candidate


@pytest.mark.unit
class TestDescribeFeature:
    """Turning a candidate's attributes into a sentence."""

    def test_names_the_place_its_type_and_its_administrative_context(self):
        """The full form reads as one sentence, most specific level first."""
        # Arrange
        candidate = _candidate(
            name="Paris",
            feature_name="city",
            admin2_name="Ile-de-France",
            admin1_name="IDF",
            country_name="France",
        )

        # Act & Assert
        assert (
            describe_feature(candidate.data, GEONAMES)
            == "Paris (city) in Ile-de-France, IDF, France"
        )

    def test_omits_the_type_when_the_candidate_has_none(self):
        """A candidate without a feature type still reads correctly."""
        # Arrange
        candidate = _candidate(name="Paris", country_name="France")

        # Act & Assert
        assert describe_feature(candidate.data, GEONAMES) == "Paris in France"

    def test_omits_the_administrative_context_when_there_is_none(self):
        """With no admin levels there is no trailing "in"."""
        # Arrange
        candidate = _candidate(name="Paris", feature_name="city")

        # Act & Assert
        assert describe_feature(candidate.data, GEONAMES) == "Paris (city)"

    def test_skips_empty_administrative_levels(self):
        """Blank levels are left out rather than producing empty separators."""
        # Arrange
        candidate = _candidate(name="Paris", admin2_name="", country_name="France")

        # Act & Assert
        assert describe_feature(candidate.data, GEONAMES) == "Paris in France"

    def test_describes_a_candidate_with_no_usable_attributes_as_empty(self):
        """
        Nothing known means nothing to embed.

        The annotator has its own description helper that falls back to the
        identifier; the resolver deliberately does not, so an attribute-less
        candidate contributes an empty string rather than a bare number.
        """
        # Arrange
        candidate = _candidate()

        # Act & Assert
        assert describe_feature(candidate.data, GEONAMES) == ""


@pytest.mark.unit
class TestAdminLevels:
    """Ordering the administrative names."""

    def test_returns_levels_from_most_to_least_specific(self):
        """level3, then level2, then level1."""
        # Act
        levels = admin_levels(
            {"admin2_name": "A2", "admin1_name": "A1", "country_name": "France"},
            GEONAMES,
        )

        # Assert
        assert levels == ["A2", "A1", "France"]

    def test_returns_nothing_when_no_level_is_populated(self):
        """Missing levels yield an empty list, not blanks."""
        # Act & Assert
        assert admin_levels({}, GEONAMES) == []
