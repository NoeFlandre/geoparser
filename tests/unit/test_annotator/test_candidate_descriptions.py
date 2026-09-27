"""The annotator's candidate descriptions, shared with the resolver."""

from types import SimpleNamespace

import pytest

from geoparser.annotator.db.crud.toponym import ToponymRepository

CITY_DATA = {
    "name": "Zurich",
    "feature_name": "seat of a first-order administrative division",
    "country_name": "Switzerland",
    "admin1_name": "Zurich",
    "admin2_name": "Bezirk Zurich",
}


def _feature(data):
    return SimpleNamespace(identifier="2657896", data=data)


@pytest.mark.unit
class TestLocationDescription:
    def test_describes_geonames_cities_features(self):
        """geonames-cities gets a real description, not the bare identifier."""
        description = ToponymRepository._generate_location_description(
            _feature(CITY_DATA), "geonames-cities"
        )

        assert description == (
            "Zurich (seat of a first-order administrative division) "
            "in Bezirk Zurich, Zurich, Switzerland"
        )

    def test_matches_geonames(self):
        feature = _feature(CITY_DATA)

        assert ToponymRepository._generate_location_description(
            feature, "geonames-cities"
        ) == ToponymRepository._generate_location_description(feature, "geonames")

    def test_unknown_gazetteer_falls_back_to_identifier(self):
        assert (
            ToponymRepository._generate_location_description(
                _feature(CITY_DATA), "unknown"
            )
            == "2657896"
        )

    def test_empty_data_falls_back_to_identifier(self):
        assert (
            ToponymRepository._generate_location_description(_feature({}), "geonames")
            == "2657896"
        )

    def test_empty_description_falls_back_to_identifier(self):
        assert (
            ToponymRepository._generate_location_description(
                _feature({"other": "x"}), "geonames"
            )
            == "2657896"
        )
