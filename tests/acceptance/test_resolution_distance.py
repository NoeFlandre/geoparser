from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then

from geoparser.evaluation import Annotation, accuracy_at_km, haversine_km

pytestmark = pytest.mark.acceptance
scenarios("features/resolution_distance.feature")

# One degree of latitude spans about 111.195 km on the mean-radius sphere.
_KM_PER_DEGREE_LATITUDE = 111.195


@pytest.fixture
def distance_state() -> dict[str, Any]:
    return {}


@given("a gold place and a prediction 100 km away")
def gold_and_prediction(distance_state: dict[str, Any]) -> None:
    gold = Annotation(start=0, end=5, latitude=46.0, longitude=7.0)
    offset = 100 / _KM_PER_DEGREE_LATITUDE
    predicted = Annotation(start=0, end=5, latitude=46.0 + offset, longitude=7.0)
    distance = haversine_km(46.0, 7.0, 46.0 + offset, 7.0)
    assert distance == pytest.approx(100, abs=0.5)
    distance_state.update(expected=[gold], predicted=[predicted])


@then(parsers.parse("accuracy at {threshold:d} km is {accuracy:f}"))
def accuracy(distance_state: dict[str, Any], threshold: int, accuracy: float) -> None:
    measured = accuracy_at_km(
        distance_state["expected"],
        distance_state["predicted"],
        threshold_km=threshold,
    )
    assert measured == accuracy
