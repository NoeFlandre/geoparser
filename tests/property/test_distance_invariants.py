"""Bounded property checks for distance-based evaluation metrics."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geoparser.evaluation import (
    Annotation,
    accuracy_at_km,
    area_under_error_curve,
    haversine_km,
    mean_error_km,
    median_error_km,
    resolution_errors_km,
)

pytestmark = pytest.mark.property

EARTH_RADIUS_KM = 6371.0088
MAX_SPHERICAL_DISTANCE_KM = math.pi * EARTH_RADIUS_KM
latitude = st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)
longitude = st.floats(
    min_value=-180, max_value=180, allow_nan=False, allow_infinity=False
)
coordinate = st.tuples(latitude, longitude)
threshold = st.floats(
    min_value=0,
    max_value=MAX_SPHERICAL_DISTANCE_KM,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def error_rows_with_permutation(draw: st.DrawFn):
    rows = draw(st.lists(st.tuples(coordinate, coordinate), min_size=1, max_size=8))
    permutation = draw(st.permutations(tuple(range(len(rows)))))
    return rows, permutation


def annotation_at(index: int, point: tuple[float, float] | None) -> Annotation:
    """Give a generated point a distinct span in one document."""
    latitude_value, longitude_value = point if point is not None else (None, None)
    return Annotation(
        index * 2,
        index * 2 + 1,
        document_id="property",
        latitude=latitude_value,
        longitude=longitude_value,
    )


def annotations(points: list[tuple[float, float]]) -> list[Annotation]:
    """Give each generated coordinate a distinct span in one document."""
    return [annotation_at(index, point) for index, point in enumerate(points)]


@given(first=coordinate, middle=coordinate, last=coordinate)
def test_haversine_geodesic_invariants(
    first: tuple[float, float],
    middle: tuple[float, float],
    last: tuple[float, float],
) -> None:
    assert haversine_km(*first, *first) == 0.0
    assert haversine_km(*first, *middle) == pytest.approx(
        haversine_km(*middle, *first), abs=1e-9
    )

    direct = haversine_km(*first, *last)
    assert 0.0 <= direct <= MAX_SPHERICAL_DISTANCE_KM + 1e-9
    assert haversine_km(*first, last[0], last[1] + 360.0) == pytest.approx(
        direct, abs=1e-8
    )

    via_middle = haversine_km(*first, *middle) + haversine_km(*middle, *last)

    assert direct <= via_middle + 1e-8


@given(
    rows=st.lists(st.tuples(coordinate, coordinate), max_size=8),
    thresholds=st.tuples(threshold, threshold),
)
def test_accuracy_is_bounded_and_monotone_with_threshold(
    rows: list[tuple[tuple[float, float], tuple[float, float]]],
    thresholds: tuple[float, float],
) -> None:
    expected = annotations([gold for gold, _ in rows])
    predicted = annotations([guess for _, guess in rows])
    lower, upper = sorted(thresholds)

    lower_score = accuracy_at_km(expected, predicted, threshold_km=lower)
    upper_score = accuracy_at_km(expected, predicted, threshold_km=upper)

    assert 0.0 <= lower_score <= upper_score <= 1.0


@given(sample=error_rows_with_permutation())
def test_mean_and_median_are_bounded_and_permutation_invariant(
    sample: tuple[
        list[tuple[tuple[float, float], tuple[float, float]]], tuple[int, ...]
    ],
) -> None:
    rows, permutation = sample
    expected_points = [gold for gold, _ in rows]
    predicted_points = [guess for _, guess in rows]
    expected = annotations(expected_points)
    predicted = annotations(predicted_points)
    errors = resolution_errors_km(expected, predicted)
    shuffled_expected = annotations([expected_points[i] for i in permutation])
    shuffled_predicted = annotations([predicted_points[i] for i in permutation])

    mean = mean_error_km(expected, predicted)
    median = median_error_km(expected, predicted)

    assert min(errors) <= mean <= max(errors)
    assert min(errors) <= median <= max(errors)
    assert mean_error_km(shuffled_expected, shuffled_predicted) == pytest.approx(mean)
    assert median_error_km(shuffled_expected, shuffled_predicted) == pytest.approx(
        median
    )


@given(first=coordinate, second=coordinate, count=st.integers(min_value=1, max_value=8))
def test_median_of_identical_errors_equals_the_common_error(
    first: tuple[float, float], second: tuple[float, float], count: int
) -> None:
    expected = annotations([first] * count)
    predicted = annotations([second] * count)
    common_error = haversine_km(*first, *second)

    assert median_error_km(expected, predicted) == pytest.approx(common_error)


@given(
    rows=st.lists(st.tuples(coordinate, coordinate), max_size=8),
    unresolved_error_km=st.floats(
        min_value=5e-324,
        max_value=MAX_SPHERICAL_DISTANCE_KM * 2,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_area_under_error_curve_is_bounded_and_perfect_scores_zero(
    rows: list[tuple[tuple[float, float], tuple[float, float]]],
    unresolved_error_km: float,
) -> None:
    expected = annotations([gold for gold, _ in rows])
    predicted = annotations([guess for _, guess in rows])

    score = area_under_error_curve(
        expected, predicted, unresolved_error_km=unresolved_error_km
    )
    perfect_score = area_under_error_curve(
        expected, expected, unresolved_error_km=unresolved_error_km
    )

    assert 0.0 <= score <= 1.0
    assert perfect_score == 0.0


@given(
    rows=st.lists(
        st.tuples(st.one_of(st.none(), coordinate), st.one_of(st.none(), coordinate)),
        max_size=8,
    )
)
def test_resolution_errors_match_located_gold_spans_and_are_nonnegative(
    rows: list[tuple[tuple[float, float] | None, tuple[float, float] | None]],
) -> None:
    expected = [annotation_at(index, gold) for index, (gold, _) in enumerate(rows)]
    predicted = [
        annotation_at(index, guess)
        for index, (_, guess) in enumerate(rows)
        if guess is not None
    ]
    errors = resolution_errors_km(expected, predicted)

    assert len(errors) == sum(gold is not None for gold, _ in rows)
    assert all(error >= 0.0 for error in errors)
