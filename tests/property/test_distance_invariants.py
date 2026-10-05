"""Bounded property checks for distance-based evaluation metrics."""

import math
from operator import itemgetter

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
DISTANCE_ROUNDING_TOLERANCE_KM = 1e-5
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


def _error_range_rounding_tolerance(errors: list[float]) -> float:
    """Allow one ULP per value for floating-point accumulation and division."""
    return len(errors) * math.ulp(max(errors))


@given(first=coordinate)
def test_haversine_returns_zero_for_identical_coordinates(
    first: tuple[float, float],
) -> None:
    assert haversine_km(*first, *first) == 0.0


@given(first=coordinate, middle=coordinate)
def test_haversine_is_symmetric(
    first: tuple[float, float], middle: tuple[float, float]
) -> None:
    assert haversine_km(*first, *middle) == pytest.approx(
        haversine_km(*middle, *first), abs=1e-9
    )


@given(first=coordinate, last=coordinate)
def test_haversine_distance_is_bounded_by_half_the_earth_circumference(
    first: tuple[float, float], last: tuple[float, float]
) -> None:
    direct = haversine_km(*first, *last)
    assert 0.0 <= direct <= MAX_SPHERICAL_DISTANCE_KM + 1e-9


@given(first=coordinate, last=coordinate)
def test_haversine_longitude_wrap_preserves_distance(
    first: tuple[float, float], last: tuple[float, float]
) -> None:
    direct = haversine_km(*first, *last)
    assert haversine_km(*first, last[0], last[1] + 360.0) == pytest.approx(
        direct, abs=1e-8
    )


@given(first=coordinate, middle=coordinate, last=coordinate)
def test_haversine_direct_route_is_shorter_than_a_two_leg_route(
    first: tuple[float, float],
    middle: tuple[float, float],
    last: tuple[float, float],
) -> None:
    direct = haversine_km(*first, *last)
    via_middle = haversine_km(*first, *middle) + haversine_km(*middle, *last)
    # Near-antipodal paths can differ by a few millimetres from rounding.
    assert direct <= via_middle + DISTANCE_ROUNDING_TOLERANCE_KM


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
def test_mean_and_median_stay_within_observed_error_range(
    sample: tuple[
        list[tuple[tuple[float, float], tuple[float, float]]], tuple[int, ...]
    ],
) -> None:
    rows, _ = sample
    expected = annotations(list(map(itemgetter(0), rows)))
    predicted = annotations(list(map(itemgetter(1), rows)))
    errors = resolution_errors_km(expected, predicted)

    mean = mean_error_km(expected, predicted)
    median = median_error_km(expected, predicted)

    # The mean sums up to eight values before dividing. Keep the allowance to
    # one ULP per observed value at the scale of the largest distance.
    rounding_tolerance = _error_range_rounding_tolerance(errors)
    lower_bound = min(errors) - rounding_tolerance
    upper_bound = max(errors) + rounding_tolerance

    assert lower_bound <= mean <= upper_bound
    assert lower_bound <= median <= upper_bound


def test_mean_and_median_range_keeps_repeated_large_error_regression() -> None:
    """Repeated equal distances expose rounding above their exact maximum."""
    expected = annotations([(3.0, 0.0)] * 3)
    predicted = annotations([(21.0, 156.0)] * 3)
    errors = resolution_errors_km(expected, predicted)

    mean = mean_error_km(expected, predicted)
    median = median_error_km(expected, predicted)

    rounding_tolerance = _error_range_rounding_tolerance(errors)
    lower_bound = min(errors) - rounding_tolerance
    upper_bound = max(errors) + rounding_tolerance

    assert lower_bound <= mean <= upper_bound
    assert lower_bound <= median <= upper_bound


def _permuted_points(
    points: list[tuple[float, float]], permutation: tuple[int, ...]
) -> list[tuple[float, float]]:
    """Apply one generated ordering to a coordinate sequence."""
    return [points[index] for index in permutation]


@given(sample=error_rows_with_permutation())
def test_mean_and_median_are_permutation_invariant(
    sample: tuple[
        list[tuple[tuple[float, float], tuple[float, float]]], tuple[int, ...]
    ],
) -> None:
    rows, permutation = sample
    expected_points = list(map(itemgetter(0), rows))
    predicted_points = list(map(itemgetter(1), rows))
    expected = annotations(expected_points)
    predicted = annotations(predicted_points)
    shuffled_expected = annotations(_permuted_points(expected_points, permutation))
    shuffled_predicted = annotations(_permuted_points(predicted_points, permutation))
    mean = mean_error_km(expected, predicted)
    median = median_error_km(expected, predicted)

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


def _annotations_with_unresolved_rows(
    rows: list[tuple[tuple[float, float] | None, tuple[float, float] | None]],
) -> tuple[list[Annotation], list[Annotation]]:
    """Retain every gold span and only the resolved predictions."""
    expected: list[Annotation] = []
    predicted: list[Annotation] = []
    for index, (gold, guess) in enumerate(rows):
        expected.append(annotation_at(index, gold))
        if guess is not None:
            predicted.append(annotation_at(index, guess))
    return expected, predicted


@given(
    rows=st.lists(
        st.tuples(st.one_of(st.none(), coordinate), st.one_of(st.none(), coordinate)),
        max_size=8,
    )
)
def test_resolution_errors_count_each_located_gold_span(
    rows: list[tuple[tuple[float, float] | None, tuple[float, float] | None]],
) -> None:
    expected, predicted = _annotations_with_unresolved_rows(rows)
    errors = resolution_errors_km(expected, predicted)
    assert len(errors) == sum(gold is not None for gold, _ in rows)


@given(
    rows=st.lists(
        st.tuples(st.one_of(st.none(), coordinate), st.one_of(st.none(), coordinate)),
        max_size=8,
    )
)
def test_resolution_errors_are_nonnegative(
    rows: list[tuple[tuple[float, float] | None, tuple[float, float] | None]],
) -> None:
    expected, predicted = _annotations_with_unresolved_rows(rows)
    errors = resolution_errors_km(expected, predicted)
    assert all(error >= 0.0 for error in errors)
