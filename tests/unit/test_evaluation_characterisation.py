"""Characterisation of the evaluation module's values and accepted inputs.

These pin what the public functions return and which inputs they accept, using
small in-memory fixtures, so that type-only refactors cannot change either.
"""

from types import SimpleNamespace

import pytest

from geoparser.evaluation import (
    ACCURACY_THRESHOLD_KM,
    MAX_ERROR_KM,
    Annotation,
    _location_coordinates,
    _record_unique,
    accuracy_at_km,
    area_under_error_curve,
    haversine_km,
    mean_error_km,
    median_error_km,
    recognition_f1,
    recognition_precision,
    recognition_recall,
    resolution_accuracy,
    resolution_errors_km,
    toponym_annotation,
)

# Recorded from the implementation for the fixtures below; the distance values
# are compared with a relative tolerance because they depend on trigonometry.
EARTH_QUARTER_KM = 10007.557221017962
EARTH_HALF_KM = 20015.114442035923


class _Feature:
    """Shaped like a gazetteer feature: an identifier and a data mapping."""

    def __init__(self, identifier, data):
        self.identifier = identifier
        self._data = data

    @property
    def data(self):
        return self._data


class _Reference:
    """Shaped like a recognised reference: a span and an optional location."""

    def __init__(self, start, end, location):
        self.start = start
        self.end = end
        self._location = location

    @property
    def location(self):
        return self._location


def _gold():
    return [
        Annotation(0, 3, "g:1", latitude=0.0, longitude=0.0),
        Annotation(5, 9, "g:2", latitude=0.0, longitude=90.0),
        Annotation(10, 12, "g:3"),
    ]


def _partial_prediction():
    return [Annotation(0, 3, "g:9", latitude=0.0, longitude=0.0)]


def test_annotation_span_validation_pins_accepted_types():
    assert Annotation(0, 1).span == (0, 1)
    with pytest.raises(TypeError):
        Annotation(True, 4)
    with pytest.raises(TypeError):
        Annotation(1, 4.0)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="0 <= start < end"):
        Annotation(-1, 4)
    with pytest.raises(ValueError, match="0 <= start < end"):
        Annotation(4, 4)


def test_annotation_location_requires_both_or_neither_and_valid_ranges():
    assert Annotation(1, 4, latitude=0, longitude=0).latitude == 0
    with pytest.raises(ValueError, match="provided together"):
        Annotation(1, 4, latitude=0.0)
    with pytest.raises(ValueError, match="coordinates must have"):
        Annotation(1, 4, latitude=91.0, longitude=0.0)
    with pytest.raises(ValueError, match="coordinates must have"):
        Annotation(1, 4, latitude=float("nan"), longitude=0.0)
    with pytest.raises(TypeError, match="numeric coordinates"):
        Annotation(1, 4, latitude=True, longitude=0.0)


def test_toponym_annotation_accepts_duck_typed_toponyms_without_location():
    toponym = SimpleNamespace(start=1, end=4, location=None)
    assert toponym_annotation(toponym) == Annotation(1, 4)
    assert toponym_annotation(toponym, "d", with_coordinates=True) == Annotation(
        1, 4, None, "d", None, None
    )


def test_toponym_annotation_carries_coordinates_only_when_requested():
    location = SimpleNamespace(
        identifier="g:1", data={"latitude": 47.5, "longitude": 8.5}
    )
    toponym = SimpleNamespace(start=1, end=4, location=location)
    assert toponym_annotation(toponym) == Annotation(1, 4, "g:1")
    assert toponym_annotation(toponym, with_coordinates=True) == Annotation(
        1, 4, "g:1", None, 47.5, 8.5
    )


def test_toponym_annotation_reads_reference_and_feature_shaped_objects():
    feature = _Feature("g:1", {"latitude": "47.3769", "longitude": "8.5417"})
    reference = _Reference(2, 7, feature)
    assert toponym_annotation(reference, "doc", with_coordinates=True) == Annotation(
        2, 7, "g:1", "doc", 47.3769, 8.5417
    )
    unresolved = _Reference(2, 7, None)
    assert toponym_annotation(unresolved, with_coordinates=True) == Annotation(2, 7)


def test_toponym_annotation_keeps_identifier_when_coordinates_are_unusable():
    location = SimpleNamespace(identifier="g:1", data={"latitude": "unknown"})
    toponym = SimpleNamespace(start=1, end=4, location=location)
    assert toponym_annotation(toponym, with_coordinates=True) == Annotation(1, 4, "g:1")


def test_toponym_annotation_propagates_span_validation():
    with pytest.raises(TypeError):
        toponym_annotation(SimpleNamespace(start=True, end=4, location=None))


def test_toponym_annotation_requires_a_location_attribute():
    with pytest.raises(AttributeError):
        toponym_annotation(SimpleNamespace(start=1, end=4))


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (None, (None, None)),
        (SimpleNamespace(data=None), (None, None)),
        (SimpleNamespace(), (None, None)),
        (SimpleNamespace(data={}), (None, None)),
        (SimpleNamespace(data={"latitude": 1}), (None, None)),
        (SimpleNamespace(data={"latitude": 1, "longitude": 2}), (1.0, 2.0)),
        (
            SimpleNamespace(data={"latitude": "47.3769", "longitude": "8.5417"}),
            (47.3769, 8.5417),
        ),
        (SimpleNamespace(data={"latitude": True, "longitude": False}), (1.0, 0.0)),
        (SimpleNamespace(data={"latitude": [1], "longitude": 2}), (None, None)),
    ],
)
def test_location_coordinates_pins_accepted_inputs(location, expected):
    assert _location_coordinates(location) == expected


def test_record_unique_keeps_first_value_and_rejects_a_different_one():
    seen: dict = {}
    _record_unique(seen, (None, 1, 4), "g:1", "identifiers")
    _record_unique(seen, (None, 1, 4), "g:1", "identifiers")
    assert seen == {(None, 1, 4): "g:1"}
    with pytest.raises(ValueError, match=r"span \(None, 1, 4\): identifiers 'g:1'"):
        _record_unique(seen, (None, 1, 4), "g:2", "identifiers")
    assert seen == {(None, 1, 4): "g:1"}


def test_record_unique_accepts_coordinate_tuples():
    seen: dict = {}
    _record_unique(seen, (None, 0, 3), (0.0, 1.0), "coordinates")
    with pytest.raises(
        ValueError, match=r"coordinates \(0\.0, 1\.0\) and \(0\.0, 2\.0\)"
    ):
        _record_unique(seen, (None, 0, 3), (0.0, 2.0), "coordinates")


def test_recognition_metrics_pin_values_and_empty_conventions():
    expected = [Annotation(0, 3), Annotation(5, 9), Annotation(10, 12)]
    predicted = [Annotation(0, 3, "x"), Annotation(5, 9), Annotation(20, 25)]
    assert recognition_precision(expected, predicted) == 2 / 3
    assert recognition_recall(expected, predicted) == 2 / 3
    assert recognition_f1(expected, predicted) == 2 / 3
    assert recognition_f1([], []) == 1.0


def test_recognition_metrics_pin_one_sided_empty_conventions():
    assert recognition_f1([Annotation(0, 3)], []) == 0.0
    assert recognition_f1([], [Annotation(0, 3)]) == 0.0
    assert recognition_precision([Annotation(0, 3)], []) == 1.0
    assert recognition_recall([], [Annotation(0, 3)]) == 1.0


def test_recognition_metrics_ignore_identifiers_duplicates_and_accept_tuples():
    expected = (Annotation(0, 3, "g:1"), Annotation(0, 3, "g:2"))
    predicted = (Annotation(0, 3, "g:9"),)
    assert recognition_precision(expected, predicted) == 1.0
    assert recognition_recall(expected, predicted) == 1.0
    assert recognition_f1(list(expected), list(predicted)) == 1.0
    tuple_expected = (Annotation(0, 3), Annotation(5, 9), Annotation(10, 12))
    tuple_predicted = (Annotation(0, 3, "x"), Annotation(5, 9), Annotation(20, 25))
    assert recognition_precision(tuple_expected, tuple_predicted) == 2 / 3


def test_resolution_accuracy_pins_value_and_empty_convention():
    expected = [Annotation(0, 3, "g:1"), Annotation(5, 9, "g:2"), Annotation(10, 12)]
    predicted = [Annotation(0, 3, "g:1"), Annotation(5, 9, "g:9")]
    assert resolution_accuracy(expected, predicted) == 0.5
    assert resolution_accuracy([Annotation(0, 3)], []) == 1.0
    assert resolution_accuracy([Annotation(0, 3, "g:1")], [Annotation(0, 3)]) == 0.0


def test_resolution_accuracy_scopes_spans_by_document():
    gold = [Annotation(0, 3, "g:1", "d1")]
    assert resolution_accuracy(gold, [Annotation(0, 3, "g:1", "d2")]) == 0.0
    assert resolution_accuracy(gold, [Annotation(0, 3, "g:1", "d1")]) == 1.0


def test_resolution_accuracy_rejects_conflicting_gold_identifiers():
    conflicting = [Annotation(0, 3, "g:1"), Annotation(0, 3, "g:2")]
    with pytest.raises(
        ValueError,
        match=r"span \(None, 0, 3\): identifiers 'g:1' and 'g:2'",
    ):
        resolution_accuracy(conflicting, [])
    repeated = [Annotation(0, 3, "g:1"), Annotation(0, 3, "g:1")]
    assert resolution_accuracy(repeated, [Annotation(0, 3, "g:1")]) == 1.0


def test_resolution_errors_pin_distances_and_unresolved_charge():
    gold = _gold()
    assert resolution_errors_km(gold, _partial_prediction()) == [MAX_ERROR_KM, 0.0]
    assert resolution_errors_km(
        gold, _partial_prediction(), unresolved_error_km=100.0
    ) == [100.0, 0.0]


def test_resolution_errors_measure_distance_to_a_predicted_location():
    gold = [Annotation(0, 3, "g:1", latitude=0.0, longitude=0.0)]
    predicted = [Annotation(0, 3, "g:1", latitude=0.0, longitude=90.0)]
    errors = resolution_errors_km(gold, predicted)
    assert errors == pytest.approx([EARTH_QUARTER_KM], rel=1e-12)


def test_resolution_errors_reject_conflicting_gold_coordinates():
    conflicting = [
        Annotation(0, 3, latitude=0.0, longitude=0.0),
        Annotation(0, 3, latitude=0.0, longitude=1.0),
    ]
    with pytest.raises(
        ValueError,
        match=r"span \(None, 0, 3\): coordinates \(0\.0, 0\.0\) and \(0\.0, 1\.0\)",
    ):
        resolution_errors_km(conflicting, [])


@pytest.mark.parametrize("bad", [0, -1.0, True, float("nan"), float("inf")])
def test_unresolved_error_km_rejects_bool_nonfinite_and_nonpositive(bad):
    with pytest.raises(ValueError, match="unresolved_error_km"):
        resolution_errors_km(_gold(), [], unresolved_error_km=bad)


def test_unresolved_error_km_accepts_an_integer_penalty():
    assert resolution_errors_km(_gold(), [], unresolved_error_km=5) == [5, 5]


def test_accuracy_at_km_threshold_is_inclusive_and_validated():
    assert accuracy_at_km(_gold(), _partial_prediction()) == 0.5
    assert (
        accuracy_at_km(
            _gold(),
            _partial_prediction(),
            threshold_km=100.0,
            unresolved_error_km=100.0,
        )
        == 1.0
    )
    assert accuracy_at_km(_gold(), _partial_prediction(), threshold_km=0.0) == 0.5
    assert accuracy_at_km([], []) == 1.0
    with pytest.raises(ValueError, match="threshold_km"):
        accuracy_at_km([], [], threshold_km=-1.0)
    with pytest.raises(ValueError, match="threshold_km"):
        accuracy_at_km([], [], threshold_km=float("nan"))


def _four_gold_spans():
    return [
        Annotation(i * 10, i * 10 + 3, latitude=0.0, longitude=0.0) for i in range(4)
    ]


def test_mean_and_median_error_pin_even_and_odd_counts():
    gold = _four_gold_spans()
    predicted = [
        Annotation(0, 3, latitude=0.0, longitude=0.0),
        Annotation(10, 13, latitude=0.0, longitude=0.0),
    ]
    assert mean_error_km(gold, predicted, unresolved_error_km=4.0) == 2.0
    assert median_error_km(gold, predicted, unresolved_error_km=4.0) == 2.0
    odd_gold = gold[:3]
    odd_predicted = [Annotation(0, 3, latitude=0.0, longitude=0.0)]
    assert mean_error_km(odd_gold, odd_predicted, unresolved_error_km=4.0) == 8 / 3
    assert median_error_km(odd_gold, odd_predicted, unresolved_error_km=4.0) == 4.0


def test_mean_and_median_error_default_to_zero_when_nothing_is_compared():
    assert mean_error_km([], []) == 0.0
    assert median_error_km([], []) == 0.0
    assert mean_error_km([Annotation(0, 3)], []) == 0.0
    assert median_error_km([Annotation(0, 3)], []) == 0.0


def test_area_under_error_curve_pins_default_and_custom_normalisation():
    gold = _four_gold_spans()
    assert area_under_error_curve([], []) == 0.0
    assert area_under_error_curve(gold, _four_gold_spans()) == 0.0
    assert area_under_error_curve(gold, []) == 1.0
    assert area_under_error_curve(gold, [], unresolved_error_km=50000.0) == 1.0


def test_area_under_error_curve_pins_custom_unresolved_penalty():
    gold = _four_gold_spans()
    assert area_under_error_curve(gold, [], unresolved_error_km=100.0) == pytest.approx(
        0.4659156273686207, rel=1e-12
    )


def test_area_under_error_curve_default_path_keeps_legacy_rounding():
    gold = [Annotation(0, 3, "g:1", latitude=0.0, longitude=0.0)]
    # About a tenth of a millimetre away. At this size log(1 + error) and
    # log1p(error) differ in their low digits, so the default path must keep
    # the legacy formula rather than the unified one.
    predicted = [Annotation(0, 3, "g:1", latitude=1e-9, longitude=0.0)]
    assert area_under_error_curve(gold, predicted) == pytest.approx(
        1.1225605595281188e-08, rel=1e-12
    )


def test_haversine_pins_known_distances_and_accepts_integers():
    assert haversine_km(47.0, 8.0, 47.0, 8.0) == 0.0
    assert haversine_km(0, 0, 0, 90) == pytest.approx(EARTH_QUARTER_KM, rel=1e-12)
    assert haversine_km(0, 0, 0, 180) == pytest.approx(EARTH_HALF_KM, rel=1e-12)
    assert haversine_km(90, 0, -90, 0) == pytest.approx(EARTH_HALF_KM, rel=1e-12)


def test_haversine_is_symmetric_in_its_endpoints():
    assert haversine_km(0, 0, 0, 90) == pytest.approx(haversine_km(0, 90, 0, 0))


def test_public_constants_are_pinned():
    assert MAX_ERROR_KM == 20039.0
    assert ACCURACY_THRESHOLD_KM == 161.0
