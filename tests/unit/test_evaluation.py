import math
from dataclasses import FrozenInstanceError

import pytest

from geoparser.evaluation import (
    ACCURACY_THRESHOLD_KM,
    MAX_ERROR_KM,
    Annotation,
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
)


def test_annotation_is_an_immutable_value_object() -> None:
    annotation = Annotation(2, 8, "3041563")

    assert annotation == Annotation(2, 8, "3041563")
    assert annotation.span == (2, 8)
    with pytest.raises(FrozenInstanceError):
        field_name = "start"
        setattr(annotation, field_name, 3)


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 2), (3, 2)])
def test_annotation_rejects_invalid_character_spans(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="span"):
        Annotation(start, end)


@pytest.mark.parametrize("start,end", [(1.5, 3), (1, True)])
def test_annotation_rejects_non_integer_span_offsets(start: int, end: int) -> None:
    with pytest.raises(TypeError, match="integer"):
        Annotation(start, end)


@pytest.mark.parametrize("latitude,longitude", [(47.0, None), (None, 8.0)])
def test_annotation_requires_latitude_and_longitude_together(
    latitude: float | None, longitude: float | None
) -> None:
    with pytest.raises(ValueError, match="latitude and longitude"):
        Annotation(0, 4, latitude=latitude, longitude=longitude)


@pytest.mark.parametrize(
    "latitude,longitude",
    [
        (-90.1, 0.0),
        (90.1, 0.0),
        (0.0, -180.1),
        (0.0, 180.1),
        (math.nan, 0.0),
        (0.0, math.inf),
    ],
)
def test_annotation_rejects_out_of_range_or_non_finite_coordinates(
    latitude: float, longitude: float
) -> None:
    with pytest.raises(ValueError, match="coordinate"):
        Annotation(0, 4, latitude=latitude, longitude=longitude)


@pytest.mark.parametrize("latitude,longitude", [("47.0", 8.0), (47.0, True)])
def test_annotation_rejects_non_numeric_coordinates(
    latitude: float, longitude: float
) -> None:
    with pytest.raises(TypeError, match="coordinate"):
        Annotation(0, 4, latitude=latitude, longitude=longitude)


def test_annotation_accepts_valid_coordinate_boundaries() -> None:
    annotation = Annotation(0, 4, latitude=-90.0, longitude=180.0)

    assert annotation.latitude == -90.0
    assert annotation.longitude == 180.0


def test_annotation_identity_is_the_span_until_a_document_qualifies_it() -> None:
    unqualified = Annotation(2, 8, "3041563")
    qualified = Annotation(2, 8, "3041563", "capital")

    assert unqualified.identity == (None, 2, 8)
    assert qualified.identity == ("capital", 2, 8)
    assert qualified.span == (2, 8)


def test_identical_spans_in_different_documents_do_not_collide() -> None:
    expected = [
        Annotation(0, 6, "3040686", "encamp"),
        Annotation(0, 6, "3041204", "canillo"),
    ]
    predicted = [Annotation(0, 6, "3040686", "encamp")]

    assert recognition_recall(expected, predicted) == 0.5
    assert resolution_accuracy(expected, predicted) == 0.5


def test_predictions_are_not_credited_to_another_document() -> None:
    expected = [Annotation(0, 6, "3040686", "encamp")]
    predicted = [Annotation(0, 6, "3040686", "canillo")]

    assert recognition_precision(expected, predicted) == 0.0
    assert recognition_recall(expected, predicted) == 0.0
    assert resolution_accuracy(expected, predicted) == 0.0


@pytest.mark.parametrize(
    ("expected", "predicted", "precision", "recall", "f1"),
    [
        ([], [], 1.0, 1.0, 1.0),
        ([], [Annotation(0, 4)], 0.0, 1.0, 0.0),
        ([Annotation(0, 4)], [], 1.0, 0.0, 0.0),
        (
            [Annotation(0, 4), Annotation(10, 14)],
            [Annotation(0, 4), Annotation(20, 24)],
            0.5,
            0.5,
            0.5,
        ),
    ],
)
def test_recognition_metrics_cover_empty_and_partial_predictions(
    expected: list[Annotation],
    predicted: list[Annotation],
    precision: float,
    recall: float,
    f1: float,
) -> None:
    assert recognition_precision(expected, predicted) == precision
    assert recognition_recall(expected, predicted) == recall
    assert recognition_f1(expected, predicted) == f1


def test_precision_deduplicates_predicted_spans() -> None:
    expected = [Annotation(0, 4)]
    predicted = [Annotation(0, 4), Annotation(0, 4), Annotation(9, 13)]

    assert recognition_precision(expected, predicted) == 0.5


def test_recall_deduplicates_expected_spans() -> None:
    expected = [Annotation(0, 4), Annotation(0, 4), Annotation(9, 13)]
    predicted = [Annotation(0, 4)]

    assert recognition_recall(expected, predicted) == 0.5


def test_f1_is_zero_when_nonempty_inputs_have_no_overlap() -> None:
    expected = [Annotation(0, 4)]
    predicted = [Annotation(9, 13)]

    assert recognition_f1(expected, predicted) == 0.0


def test_recognition_ignores_identifiers() -> None:
    expected = [Annotation(0, 4, "expected")]
    predicted = [Annotation(0, 4, "predicted")]

    assert recognition_precision(expected, predicted) == 1.0
    assert recognition_recall(expected, predicted) == 1.0


def test_resolution_accuracy_requires_the_expected_identifier() -> None:
    expected = [Annotation(0, 16, "3041563")]

    assert resolution_accuracy(expected, [Annotation(0, 16, "3041563")]) == 1.0
    assert resolution_accuracy(expected, [Annotation(0, 16, "3041564")]) == 0.0
    assert resolution_accuracy(expected, [Annotation(0, 16)]) == 0.0


def test_resolution_accuracy_counts_each_gold_pair_once() -> None:
    expected = [Annotation(0, 16, "3041563")]
    predicted = [
        Annotation(0, 16, "3041563"),
        Annotation(0, 16, "3041563"),
        Annotation(30, 36, "9999999"),
    ]

    assert resolution_accuracy(expected, predicted) == 1.0


def test_resolution_accuracy_rejects_conflicting_gold_identifiers_for_one_span() -> (
    None
):
    expected = [
        Annotation(0, 16, "3041563", "doc"),
        Annotation(0, 16, "3041564", "doc"),
    ]

    with pytest.raises(ValueError, match="conflicting gold annotations"):
        resolution_accuracy(expected, [])


def test_resolution_accuracy_is_one_without_resolvable_gold_annotations() -> None:
    assert resolution_accuracy([], [Annotation(0, 4, "3041563")]) == 1.0


# Exact contracts pinned for mutation testing.


@pytest.mark.parametrize(
    ("latitude", "longitude", "error", "message"),
    [
        (
            "47",
            8.0,
            TypeError,
            "latitude and longitude must be numeric coordinates",
        ),
        (
            91.0,
            8.0,
            ValueError,
            "coordinates must have latitude in [-90, 90] and longitude in [-180, 180]",
        ),
    ],
)
def test_coordinate_errors_are_exact(latitude, longitude, error, message) -> None:
    with pytest.raises(error) as raised:
        Annotation(0, 4, latitude=latitude, longitude=longitude)

    assert str(raised.value) == message


@pytest.mark.parametrize(("latitude", "longitude"), [(90.0, 0.0), (0.0, -180.0)])
def test_the_closed_coordinate_bounds_are_accepted(latitude, longitude) -> None:
    Annotation(0, 4, latitude=latitude, longitude=longitude)


def test_conflicting_gold_identifiers_are_named_exactly() -> None:
    gold = [Annotation(0, 4, identifier="a"), Annotation(0, 4, identifier="b")]

    with pytest.raises(ValueError) as raised:
        resolution_accuracy(gold, [])

    assert str(raised.value) == (
        "conflicting gold annotations for span (None, 0, 4): identifiers 'a' and 'b'"
    )


def test_conflicting_gold_coordinates_are_named_exactly() -> None:
    gold = [
        Annotation(0, 4, latitude=1.0, longitude=2.0),
        Annotation(0, 4, latitude=3.0, longitude=4.0),
    ]

    with pytest.raises(ValueError) as raised:
        accuracy_at_km(gold, [])

    assert str(raised.value) == (
        "conflicting gold annotations for span (None, 0, 4): "
        "coordinates (1.0, 2.0) and (3.0, 4.0)"
    )


def test_invalid_unresolved_penalty_is_named_exactly() -> None:
    with pytest.raises(ValueError) as raised:
        accuracy_at_km([], [], unresolved_error_km=0)

    assert str(raised.value) == "unresolved_error_km must be a finite positive number"


def test_invalid_threshold_is_named_exactly() -> None:
    with pytest.raises(ValueError) as raised:
        accuracy_at_km([], [], threshold_km=-1)

    assert str(raised.value) == "threshold_km must be a finite non-negative number"


def test_a_zero_km_threshold_counts_exact_placements() -> None:
    gold = [Annotation(0, 4, latitude=1.0, longitude=2.0)]

    assert accuracy_at_km(gold, gold, threshold_km=0) == 1.0


def test_area_under_error_curve_averages_over_every_toponym() -> None:
    gold = [
        Annotation(0, 4, latitude=0.0, longitude=0.0),
        Annotation(5, 9, latitude=0.0, longitude=0.0),
    ]
    predicted = [Annotation(0, 4, latitude=0.0, longitude=0.0)]

    assert area_under_error_curve(gold, predicted) == pytest.approx(0.5)


def test_custom_penalty_curve_averages_over_every_toponym() -> None:
    gold = [
        Annotation(0, 4, latitude=0.0, longitude=0.0),
        Annotation(5, 9, latitude=0.0, longitude=0.0),
    ]
    predicted = [Annotation(0, 4, latitude=0.0, longitude=0.0)]

    assert area_under_error_curve(
        gold, predicted, unresolved_error_km=50_000
    ) == pytest.approx(0.5)


def test_a_half_located_prediction_is_not_treated_as_located() -> None:
    """Defensive: only annotations carrying both coordinates are placed."""
    gold = [Annotation(0, 4, latitude=0.0, longitude=0.0)]
    half = Annotation(0, 4, latitude=0.0, longitude=0.0)
    object.__setattr__(half, "longitude", None)

    assert accuracy_at_km(gold, [half]) == 0.0


ZURICH = (47.3769, 8.5417)


GENEVA = (46.2044, 6.1432)


def located(start, end, latitude, longitude, document="d"):
    """Build an annotation that places a span at a point."""
    return Annotation(start, end, None, document, latitude, longitude)


class TestHaversine:
    """The great-circle distance underlying every other metric."""

    def test_returns_zero_for_one_point(self):
        """Test that a point is no distance from itself."""
        assert haversine_km(*ZURICH, *ZURICH) == 0.0

    def test_matches_a_known_distance(self):
        """Test Zurich to Geneva, which is about 224 km."""
        assert haversine_km(*ZURICH, *GENEVA) == pytest.approx(224.4, abs=1.0)

    def test_is_symmetric(self):
        """Test that distance does not depend on argument order."""
        assert haversine_km(*ZURICH, *GENEVA) == pytest.approx(
            haversine_km(*GENEVA, *ZURICH)
        )

    def test_handles_antipodes(self):
        """Test the clamp that keeps rounding from pushing asin out of domain."""
        half_circumference = math.pi * 6371.0088
        assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(
            half_circumference, rel=1e-9
        )


class TestResolutionErrors:
    """How gold spans are paired with predictions and charged for misses."""

    def test_pairs_a_prediction_with_its_gold_span(self):
        """Test that a matched span is scored by its distance."""
        gold = [located(0, 6, *ZURICH)]
        predicted = [located(0, 6, 47.36667, 8.55)]

        assert resolution_errors_km(gold, predicted) == pytest.approx([1.3], abs=0.5)

    def test_charges_the_maximum_for_an_unrecognized_span(self):
        """Test that a gold toponym nothing predicted costs the maximum."""
        assert resolution_errors_km([located(0, 6, *ZURICH)], []) == [MAX_ERROR_KM]

    def test_charges_the_maximum_for_an_unresolved_span(self):
        """Test that recognizing a span without placing it earns no credit."""
        gold = [located(0, 6, *ZURICH)]
        predicted = [Annotation(0, 6, None, "d")]

        assert resolution_errors_km(gold, predicted) == [MAX_ERROR_KM]

    def test_rejects_a_prediction_with_only_one_coordinate(self):
        """A placement must supply both coordinates or neither."""
        with pytest.raises(ValueError, match="latitude and longitude"):
            Annotation(0, 6, None, "d", 47.3769, None)

    def test_keeps_documents_apart(self):
        """Test that the same offsets in two documents are different spans."""
        gold = [located(0, 6, *ZURICH, document="a")]
        predicted = [located(0, 6, *ZURICH, document="b")]

        assert resolution_errors_km(gold, predicted) == [MAX_ERROR_KM]

    def test_ignores_gold_spans_without_coordinates(self):
        """Test that only located gold spans are scored."""
        assert resolution_errors_km([Annotation(0, 6, None, "d")], []) == []

    def test_returns_the_worst_error_first(self):
        """Test the ordering the report and the median rely on."""
        gold = [located(0, 1, *ZURICH), located(2, 3, *ZURICH)]
        predicted = [located(0, 1, *ZURICH), located(2, 3, *GENEVA)]

        errors = resolution_errors_km(gold, predicted)

        assert errors == sorted(errors, reverse=True)

    def test_honours_a_custom_unresolved_error(self):
        """Test that the miss penalty is configurable."""
        assert resolution_errors_km(
            [located(0, 6, *ZURICH)], [], unresolved_error_km=5.0
        ) == [5.0]


class TestAccuracyAtKm:
    """The fraction of gold toponyms placed close enough."""

    def test_counts_a_close_prediction(self):
        """Test that a prediction inside the threshold counts."""
        gold = [located(0, 6, *ZURICH)]

        assert accuracy_at_km(gold, [located(0, 6, 47.36667, 8.55)]) == 1.0

    def test_rejects_a_distant_prediction(self):
        """Test that Geneva is not Zurich at the conventional threshold."""
        assert accuracy_at_km([located(0, 6, *ZURICH)], [located(0, 6, *GENEVA)]) == 0.0

    def test_counts_a_prediction_exactly_on_the_threshold(self):
        """Test that the boundary is inclusive."""
        gold = [located(0, 6, 0.0, 0.0)]
        degrees = ACCURACY_THRESHOLD_KM / (math.pi * 6371.0088 / 180)

        assert accuracy_at_km(gold, [located(0, 6, 0.0, degrees)]) == 1.0

    def test_treats_an_empty_comparison_as_perfect(self):
        """Test that scoring nothing is not scored as failure."""
        assert accuracy_at_km([], []) == 1.0

    def test_honours_a_custom_threshold(self):
        """Test that a stricter threshold rejects what the default accepts."""
        gold = [located(0, 6, *ZURICH)]
        predicted = [located(0, 6, 47.36667, 8.55)]

        assert accuracy_at_km(gold, predicted, threshold_km=0.5) == 0.0

    def test_rejects_a_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold_km"):
            accuracy_at_km([], [], threshold_km=-0.1)

    @pytest.mark.parametrize("threshold_km", [math.nan, math.inf])
    def test_rejects_a_non_finite_threshold(self, threshold_km):
        with pytest.raises(ValueError, match="threshold_km"):
            accuracy_at_km([], [], threshold_km=threshold_km)


class TestErrorSummaries:
    """The mean and median, which are reported together on purpose."""

    def test_mean_averages_the_errors(self):
        """Test the mean over two known errors."""
        gold = [located(0, 1, 0.0, 0.0), located(2, 3, 0.0, 0.0)]
        predicted = [located(0, 1, 0.0, 0.0), located(2, 3, 0.0, 1.0)]
        one_degree = haversine_km(0.0, 0.0, 0.0, 1.0)

        assert mean_error_km(gold, predicted) == pytest.approx(one_degree / 2)

    def test_mean_of_nothing_is_zero(self):
        """Test that an empty comparison does not divide by zero."""
        assert mean_error_km([], []) == 0.0

    def test_median_of_an_odd_count_is_the_middle_error(self):
        """Test the middle of three errors."""
        gold = [located(i, i + 1, 0.0, 0.0) for i in range(3)]
        predicted = [
            located(0, 1, 0.0, 0.0),
            located(1, 2, 0.0, 1.0),
            located(2, 3, 0.0, 2.0),
        ]

        assert median_error_km(gold, predicted) == pytest.approx(
            haversine_km(0.0, 0.0, 0.0, 1.0)
        )

    def test_median_of_an_even_count_averages_the_middle_pair(self):
        """Test that an even count interpolates."""
        gold = [located(i, i + 1, 0.0, 0.0) for i in range(2)]
        predicted = [located(0, 1, 0.0, 0.0), located(1, 2, 0.0, 1.0)]

        assert median_error_km(gold, predicted) == pytest.approx(
            haversine_km(0.0, 0.0, 0.0, 1.0) / 2
        )

    def test_median_of_nothing_is_zero(self):
        """Test that an empty comparison does not index an empty list."""
        assert median_error_km([], []) == 0.0


class TestAreaUnderErrorCurve:
    """The single-number summary of the whole error distribution."""

    def test_is_zero_when_every_toponym_is_exact(self):
        """Test the best attainable value."""
        gold = [located(0, 6, *ZURICH)]

        assert area_under_error_curve(gold, [located(0, 6, *ZURICH)]) == 0.0

    def test_is_one_when_nothing_is_resolved(self):
        """Test the worst attainable value."""
        assert area_under_error_curve([located(0, 6, *ZURICH)], []) == 1.0

    def test_is_zero_for_an_empty_comparison(self):
        """Test that an empty comparison does not divide by zero."""
        assert area_under_error_curve([], []) == 0.0

    def test_rewards_a_closer_prediction(self):
        """Test that the summary moves the right way."""
        gold = [located(0, 6, *ZURICH)]
        near = area_under_error_curve(gold, [located(0, 6, 47.36667, 8.55)])
        far = area_under_error_curve(gold, [located(0, 6, *GENEVA)])

        assert 0.0 < near < far < 1.0

    def test_default_auc_preserves_the_pre_73_sub_ulp_golden_value(self):
        """Tiny nonzero errors retain the legacy default's exact rounding."""
        gold = [located(0, 1, 0.0, 0.0)]
        predicted = [located(0, 1, 0.0, 1e-19)]

        assert area_under_error_curve(gold, predicted) == 0.0

    def test_compresses_the_scale_logarithmically(self):
        """Test that a tenfold worse error is not a tenfold worse score."""
        gold = [located(0, 1, 0.0, 0.0)]
        ten_km = area_under_error_curve(gold, [located(0, 1, 0.0, 0.0898)])
        hundred_km = area_under_error_curve(gold, [located(0, 1, 0.0, 0.898)])

        assert hundred_km < 10 * ten_km

    def test_custom_miss_penalty_does_not_clip_resolved_errors(self):
        """Only an unresolved toponym receives the configured miss penalty."""
        gold = [located(0, 6, *ZURICH)]

        score = area_under_error_curve(
            gold, [located(0, 6, *GENEVA)], unresolved_error_km=100.0
        )

        distance = haversine_km(*ZURICH, *GENEVA)
        assert score == pytest.approx(math.log1p(distance) / math.log1p(MAX_ERROR_KM))

    def test_custom_miss_penalty_does_not_rescale_the_auc(self):
        """The configured miss cost uses the fixed maximum-error scale."""
        gold = [located(0, 6, *ZURICH)]
        unresolved_error_km = 1000.0

        score = area_under_error_curve(
            gold, [], unresolved_error_km=unresolved_error_km
        )

        assert score == pytest.approx(
            math.log1p(unresolved_error_km) / math.log1p(MAX_ERROR_KM)
        )
        resolved_default = area_under_error_curve(gold, [located(0, 6, *GENEVA)])
        resolved_custom = area_under_error_curve(
            gold, [located(0, 6, *GENEVA)], unresolved_error_km=unresolved_error_km
        )
        assert resolved_custom == resolved_default

    def test_handles_the_smallest_positive_miss_penalty(self):
        """A valid tiny penalty still has a finite logarithmic normalization."""
        gold = [located(0, 6, *ZURICH)]

        score = area_under_error_curve(gold, [], unresolved_error_km=5e-324)

        assert score == pytest.approx(math.log1p(5e-324) / math.log1p(MAX_ERROR_KM))


class TestMutationPins:
    """Edges of the arithmetic that a plausible slip would get wrong."""

    def test_rounding_past_one_does_not_leave_the_asin_domain(self, monkeypatch):
        """
        A haversine term rounded just past 1 is clamped, not passed to asin.

        Whether a real antipodal pair rounds past 1 depends on the platform's
        libm, so the rounding is forced: sin is made to overshoot by 1e-15.
        """
        import types

        import geoparser.evaluation as evaluation

        overshooting = types.SimpleNamespace(
            **{
                name: getattr(math, name) for name in ("radians", "cos", "asin", "sqrt")
            },
            sin=lambda x: math.sin(x) * (1 + 1e-15),
        )
        monkeypatch.setattr(evaluation, "math", overshooting)

        assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(
            math.pi * 6371.0088, rel=1e-9
        )

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            (mean_error_km, 100.0),
            (median_error_km, 100.0),
            (lambda e, p, **k: accuracy_at_km(e, p, **k), 1.0),
            (
                area_under_error_curve,
                math.log1p(100.0) / math.log1p(MAX_ERROR_KM),
            ),
        ],
    )
    def test_every_summary_passes_on_a_custom_unresolved_error(self, metric, expected):
        """An unplaced toponym costs the configured error, not the maximum."""
        gold = [located(0, 1, *ZURICH)]

        assert metric(gold, [], unresolved_error_km=100.0) == pytest.approx(expected)

    def test_median_of_four_averages_the_two_middle_errors(self):
        """Sorted errors [U, 224, 224, 0] have a median of 224."""
        gold = [located(i, i + 1, *ZURICH) for i in range(4)]
        predicted = [
            located(1, 2, *GENEVA),
            located(2, 3, *GENEVA),
            located(3, 4, *ZURICH),
        ]

        assert median_error_km(gold, predicted) == pytest.approx(
            haversine_km(*ZURICH, *GENEVA)
        )

    def test_area_averages_over_every_toponym(self):
        """Two errors average; they are not multiplied by the count."""
        gold = [located(0, 1, *ZURICH), located(1, 2, *ZURICH)]
        predicted = [located(0, 1, *ZURICH)]

        assert area_under_error_curve(gold, predicted) == pytest.approx(0.5)

    @pytest.mark.parametrize("unresolved_error_km", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite_unresolved_error(
        self, unresolved_error_km
    ):
        with pytest.raises(ValueError, match="unresolved_error_km"):
            resolution_errors_km([], [], unresolved_error_km=unresolved_error_km)

    def test_rejects_conflicting_gold_locations_for_one_span(self):
        """One gold span cannot silently select whichever location came last."""
        gold = [located(0, 6, *ZURICH), located(0, 6, *GENEVA)]

        with pytest.raises(ValueError, match="conflicting gold annotations"):
            resolution_errors_km(gold, [])

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("errors", pytest.approx([MAX_ERROR_KM, 224.3513426985906, 0.0])),
            ("accuracy", pytest.approx(1 / 3)),
            ("mean", pytest.approx(6754.450447566197)),
            ("median", pytest.approx(224.3513426985906)),
            ("area", 0.5156451334367401),
        ],
    )
    def test_default_distance_metrics_preserve_the_legacy_golden_values(
        self, default_distance_metric_results, metric, expected
    ):
        """Pin each default score while custom miss penalties gain a bounded AUC."""
        assert default_distance_metric_results[metric] == expected


@pytest.fixture
def default_distance_metric_results():
    """Calculate all legacy distance scores from the same annotation rows."""
    gold = [
        located(0, 6, *ZURICH),
        located(7, 13, *ZURICH),
        located(14, 20, *ZURICH),
    ]
    predicted = [located(0, 6, *ZURICH), located(7, 13, *GENEVA)]
    return {
        "errors": resolution_errors_km(gold, predicted),
        "accuracy": accuracy_at_km(gold, predicted),
        "mean": mean_error_km(gold, predicted),
        "median": median_error_km(gold, predicted),
        "area": area_under_error_curve(gold, predicted),
    }
