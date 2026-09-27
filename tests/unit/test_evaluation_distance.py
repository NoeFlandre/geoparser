"""
Tests for the distance-based resolution metrics.

Gold geoparsing corpora normally give coordinates rather than gazetteer
identifiers, and two gazetteers disagree on identifiers for the same place, so
distance is what compares a resolver against gold data or against another
resolver. These tests pin the arithmetic and, more importantly, the scoring
convention: a toponym a pipeline declined to place costs it the maximum error,
so resolving less can never improve a score.
"""

import math

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
    resolution_errors_km,
)

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

    def test_default_distance_metrics_preserve_the_legacy_golden_values(self):
        """Pin default scores while custom miss penalties gain a bounded AUC."""
        gold = [
            located(0, 6, *ZURICH),
            located(7, 13, *ZURICH),
            located(14, 20, *ZURICH),
        ]
        predicted = [located(0, 6, *ZURICH), located(7, 13, *GENEVA)]

        assert resolution_errors_km(gold, predicted) == pytest.approx(
            [MAX_ERROR_KM, 224.3513426985906, 0.0]
        )
        assert accuracy_at_km(gold, predicted) == pytest.approx(1 / 3)
        assert mean_error_km(gold, predicted) == pytest.approx(6754.450447566197)
        assert median_error_km(gold, predicted) == pytest.approx(224.3513426985906)
        assert area_under_error_curve(gold, predicted) == 0.5156451334367401
