import math
from dataclasses import FrozenInstanceError

import pytest

from geoparser.evaluation import (
    Annotation,
    recognition_f1,
    recognition_precision,
    recognition_recall,
    resolution_accuracy,
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
