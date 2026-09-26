"""Pure evaluation metrics for recognition and resolution pilots."""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Annotation:
    """A character span with an optional gazetteer identifier."""

    start: int
    end: int
    identifier: str | None = None
    document_id: str | None = None
    """The document this span belongs to, when annotations from several are
    compared at once. Spans only identify a place within one document, so
    without it the same offsets in two documents are the same annotation."""
    latitude: float | None = None
    longitude: float | None = None
    """Where the annotation places the span. Gold data often gives coordinates
    without a gazetteer identifier, and two gazetteers disagree on identifiers
    for the same place, so distance is what compares across them."""

    def __post_init__(self) -> None:
        """Reject spans and locations that cannot describe an annotation."""
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("start offset must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("end offset must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("annotation span must satisfy 0 <= start < end")

        latitude = self.latitude
        longitude = self.longitude
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if latitude is None or longitude is None:
            return
        if (
            isinstance(latitude, bool)
            or not isinstance(latitude, (int, float))
            or isinstance(longitude, bool)
            or not isinstance(longitude, (int, float))
        ):
            raise TypeError("latitude and longitude must be numeric coordinates")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(
                "coordinates must have latitude in [-90, 90] and longitude in [-180, 180]"
            )
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("coordinates must be finite")

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open character span."""
        return self.start, self.end

    @property
    def identity(self) -> tuple[str | None, int, int]:
        """Return the span qualified by its document, if one was given."""
        return self.document_id, self.start, self.end


Identity = tuple[str | None, int, int]


def _unique_spans(annotations: Sequence[Annotation]) -> set[Identity]:
    """Return distinct spans, ignoring any resolution identifiers."""
    return {annotation.identity for annotation in annotations}


def _resolved_pairs(
    annotations: Sequence[Annotation],
) -> set[tuple[Identity, str]]:
    """Return distinct spans paired with their available identifiers."""
    return {
        (annotation.identity, annotation.identifier)
        for annotation in annotations
        if annotation.identifier is not None
    }


def _validate_gold_annotations(annotations: Sequence[Annotation]) -> None:
    """Reject one gold span assigned multiple identifiers or locations."""
    identifiers: dict[Identity, str] = {}
    locations: dict[Identity, tuple[float, float]] = {}
    for annotation in annotations:
        identity = annotation.identity
        if annotation.identifier is not None:
            previous_identifier = identifiers.get(identity)
            if (
                previous_identifier is not None
                and previous_identifier != annotation.identifier
            ):
                raise ValueError(
                    "conflicting gold annotations for span "
                    f"{identity}: identifiers {previous_identifier!r} and "
                    f"{annotation.identifier!r}"
                )
            identifiers[identity] = annotation.identifier

        if annotation.latitude is not None and annotation.longitude is not None:
            location = (annotation.latitude, annotation.longitude)
            previous_location = locations.get(identity)
            if previous_location is not None and previous_location != location:
                raise ValueError(
                    "conflicting gold annotations for span "
                    f"{identity}: coordinates {previous_location!r} and {location!r}"
                )
            locations[identity] = location


def _validate_unresolved_error_km(value: float) -> None:
    """Require a finite, positive penalty for an unplaced gold span."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise ValueError("unresolved_error_km must be a finite positive number")


def _ratio(numerator: int, denominator: int) -> float:
    """Return a bounded ratio, treating an empty comparison as perfect."""
    return numerator / denominator if denominator else 1.0


def recognition_precision(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> float:
    """Measure the fraction of predicted spans that are expected."""
    expected_spans = _unique_spans(expected)
    predicted_spans = _unique_spans(predicted)
    true_positives = len(expected_spans & predicted_spans)
    return _ratio(true_positives, len(predicted_spans))


def recognition_recall(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> float:
    """Measure the fraction of expected spans that were predicted."""
    expected_spans = _unique_spans(expected)
    predicted_spans = _unique_spans(predicted)
    true_positives = len(expected_spans & predicted_spans)
    return _ratio(true_positives, len(expected_spans))


def recognition_f1(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> float:
    """Return the harmonic mean of recognition precision and recall."""
    precision = recognition_precision(expected, predicted)
    recall = recognition_recall(expected, predicted)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def resolution_accuracy(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> float:
    """Measure exact span-and-identifier matches against resolved gold pairs."""
    _validate_gold_annotations(expected)
    expected_pairs = _resolved_pairs(expected)
    predicted_pairs = _resolved_pairs(predicted)
    correct = len(expected_pairs & predicted_pairs)
    return _ratio(correct, len(expected_pairs))


# Half the Earth's circumference: no two points on the surface are further
# apart, so this is the error assigned to a toponym a pipeline did not resolve.
# Scoring those as "no error" would reward a resolver for staying silent.
MAX_ERROR_KM = 20039.0

# The conventional threshold in the toponym resolution literature, and the
# distance below which a prediction is normally counted as the right place.
ACCURACY_THRESHOLD_KM = 161.0

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    latitude: float, longitude: float, other_latitude: float, other_longitude: float
) -> float:
    """
    Return the great-circle distance between two points in kilometres.

    Args:
        latitude: Latitude of the first point in degrees
        longitude: Longitude of the first point in degrees
        other_latitude: Latitude of the second point in degrees
        other_longitude: Longitude of the second point in degrees

    Returns:
        The distance along the Earth's surface in kilometres
    """
    phi, other_phi = math.radians(latitude), math.radians(other_latitude)
    delta_phi = other_phi - phi
    delta_lambda = math.radians(other_longitude - longitude)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi) * math.cos(other_phi) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def _located(
    annotations: Sequence[Annotation],
) -> dict[Identity, tuple[float, float]]:
    """Return the coordinates of every annotation that carries them, by span."""
    return {
        annotation.identity: (annotation.latitude, annotation.longitude)
        for annotation in annotations
        if annotation.latitude is not None and annotation.longitude is not None
    }


def resolution_errors_km(
    expected: Sequence[Annotation],
    predicted: Sequence[Annotation],
    *,
    unresolved_error_km: float = MAX_ERROR_KM,
) -> list[float]:
    """
    Return one distance error per located gold annotation, worst first.

    A gold toponym the pipeline did not place -- because it recognized no such
    span, or recognized it and resolved nothing -- scores
    ``unresolved_error_km`` rather than being dropped, so that a pipeline
    cannot improve its score by resolving less.

    Args:
        expected: Gold annotations, of which those carrying coordinates count
        predicted: Annotations produced by the pipeline
        unresolved_error_km: Error charged for a gold toponym left unplaced

    Returns:
        Distance errors in kilometres, in descending order
    """
    _validate_unresolved_error_km(unresolved_error_km)
    _validate_gold_annotations(expected)
    predictions = _located(predicted)
    errors = [
        haversine_km(latitude, longitude, *predictions[identity])
        if identity in predictions
        else unresolved_error_km
        for identity, (latitude, longitude) in _located(expected).items()
    ]
    return sorted(errors, reverse=True)


def accuracy_at_km(
    expected: Sequence[Annotation],
    predicted: Sequence[Annotation],
    *,
    threshold_km: float = ACCURACY_THRESHOLD_KM,
    unresolved_error_km: float = MAX_ERROR_KM,
) -> float:
    """
    Measure the fraction of gold toponyms placed within ``threshold_km``.

    Args:
        expected: Gold annotations carrying coordinates
        predicted: Annotations produced by the pipeline
        threshold_km: Distance within which a prediction counts as correct
        unresolved_error_km: Error charged for a gold toponym left unplaced

    Returns:
        The fraction placed close enough, where an empty comparison is perfect
    """
    if (
        isinstance(threshold_km, bool)
        or not isinstance(threshold_km, (int, float))
        or threshold_km < 0
        or (isinstance(threshold_km, float) and not math.isfinite(threshold_km))
    ):
        raise ValueError("threshold_km must be a finite non-negative number")
    errors = resolution_errors_km(
        expected, predicted, unresolved_error_km=unresolved_error_km
    )
    return _ratio(sum(1 for error in errors if error <= threshold_km), len(errors))


def mean_error_km(
    expected: Sequence[Annotation],
    predicted: Sequence[Annotation],
    *,
    unresolved_error_km: float = MAX_ERROR_KM,
) -> float:
    """Return the mean distance error, or 0.0 when nothing is compared."""
    errors = resolution_errors_km(
        expected, predicted, unresolved_error_km=unresolved_error_km
    )
    return sum(errors) / len(errors) if errors else 0.0


def median_error_km(
    expected: Sequence[Annotation],
    predicted: Sequence[Annotation],
    *,
    unresolved_error_km: float = MAX_ERROR_KM,
) -> float:
    """
    Return the median distance error, or 0.0 when nothing is compared.

    The median is reported alongside the mean because a handful of
    hemisphere-scale mistakes dominate the mean on any real corpus.
    """
    errors = resolution_errors_km(
        expected, predicted, unresolved_error_km=unresolved_error_km
    )
    if not errors:
        return 0.0
    middle = len(errors) // 2
    if len(errors) % 2:
        return errors[middle]
    return (errors[middle - 1] + errors[middle]) / 2


def area_under_error_curve(
    expected: Sequence[Annotation],
    predicted: Sequence[Annotation],
    *,
    unresolved_error_km: float = MAX_ERROR_KM,
) -> float:
    """
    Summarize the whole error distribution as one number, lower being better.

    Errors are capped at ``unresolved_error_km``, compressed with
    ``ln(1 + error)``, normalized by ``ln(1 + unresolved_error_km)``, and
    averaged. The default penalty is ``MAX_ERROR_KM``, above every distance
    between valid coordinates. The log scale is what makes the number
    informative: on a linear scale a few hemisphere-scale mistakes would
    drown out every difference between a 5 km and a 500 km error, which is
    the range an improvement actually moves.

    This follows the shape of the AUC used in the toponym resolution
    literature, but the exact normalization here is this repository's own --
    compare runs of this harness against each other, not against published
    AUC figures.

    Args:
        expected: Gold annotations carrying coordinates
        predicted: Annotations produced by the pipeline
        unresolved_error_km: Error charged for a gold toponym left unplaced

    Returns:
        A value in [0, 1], where 0.0 is every toponym placed exactly
    """
    errors = resolution_errors_km(
        expected, predicted, unresolved_error_km=unresolved_error_km
    )
    if not errors:
        return 0.0
    if unresolved_error_km == MAX_ERROR_KM:
        # Preserve the legacy default's exact floating-point rounding.
        ceiling = math.log(1 + unresolved_error_km)
        return sum(
            math.log(1 + min(error, unresolved_error_km)) / ceiling for error in errors
        ) / len(errors)

    ceiling = math.log1p(unresolved_error_km)
    return sum(
        math.log1p(min(error, unresolved_error_km)) / ceiling for error in errors
    ) / len(errors)
