"""Pure evaluation metrics for recognition and resolution pilots."""

import math
import typing as t
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence
from dataclasses import dataclass


def _is_integer(value: object) -> bool:
    """Whether a value is an int, excluding bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """Whether a value is an int or float, excluding bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    """Whether a value is a finite int or float, excluding bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_coordinates(latitude: float, longitude: float) -> None:
    """Require numeric coordinates within the WGS 84 ranges."""
    if not (_is_number(latitude) and _is_number(longitude)):
        msg = "latitude and longitude must be numeric coordinates"
        raise TypeError(msg)
    # NaN compares false and infinities fall outside the range, so this
    # also rejects non-finite coordinates.
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        msg = "coordinates must have latitude in [-90, 90] and longitude in [-180, 180]"
        raise ValueError(msg)


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
        self._validate_span()
        self._validate_location()

    def _validate_span(self) -> None:
        """Require integer offsets that bound a non-empty span."""
        if not _is_integer(self.start):
            msg = "start offset must be an integer"
            raise TypeError(msg)
        if not _is_integer(self.end):
            msg = "end offset must be an integer"
            raise TypeError(msg)
        if self.start < 0 or self.end <= self.start:
            msg = "annotation span must satisfy 0 <= start < end"
            raise ValueError(msg)

    def _validate_location(self) -> None:
        """Require both coordinates or neither, and valid ones when given."""
        latitude = self.latitude
        longitude = self.longitude
        if (latitude is None) != (longitude is None):
            msg = "latitude and longitude must be provided together"
            raise ValueError(msg)
        if latitude is None or longitude is None:
            return
        _validate_coordinates(latitude, longitude)

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open character span."""
        return self.start, self.end

    @property
    def identity(self) -> tuple[str | None, int, int]:
        """Return the span qualified by its document, if one was given."""
        return self.document_id, self.start, self.end


Identity = tuple[str | None, int, int]


class _ResolvedLocation(t.Protocol):
    """A place a toponym resolved to: its identifier and gazetteer record."""

    @property
    def identifier(self) -> str | None: ...

    # Record values are JSON, so their type is left open here: the coordinate
    # conversion in _location_coordinates is what checks them at runtime.
    @property
    def data(self) -> _Mapping[str, t.Any]: ...


class _ParsedToponym(t.Protocol):
    """A recognised toponym: its character span and the place it resolved to."""

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...

    @property
    def location(self) -> _ResolvedLocation | None: ...


def _location_coordinates(
    location: _ResolvedLocation | None,
) -> tuple[float | None, float | None]:
    """Return a resolved location's coordinates, when it has usable ones."""
    if location is None:
        return None, None
    try:
        return float(location.data["latitude"]), float(location.data["longitude"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, None


def toponym_annotation(
    toponym: _ParsedToponym,
    document_id: str | None = None,
    *,
    with_coordinates: bool = False,
) -> Annotation:
    """
    Convert a parsed toponym into an evaluation annotation.

    Args:
        toponym: A parsed toponym with ``start``, ``end`` and ``location``
        document_id: The document the span belongs to, if comparing corpus-wide
        with_coordinates: Also carry the resolved location's coordinates

    Returns:
        The annotation, with the resolved identifier or ``None``
    """
    location = toponym.location
    latitude, longitude = (
        _location_coordinates(location) if with_coordinates else (None, None)
    )
    return Annotation(
        toponym.start,
        toponym.end,
        location.identifier if location is not None else None,
        document_id,
        latitude,
        longitude,
    )


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


_V = t.TypeVar("_V")


def _record_unique(
    seen: dict[Identity, _V], identity: Identity, value: _V, kind: str
) -> None:
    """Remember a span's value, rejecting a different one for the same span."""
    previous = seen.get(identity)
    if previous is not None and previous != value:
        msg = (
            "conflicting gold annotations for span "
            f"{identity}: {kind} {previous!r} and {value!r}"
        )
        raise ValueError(msg)
    seen[identity] = value


def _validate_gold_annotations(annotations: Sequence[Annotation]) -> None:
    """Reject one gold span assigned multiple identifiers or locations."""
    identifiers: dict[Identity, str] = {}
    locations: dict[Identity, tuple[float, float]] = {}
    for annotation in annotations:
        if annotation.identifier is not None:
            _record_unique(
                identifiers, annotation.identity, annotation.identifier, "identifiers"
            )
        # pragma: no mutate start - Annotation guarantees both coordinates or
        # neither, so `or` here would behave identically.
        if annotation.latitude is not None and annotation.longitude is not None:
            # pragma: no mutate end
            location = (annotation.latitude, annotation.longitude)
            _record_unique(locations, annotation.identity, location, "coordinates")


def _validate_unresolved_error_km(value: float) -> None:
    """Require a finite, positive penalty for an unplaced gold span."""
    if not (_is_finite_number(value) and value > 0):
        msg = "unresolved_error_km must be a finite positive number"
        raise ValueError(msg)


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
    if not (_is_finite_number(threshold_km) and threshold_km >= 0):
        msg = "threshold_km must be a finite non-negative number"
        raise ValueError(msg)
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

    Errors are capped at ``MAX_ERROR_KM``, compressed with
    ``ln(1 + error)``, normalized by ``ln(1 + MAX_ERROR_KM)``, and averaged.
    An unplaced toponym receives ``unresolved_error_km`` before that cap. The
    log scale makes the number informative: on a linear scale a few
    hemisphere-scale mistakes would drown out every difference between a 5 km
    and a 500 km error, which is the range an improvement actually moves.

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
        ceiling = math.log(1 + MAX_ERROR_KM)
        return sum(
            math.log(1 + min(error, MAX_ERROR_KM)) / ceiling for error in errors
        ) / len(errors)

    ceiling = math.log1p(MAX_ERROR_KM)
    return sum(
        math.log1p(min(error, MAX_ERROR_KM)) / ceiling for error in errors
    ) / len(errors)
