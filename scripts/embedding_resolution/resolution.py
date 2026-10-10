"""Candidate ranking, abstention and gold-span counts for the #163 comparison.

Everything here is arithmetic over embeddings that an adapter already produced.
The population prior is the one the library's PriorResolver uses, with the same
weight, so the comparison and the library cannot drift apart. The prior policy
and the no-prior policy differ only in that weight; a threshold always applies
to the raw cosine similarity of the chosen candidate, as it does in the library.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from geoparser.evaluation import haversine_km
from geoparser.modules.resolvers.ranking import combined_scores, population_prior
from scripts.benchmark_protocol.schema import ResolutionCounts

Policy = Literal["similarity", "population"]

# The weight PriorResolver.DEFAULT_WEIGHT uses; a test pins the two together.
POPULATION_WEIGHT = 0.3
DISTANCE_THRESHOLDS_KM = (1.0, 10.0, 50.0)
TASK = "gold_span_resolution"

# Cosine similarity lies in [-1, 1]. Calibration scans the whole range rather
# than starting from any model's historical cutoff, so no scale is assumed.
# A chosen candidate is accepted when its similarity reaches the threshold, so a
# cutoff above every cosine abstains on every span. Without that point the grid
# could never score abstaining on everything, and a wrong resolution at cosine
# exactly 1.0 would always be accepted.
ABSTAIN_ALL_THRESHOLD = 1.01
CALIBRATION_GRID = (
    *(round(-1.0 + 0.01 * step, 2) for step in range(201)),
    ABSTAIN_ALL_THRESHOLD,
)

_POLICIES: dict[Policy, float] = {"similarity": 0.0, "population": POPULATION_WEIGHT}


@dataclass(frozen=True)
class Candidate:
    """A gazetteer feature as the resolver sees it: an ID and its ranking inputs."""

    identifier: str
    population: float | None
    latitude: float | None
    longitude: float | None

    @property
    def point(self) -> tuple[float, float] | None:
        """The (latitude, longitude) pair, or None when either is missing."""
        return _point(self.latitude, self.longitude)


@dataclass(frozen=True)
class GoldSpan:
    """A gold span's canonical target. Missing fields make it ineligible for that metric."""

    identifier: str | None
    latitude: float | None
    longitude: float | None

    @property
    def point(self) -> tuple[float, float] | None:
        """The (latitude, longitude) pair, or None when either is missing."""
        return _point(self.latitude, self.longitude)


def _point(
    latitude: float | None, longitude: float | None
) -> tuple[float, float] | None:
    """Pair two coordinates, or None when either is absent."""
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


@dataclass(frozen=True)
class Prediction:
    """What the pipeline returned for one span: a chosen candidate, or abstention."""

    chosen: Candidate | None
    similarity: float | None
    invalid: bool = False


def cosine_similarities(query: np.ndarray, candidates: np.ndarray) -> list[float]:
    """
    Cosine similarity between one query vector and each candidate row.

    Args:
        query: One embedding, shape ``(dimension,)``
        candidates: Candidate embeddings, shape ``(count, dimension)``; a zero
            count returns an empty list, which is how an empty candidate set
            reaches the abstention path without an error

    Returns:
        One similarity per candidate, in order

    Raises:
        ValueError: If the shapes disagree or an embedding has zero length
    """
    matrix = np.asarray(candidates, dtype=np.float64)
    vector = np.asarray(query, dtype=np.float64)
    if matrix.size == 0:
        return []
    _check_shapes(vector, matrix)
    denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    if (denominators == 0).any():
        msg = "cosine similarity is undefined for a zero-length embedding"
        raise ValueError(msg)
    return (matrix @ vector / denominators).tolist()


def _check_shapes(vector: np.ndarray, matrix: np.ndarray) -> None:
    """Require one vector and a two-dimensional matrix of the same width."""
    is_pair = matrix.ndim == 2 and vector.ndim == 1
    if not is_pair or matrix.shape[1] != vector.shape[0]:
        msg = (
            f"query shape {vector.shape} does not match candidate shape {matrix.shape}"
        )
        raise ValueError(msg)


def policy_weight(policy: Policy) -> float:
    """Return the population-prior weight for a policy.

    Args:
        policy: ``"similarity"`` for no prior, ``"population"`` for the prior

    Returns:
        The weight; 0.0 for ``"similarity"``

    Raises:
        ValueError: If the policy is not one of the two compared
    """
    if policy not in _POLICIES:
        msg = f"Unknown policy {policy!r}; expected one of {sorted(_POLICIES)}."
        raise ValueError(msg)
    return _POLICIES[policy]


def decide(
    candidates: Sequence[Candidate],
    similarities: Sequence[float],
    *,
    policy: Policy,
    min_similarity: float,
) -> Prediction:
    """
    Choose the best candidate under a policy, or abstain below the threshold.

    Args:
        candidates: The retrieved candidates, in gazetteer order
        similarities: Each candidate's cosine similarity to the context
        policy: Whether to add the population prior to the similarity
        min_similarity: The chosen candidate's similarity must reach this

    Returns:
        The chosen candidate, or an abstention; ties go to the earlier candidate

    Raises:
        ValueError: If the candidate and similarity lists differ in length
    """
    if len(candidates) != len(similarities):
        msg = "each candidate needs exactly one similarity"
        raise ValueError(msg)
    if not candidates:
        return Prediction(chosen=None, similarity=None)
    scores = combined_scores(
        similarities,
        [candidate.population for candidate in candidates],
        policy_weight(policy),
    )
    best = max(range(len(scores)), key=scores.__getitem__)
    similarity = float(similarities[best])
    if similarity < min_similarity:
        return Prediction(chosen=None, similarity=similarity)
    return Prediction(chosen=candidates[best], similarity=similarity)


def decide_population_only(candidates: Sequence[Candidate]) -> Prediction:
    """
    Choose the most populous candidate, ignoring the context entirely.

    This is the baseline the embedding models must beat. It abstains only when
    no candidate was retrieved, so it has no similarity to threshold.

    Args:
        candidates: The retrieved candidates, in gazetteer order

    Returns:
        The most populous candidate, earliest on ties, or an abstention
    """
    if not candidates:
        return Prediction(chosen=None, similarity=None)
    best = max(
        range(len(candidates)),
        key=lambda index: population_prior(candidates[index].population),
    )
    return Prediction(chosen=candidates[best], similarity=None)


def _distance_km(gold: GoldSpan, chosen: Candidate | None) -> float | None:
    """Great-circle distance from the gold point to the chosen one, if both exist."""
    if chosen is None:
        return None
    gold_point, chosen_point = gold.point, chosen.point
    if gold_point is None or chosen_point is None:
        return None
    return haversine_km(*gold_point, *chosen_point)


def _usable(prediction: Prediction, retrieved_ids: frozenset[str]) -> bool:
    """Whether a prediction is a valid resolution or abstention for this span.

    A chosen candidate must have been retrieved; otherwise the output is invalid.
    """
    if prediction.invalid:
        return False
    return prediction.chosen is None or prediction.chosen.identifier in retrieved_ids


@dataclass
class ResolutionTally:
    """Running gold-span counts, scored one span at a time.

    Each count's denominator is explicit. A missing gold target counts as a miss
    in coverage and candidate recall; it is never dropped because retrieval failed.
    """

    gold_spans: int = 0
    resolved: int = 0
    abstained: int = 0
    invalid_outputs: int = 0
    exact_id_eligible: int = 0
    exact_id_correct: int = 0
    coordinate_eligible: int = 0
    within_1km: int = 0
    within_10km: int = 0
    within_50km: int = 0
    candidate_eligible: int = 0
    candidate_found: int = 0

    def add(
        self, gold: GoldSpan, retrieved: Sequence[str], prediction: Prediction
    ) -> None:
        """
        Score one gold span.

        Args:
            gold: The span's canonical target
            retrieved: Identifiers of every candidate retrieved, before ranking
            prediction: What the pipeline returned for the span
        """
        retrieved_ids = frozenset(retrieved)
        usable = _usable(prediction, retrieved_ids)
        self.gold_spans += 1
        self._record_disposition(prediction, usable=usable)
        if gold.identifier is not None:
            self._record_retrieval(gold.identifier, retrieved_ids)
            if usable and prediction.chosen is not None:
                self._record_exact_id(gold.identifier, prediction.chosen)
        self._record_distance(gold, prediction, usable=usable)

    def _record_disposition(self, prediction: Prediction, *, usable: bool) -> None:
        """Count the span as resolved, abstained, or an invalid output."""
        if not usable:
            self.invalid_outputs += 1
        elif prediction.chosen is None:
            self.abstained += 1
        else:
            self.resolved += 1

    def _record_retrieval(self, identifier: str, retrieved_ids: frozenset[str]) -> None:
        """Count one exact-ID-eligible span and whether retrieval reached its target."""
        self.exact_id_eligible += 1
        self.candidate_eligible += 1
        self.candidate_found += int(identifier in retrieved_ids)

    def _record_exact_id(self, identifier: str, chosen: Candidate) -> None:
        """Count a resolution whose chosen ID is the gold canonical ID."""
        self.exact_id_correct += int(chosen.identifier == identifier)

    def _record_distance(
        self, gold: GoldSpan, prediction: Prediction, *, usable: bool
    ) -> None:
        """Count the distance bands for a located gold span."""
        if gold.point is None:
            return
        self.coordinate_eligible += 1
        distance = _distance_km(gold, prediction.chosen) if usable else None
        if distance is not None:
            self._count_within(distance)

    def _count_within(self, distance: float) -> None:
        """Count a located resolution in each distance band it falls inside."""
        self.within_1km += int(distance <= DISTANCE_THRESHOLDS_KM[0])
        self.within_10km += int(distance <= DISTANCE_THRESHOLDS_KM[1])
        self.within_50km += int(distance <= DISTANCE_THRESHOLDS_KM[2])

    def counts(self) -> ResolutionCounts:
        """Return the tally as the benchmark protocol's validated count record."""
        return ResolutionCounts(task=TASK, **asdict(self))


@dataclass(frozen=True)
class Observation:
    """One development span, reduced to what threshold calibration needs."""

    similarity: float | None
    correct: bool


def observe(
    candidates: Sequence[Candidate],
    similarities: Sequence[float],
    gold: GoldSpan,
    *,
    policy: Policy,
) -> Observation:
    """
    Record the best candidate under a policy, before any threshold is applied.

    Args:
        candidates: The retrieved candidates
        similarities: Each candidate's cosine similarity to the context
        gold: The span's canonical target, which must have an identifier
        policy: The prior policy being calibrated

    Returns:
        The chosen candidate's similarity and whether it is the gold identifier

    Raises:
        ValueError: If the gold span has no canonical identifier
    """
    if gold.identifier is None:
        msg = "calibration observations need a gold canonical identifier"
        raise ValueError(msg)
    prediction = decide(
        candidates, similarities, policy=policy, min_similarity=-math.inf
    )
    correct = (
        prediction.chosen is not None
        and prediction.chosen.identifier == gold.identifier
    )
    return Observation(similarity=prediction.similarity, correct=correct)


@dataclass(frozen=True)
class Calibration:
    """The development threshold for one model and policy, and its objective value."""

    min_similarity: float
    net_correct: int
    observations: int


def calibrate_min_similarity(
    observations: Sequence[Observation], grid: Sequence[float] = CALIBRATION_GRID
) -> Calibration:
    """
    Choose the threshold that maximizes correct minus incorrect resolutions.

    A correct resolution scores +1, an incorrect one -1, and an abstention 0, so
    raising the threshold only pays while it removes more errors than correct
    answers. Ties go to the lowest threshold in the grid, which keeps coverage.
    The objective is calibrated per model on development spans only.

    Args:
        observations: One record per development span with a gold identifier
        grid: Candidate thresholds, in ascending order

    Returns:
        The selected threshold with its objective value

    Raises:
        ValueError: If there are no observations, or the grid is empty
    """
    if not observations:
        msg = "threshold calibration needs at least one development observation"
        raise ValueError(msg)
    if not grid:
        msg = "threshold grid is empty"
        raise ValueError(msg)
    scored = [(_net_correct(observations, threshold), threshold) for threshold in grid]
    # Maximize the net count, then the negated threshold so the lowest one wins a tie.
    net, threshold = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return Calibration(
        min_similarity=threshold, net_correct=net, observations=len(observations)
    )


def _net_correct(observations: Sequence[Observation], threshold: float) -> int:
    """Correct minus incorrect resolutions among those reaching the threshold."""
    return sum(
        1 if observation.correct else -1
        for observation in observations
        if observation.similarity is not None and observation.similarity >= threshold
    )
