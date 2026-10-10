"""Hand-counted oracles for ranking, abstention, gold-span counts and calibration."""

from typing import Any

import numpy as np
import pytest

from geoparser.modules.resolvers.prior import PriorResolver
from scripts.benchmark_protocol.schema import ResolutionCounts
from scripts.embedding_resolution.resolution import (
    ABSTAIN_ALL_THRESHOLD,
    CALIBRATION_GRID,
    POPULATION_WEIGHT,
    Candidate,
    GoldSpan,
    Observation,
    Prediction,
    ResolutionTally,
    calibrate_min_similarity,
    cosine_similarities,
    decide,
    decide_population_only,
    observe,
)

PARIS = Candidate(
    "geonames:1", population=2_000_000, latitude=48.8566, longitude=2.3522
)
LONDON = Candidate("geonames:2", population=None, latitude=51.5074, longitude=-0.1278)
PARIS_GOLD = GoldSpan("geonames:1", latitude=48.8566, longitude=2.3522)
LONDON_GOLD = GoldSpan("geonames:2", latitude=51.5074, longitude=-0.1278)


def test_cosine_similarity_matches_hand_values():
    query = np.array([1.0, 0.0])
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    assert cosine_similarities(query, candidates) == pytest.approx([1.0, 0.0, -1.0])


def test_an_empty_candidate_set_scores_to_an_empty_list():
    assert cosine_similarities(np.array([1.0, 0.0]), np.empty((0, 2))) == []


def test_mismatched_shapes_and_zero_vectors_are_refused():
    with pytest.raises(ValueError, match="does not match"):
        cosine_similarities(np.array([1.0, 0.0, 0.0]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="undefined"):
        cosine_similarities(np.array([0.0, 0.0]), np.ones((1, 2)))


def test_the_population_weight_is_the_one_the_library_resolver_uses():
    assert POPULATION_WEIGHT == PriorResolver.DEFAULT_WEIGHT


def test_without_a_prior_the_higher_similarity_wins():
    big_city = Candidate("big", population=1_000_000, latitude=None, longitude=None)
    other = Candidate("other", population=None, latitude=None, longitude=None)

    prediction = decide(
        [big_city, other], [0.50, 0.60], policy="similarity", min_similarity=0.0
    )

    assert prediction.chosen == other
    assert prediction.similarity == pytest.approx(0.60)


def test_the_population_prior_can_overturn_a_near_tie():
    # log10(1 + 1e6) / 10 is about 0.6, so the prior adds about 0.18 at weight 0.3.
    big_city = Candidate("big", population=1_000_000, latitude=None, longitude=None)
    other = Candidate("other", population=None, latitude=None, longitude=None)

    prediction = decide(
        [big_city, other], [0.50, 0.60], policy="population", min_similarity=0.0
    )

    assert prediction.chosen == big_city
    assert prediction.similarity == pytest.approx(0.50)


def test_a_chosen_candidate_below_the_threshold_is_abstained_with_its_similarity():
    prediction = decide([LONDON], [0.6], policy="similarity", min_similarity=0.7)

    assert prediction.chosen is None
    assert prediction.invalid is False
    assert prediction.similarity == pytest.approx(0.6)


def test_no_candidates_means_abstention_without_a_similarity():
    assert decide([], [], policy="population", min_similarity=0.0) == Prediction(
        chosen=None, similarity=None
    )


def test_mismatched_candidate_and_similarity_lists_are_refused():
    with pytest.raises(ValueError, match="exactly one similarity"):
        decide([LONDON], [0.1, 0.2], policy="similarity", min_similarity=0.0)


def test_an_unknown_policy_is_refused():
    # Typed as Any: the policy is unknown on purpose, and decide must refuse it.
    unknown_policy: Any = "sum"
    with pytest.raises(ValueError, match="Unknown policy"):
        decide([LONDON], [0.1], policy=unknown_policy, min_similarity=0.0)


def test_ties_go_to_the_earlier_candidate():
    first = Candidate("first", population=None, latitude=None, longitude=None)
    second = Candidate("second", population=None, latitude=None, longitude=None)

    prediction = decide(
        [first, second], [0.5, 0.5], policy="similarity", min_similarity=0.0
    )

    assert prediction.chosen == first


def test_population_only_picks_the_most_populous_and_abstains_only_when_empty():
    unknown = Candidate("unknown", population=None, latitude=None, longitude=None)
    small = Candidate("small", population=50_000, latitude=None, longitude=None)
    large_first = Candidate(
        "large-a", population=200_000, latitude=None, longitude=None
    )
    large_second = Candidate(
        "large-b", population=200_000, latitude=None, longitude=None
    )

    assert (
        decide_population_only([unknown, small, large_first, large_second]).chosen
        == large_first
    )
    assert decide_population_only([]) == Prediction(chosen=None, similarity=None)


def test_gold_span_counts_match_a_hand_count_over_five_spans():
    tally = ResolutionTally()

    # 1: correct and exact, located at 0 km.
    tally.add(PARIS_GOLD, ["geonames:1", "geonames:2"], Prediction(PARIS, 0.9))
    # 2: wrong city, found by retrieval only as the gold ID is missing from it.
    tally.add(LONDON_GOLD, ["geonames:1"], Prediction(PARIS, 0.4))
    # 3: gold has no canonical ID and no coordinates; the pipeline abstains.
    tally.add(
        GoldSpan(None, None, None), ["geonames:2"], Prediction(None, similarity=0.2)
    )
    # 4: the output is invalid, so it is neither resolved nor abstained.
    tally.add(
        GoldSpan("geonames:1", None, None),
        ["geonames:1"],
        Prediction(None, None, invalid=True),
    )
    # 5: correct and exact, located at 0 km.
    tally.add(LONDON_GOLD, ["geonames:2"], Prediction(LONDON, 0.7))

    assert tally.counts() == ResolutionCounts(
        task="gold_span_resolution",
        gold_spans=5,
        resolved=3,
        abstained=1,
        invalid_outputs=1,
        exact_id_eligible=4,
        exact_id_correct=2,
        coordinate_eligible=3,
        within_1km=2,
        within_10km=2,
        within_50km=2,
        candidate_eligible=4,
        candidate_found=3,
    )


def test_a_wrong_city_is_counted_resolved_but_outside_every_distance_band():
    tally = ResolutionTally()

    tally.add(LONDON_GOLD, ["geonames:1"], Prediction(PARIS, 0.4))

    counts = tally.counts()
    assert (counts.resolved, counts.exact_id_correct, counts.within_50km) == (1, 0, 0)
    assert counts.coordinate_eligible == 1


def test_a_chosen_candidate_outside_the_retrieved_set_is_an_invalid_output():
    tally = ResolutionTally()

    tally.add(PARIS_GOLD, ["geonames:2"], Prediction(PARIS, 0.9))

    counts = tally.counts()
    assert (counts.invalid_outputs, counts.resolved, counts.exact_id_correct) == (
        1,
        0,
        0,
    )
    assert counts.candidate_found == 0


def test_missing_gold_targets_stay_in_the_denominators_when_retrieval_fails():
    tally = ResolutionTally()

    tally.add(PARIS_GOLD, [], Prediction(chosen=None, similarity=None))

    counts = tally.counts()
    assert (counts.gold_spans, counts.abstained, counts.candidate_eligible) == (1, 1, 1)
    assert counts.candidate_found == 0


def test_an_empty_tally_produces_valid_zero_counts():
    counts = ResolutionTally().counts()

    assert counts.gold_spans == 0
    assert counts.task == "gold_span_resolution"


def test_observations_record_the_policy_choice_before_any_threshold():
    observation = observe(
        [LONDON, PARIS], [0.40, 0.45], PARIS_GOLD, policy="similarity"
    )

    assert observation.correct is True
    assert observation.similarity == pytest.approx(0.45)


def test_an_observation_needs_a_gold_identifier():
    with pytest.raises(ValueError, match="gold canonical identifier"):
        observe([LONDON], [0.5], GoldSpan(None, None, None), policy="similarity")


def test_an_observation_with_no_candidates_is_never_correct():
    observation = observe([], [], PARIS_GOLD, policy="similarity")

    assert observation == Observation(similarity=None, correct=False)


def test_calibration_rewards_correct_resolutions_and_penalizes_wrong_ones():
    observations = [
        Observation(0.9, correct=True),
        Observation(0.7, correct=True),
        Observation(0.5, correct=False),
        Observation(0.3, correct=False),
        Observation(None, correct=False),
    ]

    calibration = calibrate_min_similarity(observations)

    # Accepting 0.7 and 0.9 gives +2. Accepting 0.5 as well drops it to +1.
    assert calibration.min_similarity == pytest.approx(0.51)
    assert calibration.net_correct == 2
    assert calibration.observations == 5


def test_calibration_ties_go_to_the_lowest_grid_value():
    observations = [Observation(0.9, correct=True), Observation(0.3, correct=True)]

    calibration = calibrate_min_similarity(observations, grid=(0.0, 0.5, 1.0))

    assert calibration.min_similarity == 0.0
    assert calibration.net_correct == 2


def test_calibration_on_a_custom_grid_picks_the_best_point_there():
    observations = [
        Observation(0.9, correct=True),
        Observation(0.5, correct=False),
        Observation(0.3, correct=False),
    ]

    calibration = calibrate_min_similarity(observations, grid=(0.0, 0.5, 1.0))

    # 0.5 and 1.0 both score 0 (one correct, one wrong accepted at 0.5), so the
    # lower of the two is kept: -1 at 0.0, 0 at 0.5, 0 at 1.0.
    assert calibration.min_similarity == 0.5
    assert calibration.net_correct == 0


def test_calibration_needs_observations_and_a_grid():
    with pytest.raises(ValueError, match="at least one development observation"):
        calibrate_min_similarity([])
    with pytest.raises(ValueError, match="grid is empty"):
        calibrate_min_similarity([Observation(0.5, correct=True)], grid=())


def test_the_calibration_grid_spans_the_full_cosine_range_in_hundredths():
    assert len(CALIBRATION_GRID) == 202
    assert CALIBRATION_GRID[0] == -1.0
    assert CALIBRATION_GRID[200] == pytest.approx(1.0)
    assert CALIBRATION_GRID[1] == pytest.approx(-0.99)


def test_the_calibration_grid_ends_with_a_cutoff_no_cosine_reaches():
    assert CALIBRATION_GRID[-1] == ABSTAIN_ALL_THRESHOLD
    assert CALIBRATION_GRID[-1] > 1.0


def test_a_single_wrong_resolution_at_cosine_one_is_calibrated_to_abstain():
    # Under ">=" a cosine of exactly 1.0 reaches every grid value up to 1.0, so
    # only the cutoff above it can abstain. Accepting the wrong span scores -1,
    # abstaining on it scores 0, and abstaining must be the chosen threshold.
    calibration = calibrate_min_similarity([Observation(1.0, correct=False)])

    assert calibration.min_similarity == ABSTAIN_ALL_THRESHOLD
    assert calibration.net_correct == 0
    assert calibration.observations == 1


def test_a_chosen_candidate_without_coordinates_is_located_at_no_distance():
    tally = ResolutionTally()

    unlocated = Candidate("geonames:1", population=None, latitude=None, longitude=None)
    tally.add(PARIS_GOLD, ["geonames:1"], Prediction(unlocated, 0.9))

    counts = tally.counts()
    assert (counts.resolved, counts.exact_id_correct, counts.coordinate_eligible) == (
        1,
        1,
        1,
    )
    assert (counts.within_1km, counts.within_50km) == (0, 0)
