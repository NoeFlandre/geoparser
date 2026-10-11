"""Document ranking and address-to-identifier mapping, with a stub ranker."""

import pytest

from scripts.unitoprank_benchmark.candidates import Candidate, build_candidate_set
from scripts.unitoprank_benchmark.ranking import Mention, rank_document

TEXT = "Ich habe Paris besucht und bin dann nach Nowhere geflogen."
PARIS_ADDRESS = "Paris, Ile-de-France, France, Europe"
TEXAS_ADDRESS = "Paris, Lamar County, Texas, United States, North America"
PARIS_FR = Candidate(
    "2988507",
    "Paris",
    48.8566,
    2.3522,
    "PPLC",
    2140526,
    ("Ile-de-France", "France", "Europe"),
)
PARIS_TX = Candidate(
    "4717560",
    "Paris",
    33.6609,
    -95.5555,
    "PPLA2",
    24719,
    ("Lamar County", "Texas", "United States", "North America"),
)
PARIS_MENTION = Mention(
    start=TEXT.index("Paris"), end=TEXT.index("Paris") + 5, surface="Paris"
)
NOWHERE_MENTION = Mention(
    start=TEXT.index("Nowhere"), end=TEXT.index("Nowhere") + 7, surface="Nowhere"
)
CONFIG = object()


class StubRanker:
    """Return a fixed ranking and remember what the adapter sent."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"ranked_candidates_by_toponym": self.ranking}


def ranked(address, score, lat=48.8566, lon=2.3522):
    return {"address": address, "lat": lat, "lon": lon, "score": score}


def paris_candidates():
    return build_candidate_set({"Paris": [PARIS_FR, PARIS_TX]})


def test_top_choice_is_traced_back_to_its_identifier():
    stub = StubRanker(
        {
            "paris": [
                ranked(PARIS_ADDRESS, 0.6),
                ranked(TEXAS_ADDRESS, 0.5, 33.6609, -95.5555),
            ]
        }
    )
    (outcome,) = rank_document(
        TEXT,
        [PARIS_MENTION],
        paris_candidates(),
        rank_toponyms=stub,
        ranker_config=CONFIG,
    )
    assert outcome.identifier == "2988507"
    assert (outcome.latitude, outcome.longitude) == (48.8566, 2.3522)
    assert (outcome.score, outcome.candidate_count, outcome.tied, outcome.reason) == (
        0.6,
        2,
        False,
        None,
    )
    assert (outcome.start, outcome.end, outcome.surface) == (
        TEXT.index("Paris"),
        TEXT.index("Paris") + 5,
        "Paris",
    )


def test_equal_top_scores_are_flagged_and_resolve_to_the_first_offered():
    stub = StubRanker(
        {
            "paris": [
                ranked(PARIS_ADDRESS, 0.5),
                ranked(TEXAS_ADDRESS, 0.5, 33.6609, -95.5555),
            ]
        }
    )
    (outcome,) = rank_document(
        TEXT,
        [PARIS_MENTION],
        paris_candidates(),
        rank_toponyms=stub,
        ranker_config=CONFIG,
    )
    assert outcome.tied is True
    assert outcome.identifier == "2988507"


def test_mention_with_no_candidate_is_unresolved_for_lack_of_candidates():
    stub = StubRanker({"paris": [ranked(PARIS_ADDRESS, 0.6)]})
    outcomes = rank_document(
        TEXT,
        [PARIS_MENTION, NOWHERE_MENTION],
        paris_candidates(),
        rank_toponyms=stub,
        ranker_config=CONFIG,
    )
    assert outcomes[1].identifier is None
    assert (outcomes[1].reason, outcomes[1].candidate_count) == ("no_candidates", 0)
    assert outcomes[1].score is None


def test_candidates_the_ranker_did_not_place_are_unresolved_as_not_ranked():
    outcomes = rank_document(
        TEXT,
        [PARIS_MENTION],
        paris_candidates(),
        rank_toponyms=StubRanker({}),
        ranker_config=CONFIG,
    )
    assert outcomes[0].identifier is None
    assert (outcomes[0].reason, outcomes[0].candidate_count) == ("not_ranked", 2)


def test_mentions_with_the_same_surface_share_one_ranking():
    second = Mention(start=0, end=5, surface="paris")
    stub = StubRanker({"paris": [ranked(PARIS_ADDRESS, 0.6)]})
    first_outcome, second_outcome = rank_document(
        TEXT,
        [PARIS_MENTION, second],
        paris_candidates(),
        rank_toponyms=stub,
        ranker_config=CONFIG,
    )
    assert first_outcome.identifier == second_outcome.identifier == "2988507"
    assert [item["text"] for item in stub.calls[0]["toponyms"]] == ["Paris", "paris"]


def test_the_ranker_receives_offsets_surfaces_candidates_and_the_config_unchanged():
    stub = StubRanker({})
    candidates = paris_candidates()
    rank_document(
        TEXT, [PARIS_MENTION], candidates, rank_toponyms=stub, ranker_config=CONFIG
    )
    (call,) = stub.calls
    assert call["text"] == TEXT
    assert call["toponyms"] == [{"text": "Paris", "start": 9, "end": 14}]
    assert call["candidates_by_toponym"] is candidates.by_surface
    assert call["config"] is CONFIG


def test_an_address_that_was_never_offered_is_refused():
    stub = StubRanker({"paris": [ranked("Atlantis, Ocean", 0.9)]})
    with pytest.raises(ValueError, match="not offered"):
        rank_document(
            TEXT,
            [PARIS_MENTION],
            paris_candidates(),
            rank_toponyms=stub,
            ranker_config=CONFIG,
        )
