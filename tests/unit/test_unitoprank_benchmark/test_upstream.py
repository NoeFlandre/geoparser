"""
Run the pinned UniTopRank ranker on a three-sentence example, offline.

This needs a local checkout of the reviewed commit, given in the
UNITORANK_CHECKOUT environment variable, and it is skipped without one. The
ranker runs in the interpreter named by UNITORANK_PYTHON, which must be built
from ranker-requirements.txt, because RapidFuzz's scores decide the ranking. The
checkout is verified against the pin before anything is imported. The expected
identifiers are the places the README's own worked example puts first.
"""

import os
from functools import partial
from pathlib import Path

import pytest

from scripts.unitoprank_benchmark.candidates import Candidate, build_candidate_set
from scripts.unitoprank_benchmark.isolated_ranker import rank_in_reviewed_environment
from scripts.unitoprank_benchmark.ranking import Mention, rank_document

# Only the Linux Python 3.12 test cell and the quality gauntlet fetch the pinned
# checkout (.github/workflows/test.yml and quality.yml). Cloning it in every cell
# would add Windows and macOS checkouts that pins.py has not been verified on, so
# the other cells skip this test on purpose.
CHECKOUT = os.environ.get("UNITORANK_CHECKOUT")
pytestmark = pytest.mark.skipif(
    not CHECKOUT,
    reason=(
        "UNITORANK_CHECKOUT is not set; only the Linux 3.12 test cell and the "
        "quality gauntlet fetch the pinned UniTopRank checkout"
    ),
)

TEXT = "Ich habe Paris besucht und bin dann nach Berlin geflogen."
CANDIDATES = {
    "Paris": [
        Candidate(
            "2988507",
            "Paris",
            48.8566,
            2.3522,
            "PPLC",
            2140526,
            ("Ile-de-France", "France", "Europe"),
        ),
        Candidate(
            "4717560",
            "Paris",
            33.6609,
            -95.5555,
            "PPLA2",
            24719,
            ("Lamar County", "Texas", "United States", "North America"),
        ),
    ],
    "Berlin": [
        Candidate(
            "2950159", "Berlin", 52.52, 13.405, "PPLC", 3644826, ("Germany", "Europe")
        ),
        Candidate(
            "5074472",
            "Berlin",
            44.4684,
            -71.1837,
            "PPL",
            10050,
            ("Coos County", "New Hampshire", "United States", "North America"),
        ),
    ],
}
MENTIONS = [
    Mention(TEXT.index("Paris"), TEXT.index("Paris") + 5, "Paris"),
    Mention(TEXT.index("Berlin"), TEXT.index("Berlin") + 6, "Berlin"),
]


def test_pinned_ranker_places_both_toponyms_on_the_readme_choices(no_external_network):
    rank_toponyms = partial(
        rank_in_reviewed_environment,
        Path(os.environ["UNITORANK_CHECKOUT"]),
        os.environ["UNITORANK_PYTHON"],
    )
    candidates = build_candidate_set(CANDIDATES)
    config = {"top_n": 10, "distance_threshold_km": 300.0}

    first = rank_document(
        TEXT, MENTIONS, candidates, rank_toponyms=rank_toponyms, ranker_config=config
    )
    second = rank_document(
        TEXT, MENTIONS, candidates, rank_toponyms=rank_toponyms, ranker_config=config
    )

    assert [outcome.identifier for outcome in first] == ["2988507", "2950159"]
    assert first == second
    assert all(
        outcome.reason is None and outcome.candidate_count == 2 for outcome in first
    )
