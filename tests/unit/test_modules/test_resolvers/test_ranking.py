"""
Unit tests for the pure ranking helpers.

Both helpers exist for multilingual text. Inflected forms ("Saksan", Finnish
for "of Germany") miss an exact gazetteer match and used to fall through to
fuzzy search, and exonyms ("Bâle") match hamlets as readily as the city. The
arithmetic is kept free of models so it can be pinned directly.
"""

import math

import pytest

from geoparser.modules.resolvers.ranking import (
    combined_scores,
    inflection_variants,
    population_prior,
)


@pytest.mark.unit
class TestPopulationPrior:
    """How much a candidate's size counts for."""

    def test_is_zero_without_a_population(self):
        """A feature with no population gets no boost, and no penalty."""
        assert population_prior(None) == 0.0
        assert population_prior(0) == 0.0

    def test_grows_with_the_logarithm_of_population(self):
        """Ten times the people adds a constant, not ten times the boost."""
        assert population_prior(1_000_000) == pytest.approx(math.log10(1_000_001) / 10)

    def test_stays_below_one_for_any_real_place(self):
        """Even the most populous country cannot swamp a similarity."""
        assert population_prior(1_500_000_000) < 1.0

    def test_ignores_unparsable_values(self):
        """A malformed population is treated as unknown."""
        assert population_prior("n/a") == 0.0
        assert population_prior(-5) == 0.0


@pytest.mark.unit
class TestCombinedScores:
    """Similarity plus a weighted prior."""

    def test_zero_weight_leaves_similarities_unchanged(self):
        """The prior is off unless asked for."""
        assert combined_scores([0.5, 0.4], [10, 1_000_000], 0.0) == [0.5, 0.4]

    def test_a_large_place_wins_a_near_tie(self):
        """Close similarities are separated by size."""
        scores = combined_scores([0.50, 0.49], [10, 1_000_000], 0.1)

        assert scores[1] > scores[0]

    def test_a_clear_similarity_win_survives_the_prior(self):
        """Context still decides when it is decisive."""
        scores = combined_scores([0.9, 0.4], [10, 1_000_000], 0.1)

        assert scores[0] > scores[1]


@pytest.mark.unit
class TestInflectionVariants:
    """Names to retry when an exact search finds nothing."""

    def test_trims_up_to_three_trailing_characters(self):
        """Common case endings are removed, shortest trim first."""
        assert inflection_variants("Saksan") == ["Saksa", "Saks", "Sak"]

    def test_keeps_at_least_three_characters(self):
        """A short name is not trimmed into a meaningless stub."""
        assert inflection_variants("Rome") == ["Rom"]
        assert inflection_variants("Nom") == []

    def test_does_not_trim_multiword_names(self):
        """A phrase is not a single inflected word."""
        assert inflection_variants("New York") == []

    def test_ignores_surrounding_whitespace(self):
        """The query is stripped before trimming."""
        assert inflection_variants(" Chinas ") == ["China", "Chin", "Chi"]
