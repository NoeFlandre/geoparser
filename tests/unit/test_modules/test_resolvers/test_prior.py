"""
Unit tests for the prior-aware resolver.

It changes two things about its parent, and nothing else: when an exact search
finds nothing it retries with inflection-trimmed names before the search is
widened, and it ranks candidates by similarity plus a population prior.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from geoparser.modules.resolvers.prior import PriorResolver

PARENT = "geoparser.modules.resolvers.sentencetransformer"


def _feature(feature_id, population=None):
    """A stand-in gazetteer feature."""
    data = {"name": f"f{feature_id}"}
    if population is not None:
        data["population"] = population
    return SimpleNamespace(id=feature_id, identifier=str(feature_id), data=data)


@pytest.fixture
def resolver():
    """A prior resolver whose models and gazetteer are Mocks."""
    with (
        patch(f"{PARENT}.Gazetteer"),
        patch(f"{PARENT}.SentenceTransformer"),
        patch(f"{PARENT}.AutoTokenizer.from_pretrained"),
        patch(f"{PARENT}.spacy.load"),
    ):
        yield PriorResolver(attribute_map={"name": "name", "type": "type"})


@pytest.mark.unit
class TestConfiguration:
    """What is recorded, so results are traceable to settings."""

    def test_records_its_settings(self, resolver):
        """The weight and the fallback are part of the module's identity."""
        assert resolver.config["population_weight"] == PriorResolver.DEFAULT_WEIGHT
        assert resolver.config["inflection_fallback"] is True

    def test_defaults_to_the_upstream_model(self, resolver):
        """Only ranking and search change, not the encoder."""
        assert resolver.model_name == "dguzh/geo-all-MiniLM-L6-v2"


@pytest.mark.unit
class TestInflectionFallback:
    """Retrying an exact miss with trimmed names."""

    def test_an_exact_miss_is_retried_with_a_trimmed_name(self, resolver):
        """'Saksan' misses, 'Saksa' hits, and that is what is returned."""
        hit = _feature(1)
        resolver.gazetteer.search.side_effect = lambda name, method, tiers: (
            [hit] if name == "Saksa" else []
        )

        assert list(resolver._search_candidates("Saksan", "exact", 1)) == [hit]

    def test_an_exact_hit_is_not_second_guessed(self, resolver):
        """A name that matches is not trimmed."""
        hit = _feature(1)
        resolver.gazetteer.search.return_value = [hit]

        assert list(resolver._search_candidates("Paris", "exact", 1)) == [hit]
        resolver.gazetteer.search.assert_called_once_with("Paris", "exact", tiers=1)

    def test_wider_methods_are_not_trimmed(self, resolver):
        """Only exact search is retried; fuzzy already tolerates endings."""
        resolver.gazetteer.search.return_value = []

        assert list(resolver._search_candidates("Saksan", "fuzzy", 1)) == []
        assert resolver.gazetteer.search.call_count == 1


@pytest.mark.unit
class TestRanking:
    """Similarity plus population."""

    def test_a_populous_place_wins_a_near_tie(self, resolver):
        """The hamlet loses to the city when context barely separates them."""
        hamlet, city = _feature(1, 50), _feature(2, 170_000)

        chosen = resolver._best_referent("ctx", [hamlet, city], 0.0, [0.50, 0.48])

        assert chosen == (resolver.gazetteer_name, "2")

    def test_abstention_still_uses_raw_similarity(self, resolver):
        """A prior cannot lift a poor match over the threshold."""
        city = _feature(2, 10_000_000)

        assert resolver._best_referent("ctx", [city], 0.6, [0.5]) is None


@pytest.mark.unit
class TestFallbackEdges:
    """When the fallback has nothing to add."""

    def test_every_variant_missing_returns_nothing(self, resolver):
        """A name no trim can match still comes back empty."""
        resolver.gazetteer.search.return_value = []

        assert list(resolver._search_candidates("Xyzzyq", "exact", 1)) == []
        assert resolver.gazetteer.search.call_count == 4

    def test_the_fallback_can_be_switched_off(self, resolver):
        """With the fallback off, an exact miss is not retried."""
        resolver.inflection_fallback = False
        resolver.gazetteer.search.return_value = []

        assert list(resolver._search_candidates("Saksan", "exact", 1)) == []
        assert resolver.gazetteer.search.call_count == 1


@pytest.mark.unit
def test_ranking_computes_similarities_when_none_are_given(resolver):
    """Without precomputed scores, the parent's similarity is used."""
    import torch

    city = _feature(2, 1_000)
    resolver.context_embeddings["ctx"] = torch.tensor([1.0, 0.0])
    resolver.candidate_embeddings[2] = torch.tensor([1.0, 0.0])

    assert resolver._best_referent("ctx", [city], 0.5) == (resolver.gazetteer_name, "2")
