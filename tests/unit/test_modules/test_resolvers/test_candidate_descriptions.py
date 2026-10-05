"""
Tests for the text the resolver embeds on behalf of a candidate.

A candidate is compared to a reference's context as a sentence built from its
gazetteer attributes. The exact wording is what the encoder sees, so the
separators are behaviour rather than presentation.
"""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def resolver():
    """A resolver using the geonames attribute map."""
    with (
        patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer"),
        patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer"),
        patch(
            "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
        ),
        patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model"),
    ):
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        return SentenceTransformerResolver(gazetteer_name="geonames")


def _candidate(**data) -> Mock:
    """A gazetteer candidate carrying the given attributes."""
    candidate = Mock()
    candidate.data = data
    return candidate


@pytest.mark.unit
class TestEmbeddingCaches:
    """Embedding candidates and contexts once each."""

    def test_describes_and_encodes_each_pending_candidate(self, resolver):
        """The descriptions handed to the encoder are the candidates' own."""
        # Arrange
        first, second = Mock(id=1), Mock(id=2)
        resolver.candidate_embeddings = {}
        with (
            patch.object(
                resolver,
                "_candidate_description",
                side_effect=lambda c: f"desc-{c.id}",
            ),
            patch.object(resolver, "_encode", return_value=["e1", "e2"]) as encode,
        ):
            # Act
            resolver._embed_candidates([[[first, second]]], [[None]])

        # Assert
        assert encode.call_args.args[0] == ["desc-1", "desc-2"]
        assert resolver.candidate_embeddings == {1: "e1", 2: "e2"}

    def test_does_not_call_the_encoder_when_nothing_is_pending(self, resolver):
        """Already-embedded candidates cost nothing."""
        # Arrange
        candidate = Mock(id=1)
        resolver.candidate_embeddings = {1: "cached"}
        with patch.object(resolver, "_encode") as encode:
            # Act
            resolver._embed_candidates([[[candidate]]], [[None]])

        # Assert
        encode.assert_not_called()

    def test_caches_context_embeddings_by_their_text(self, resolver):
        """Each distinct context is encoded once and stored under itself."""
        # Arrange
        resolver.context_embeddings = {}
        with patch.object(resolver, "_encode", return_value=["ea", "eb"]) as encode:
            # Act
            resolver._embed_contexts([["b", "a"], ["a"]])

        # Assert - sorted, so the order is stable run to run
        assert encode.call_args.args[0] == ["a", "b"]
        assert resolver.context_embeddings == {"a": "ea", "b": "eb"}
