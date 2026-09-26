"""SentenceTransformerResolver similarity scoring (resolvers/_similarity.py)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.unit
class TestSimilarity:
    """Cosine scoring of candidates against contexts."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_calculate_similarities_returns_list_of_floats(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _calculate_similarities returns list of similarity scores."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()

        context_embedding = torch.tensor([1.0, 0.0, 0.0])
        candidate_embeddings = [
            torch.tensor([1.0, 0.0, 0.0]),  # Perfect match
            torch.tensor([0.0, 1.0, 0.0]),  # Orthogonal
        ]

        # Act
        similarities = resolver._calculate_similarities(
            context_embedding, candidate_embeddings
        )

        # Assert
        assert isinstance(similarities, list)
        assert len(similarities) == 2
        assert all(isinstance(s, float) for s in similarities)
        # First should be higher similarity than second
        assert similarities[0] > similarities[1]

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_calculate_similarities_handles_empty_list(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _calculate_similarities handles empty candidate list."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()

        context_embedding = torch.tensor([1.0, 0.0, 0.0])
        candidate_embeddings = []

        # Act
        similarities = resolver._calculate_similarities(
            context_embedding, candidate_embeddings
        )

        # Assert
        assert similarities == []

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_vectorizes_uneven_candidate_lists_in_one_similarity_call(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Batch scoring matches scalar scoring for uneven candidate lists."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()
        candidates = [
            SimpleNamespace(id=1),
            SimpleNamespace(id=2),
            SimpleNamespace(id=3),
        ]
        resolver.context_embeddings.update(
            {
                "first": torch.tensor([1.0, 0.0]),
                "second": torch.tensor([0.0, 1.0]),
            }
        )
        resolver.candidate_embeddings.update(
            {
                1: torch.tensor([1.0, 0.0]),
                2: torch.tensor([0.0, 1.0]),
                3: torch.tensor([1.0, 1.0]),
            }
        )

        with patch(
            "geoparser.modules.resolvers.sentencetransformer.torch.nn.functional.cosine_similarity",
            wraps=torch.nn.functional.cosine_similarity,
        ) as cosine_similarity:
            batch = resolver._calculate_similarity_batches(
                ["first", "second"], [[candidates[0], candidates[1]], []]
            )

        scalar = [
            resolver._calculate_similarities(
                resolver.context_embeddings["first"],
                [resolver.candidate_embeddings[1], resolver.candidate_embeddings[2]],
            ),
            [],
        ]
        assert batch[0] == pytest.approx(scalar[0])
        assert batch[1] == []
        assert cosine_similarity.call_count == 1

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_evaluation_uses_one_similarity_batch_for_all_references(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """The search pass sends all unresolved references through one scorer."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()
        first = SimpleNamespace(id=1, identifier="A")
        second = SimpleNamespace(id=2, identifier="B")
        resolver._calculate_similarity_batches = Mock(return_value=[[0.9], [0.8]])
        resolver._best_referent = Mock(
            side_effect=lambda context, candidates, threshold, scores: (
                resolver.gazetteer_name,
                candidates[0].identifier,
            )
        )
        contexts = [["first", "second"]]
        candidates = [[[first], [second]]]
        results = [[None, None]]

        resolver._evaluate_candidates(contexts, candidates, results, 0.6)

        resolver._calculate_similarity_batches.assert_called_once_with(
            ["first", "second"], [[first], [second]]
        )
        assert results == [
            [(resolver.gazetteer_name, "A"), (resolver.gazetteer_name, "B")]
        ]

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_calculate_similarities_returns_correct_values(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _calculate_similarities returns correct cosine similarity values."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()

        # Create embeddings with known similarities
        context_embedding = torch.tensor([1.0, 0.0, 0.0])
        candidate_embeddings = [
            torch.tensor([1.0, 0.0, 0.0]),  # Similarity = 1.0 (identical)
            torch.tensor([0.5, 0.866, 0.0]),  # Similarity ≈ 0.5 (60 degree angle)
            torch.tensor([-1.0, 0.0, 0.0]),  # Similarity = -1.0 (opposite)
        ]

        # Act
        similarities = resolver._calculate_similarities(
            context_embedding, candidate_embeddings
        )

        # Assert
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 0.01  # First is perfect match
        assert abs(similarities[1] - 0.5) < 0.1  # Second is ~0.5
        assert abs(similarities[2] - (-1.0)) < 0.01  # Third is opposite
