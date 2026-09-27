"""SentenceTransformerResolver similarity scoring (resolvers/_similarity.py)."""

from types import SimpleNamespace
from typing import Any, cast
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
                ["first", "second"],
                cast(Any, [[candidates[0], candidates[1]], []]),
            )

        assert batch[0] == pytest.approx([1.0, 0.0])
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

        resolver._evaluate_candidates(contexts, cast(Any, candidates), results, 0.6)

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
    def test_similarity_batches_return_cosine_values(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Batch scoring returns each candidate's cosine similarity, as floats."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()
        resolver.context_embeddings["ctx"] = torch.tensor([1.0, 0.0, 0.0])
        resolver.candidate_embeddings.update(
            {
                1: torch.tensor([1.0, 0.0, 0.0]),  # identical: 1.0
                2: torch.tensor([0.5, 0.866, 0.0]),  # 60 degrees: about 0.5
                3: torch.tensor([-1.0, 0.0, 0.0]),  # opposite: -1.0
            }
        )
        candidates = [SimpleNamespace(id=i) for i in (1, 2, 3)]

        # Act
        (similarities,) = resolver._calculate_similarity_batches(
            ["ctx"], cast(Any, [candidates])
        )

        # Assert
        assert all(isinstance(value, float) for value in similarities)
        assert similarities == pytest.approx([1.0, 0.5, -1.0], abs=0.01)
