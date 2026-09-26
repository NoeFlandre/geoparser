"""
Unit tests for geoparser/modules/resolvers/sentencetransformer.py

Tests the SentenceTransformerResolver module with mocked dependencies.
"""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from geoparser.modules._spacy import load_spacy_model


@pytest.mark.unit
class TestSentenceTransformerResolverInitialization:
    """Test SentenceTransformerResolver initialization."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_creates_with_default_parameters(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that SentenceTransformerResolver can be created with default parameters."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        resolver = SentenceTransformerResolver()

        # Assert
        assert resolver.name == "SentenceTransformerResolver"
        assert resolver.model_name == "dguzh/geo-all-MiniLM-L6-v2"
        assert resolver.gazetteer_name == "geonames"
        assert resolver.min_similarity == 0.6
        assert resolver.max_tiers == 3

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_creates_with_custom_parameters(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that SentenceTransformerResolver can be created with custom parameters."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "name",
            "type": "type",
            "level1": "level1",
            "level2": "level2",
            "level3": "level3",
        }

        # Act
        resolver = SentenceTransformerResolver(
            model_name="custom-model",
            gazetteer_name="custom-gazetteer",
            min_similarity=0.8,
            max_tiers=5,
            attribute_map=custom_map,
        )

        # Assert
        assert resolver.model_name == "custom-model"
        assert resolver.gazetteer_name == "custom-gazetteer"
        assert resolver.min_similarity == 0.8
        assert resolver.max_tiers == 5

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_config_contains_all_parameters(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that config stores all initialization parameters."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "name",
            "type": "type",
            "level1": "level1",
            "level2": "level2",
            "level3": "level3",
        }

        # Act
        resolver = SentenceTransformerResolver(
            model_name="test-model",
            gazetteer_name="test-gazetteer",
            min_similarity=0.75,
            max_tiers=4,
            attribute_map=custom_map,
        )

        # Assert
        assert resolver.config["model_name"] == "test-model"
        assert resolver.config["gazetteer_name"] == "test-gazetteer"
        assert resolver.config["min_similarity"] == 0.75
        assert resolver.config["max_tiers"] == 4

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_loads_sentence_transformer_model(
        self, mock_gazetteer, mock_transformer_class, mock_tokenizer, mock_spacy_load
    ):
        """Test that SentenceTransformer model is loaded."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        SentenceTransformerResolver(model_name="test-model")

        # Assert
        mock_transformer_class.assert_called_once_with("test-model")

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_loads_tokenizer(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that AutoTokenizer is loaded."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        SentenceTransformerResolver(model_name="test-model")

        # Assert
        mock_tokenizer.assert_called_once_with("test-model")

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_loads_spacy_sentence_splitter(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that spaCy sentence splitter model is loaded."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        SentenceTransformerResolver()

        # Assert
        mock_spacy_load.assert_called_once_with("xx_sent_ud_sm")

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_initializes_gazetteer(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that Gazetteer is initialized."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "name",
            "type": "type",
            "level1": "level1",
            "level2": "level2",
            "level3": "level3",
        }

        # Act
        SentenceTransformerResolver(
            gazetteer_name="test-gazetteer", attribute_map=custom_map
        )

        # Assert
        mock_gazetteer_class.assert_called_once_with("test-gazetteer")

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_initializes_empty_caches(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that all caches are initialized as empty dictionaries."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        resolver = SentenceTransformerResolver()

        # Assert
        assert resolver.doc_tokens == {}
        assert resolver.doc_objects == {}
        assert resolver.context_embeddings == {}
        assert resolver.candidate_embeddings == {}
        assert resolver.candidate_search_cache == {}
        assert resolver.candidate_descriptions == {}
        assert resolver.measured_sentences == {}

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_different_configs_produce_different_ids(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that different configurations produce different module IDs."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        resolver1 = SentenceTransformerResolver(min_similarity=0.6)
        resolver2 = SentenceTransformerResolver(min_similarity=0.7)

        # Assert
        assert resolver1.id != resolver2.id

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_same_config_produces_same_id(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that same configuration produces same module ID."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        resolver1 = SentenceTransformerResolver(
            model_name="model1",
            gazetteer_name="geonames",
            min_similarity=0.6,
            max_tiers=3,
        )
        resolver2 = SentenceTransformerResolver(
            model_name="model1",
            gazetteer_name="geonames",
            min_similarity=0.6,
            max_tiers=3,
        )

        # Assert
        assert resolver1.id == resolver2.id

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_accepts_custom_attribute_map(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that custom attribute_map is accepted and stored."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "custom_name",
            "type": "custom_type",
            "level1": "custom_level1",
            "level2": "custom_level2",
            "level3": "custom_level3",
        }

        # Act
        resolver = SentenceTransformerResolver(
            gazetteer_name="custom_gazetteer", attribute_map=custom_map
        )

        # Assert
        assert resolver.attribute_map == custom_map

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_raises_error_for_unknown_gazetteer_without_attribute_map(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that error is raised for unknown gazetteer when no attribute_map provided."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="not configured in GAZETTEER_ATTRIBUTE_MAP"
        ):
            SentenceTransformerResolver(gazetteer_name="unknown_gazetteer")

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_uses_gazetteer_attribute_map_when_no_custom_map(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that GAZETTEER_ATTRIBUTE_MAP is used when no custom map provided."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Assert
        expected_map = SentenceTransformerResolver.GAZETTEER_ATTRIBUTE_MAP["geonames"]
        assert resolver.attribute_map == expected_map

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_validate_and_set_attribute_map_returns_custom_map(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _validate_and_set_attribute_map returns custom map when provided."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "test_name",
            "type": "test_type",
            "level1": "test_level1",
            "level2": "test_level2",
            "level3": "test_level3",
        }

        # Act
        resolver = SentenceTransformerResolver(
            gazetteer_name="any-gazetteer", attribute_map=custom_map
        )
        result = resolver._validate_and_set_attribute_map("any-gazetteer", custom_map)

        # Assert
        assert result == custom_map

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_validate_and_set_attribute_map_looks_up_configured_gazetteer(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _validate_and_set_attribute_map looks up configured gazetteer."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Act
        resolver = SentenceTransformerResolver(gazetteer_name="geonames")
        result = resolver._validate_and_set_attribute_map("geonames", None)

        # Assert
        expected_map = SentenceTransformerResolver.GAZETTEER_ATTRIBUTE_MAP["geonames"]
        assert result == expected_map

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_validate_and_set_attribute_map_raises_for_unknown_gazetteer(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _validate_and_set_attribute_map raises error for unknown gazetteer."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        # Create a resolver first (to access the method)
        resolver = SentenceTransformerResolver(
            gazetteer_name="geonames"
        )  # Use valid gazetteer

        # Act & Assert
        with pytest.raises(
            ValueError, match="not configured in GAZETTEER_ATTRIBUTE_MAP"
        ):
            resolver._validate_and_set_attribute_map("unknown_gazetteer", None)

    @patch(
        "geoparser.modules.resolvers.sentencetransformer.load_spacy_model",
        load_spacy_model,
    )
    @patch("geoparser.modules._spacy.spacy.cli.download")
    @patch("geoparser.modules._spacy.spacy.load")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_downloads_spacy_model_if_not_found(
        self,
        mock_gazetteer,
        mock_transformer,
        mock_tokenizer,
        mock_spacy_load,
        mock_download,
    ):
        """Test that spaCy model is automatically downloaded if not found."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_nlp = Mock()

        # First call raises OSError (model not found), second call succeeds
        mock_spacy_load.side_effect = [OSError("Model not found"), mock_nlp]

        # Act
        resolver = SentenceTransformerResolver()

        # Assert
        mock_download.assert_called_once_with("xx_sent_ud_sm")
        assert mock_spacy_load.call_count == 2
        assert resolver.nlp == mock_nlp

    @patch(
        "geoparser.modules.resolvers.sentencetransformer.load_spacy_model",
        load_spacy_model,
    )
    @patch("geoparser.modules._spacy.spacy.load")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_does_not_download_spacy_model_if_exists(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that spaCy model is not downloaded if it already exists."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp

        # Act
        resolver = SentenceTransformerResolver()

        # Assert
        # spacy.load should be called only once (no download needed)
        assert mock_spacy_load.call_count == 1
        assert resolver.nlp == mock_nlp


@pytest.mark.unit
class TestSentenceTransformerResolverPredict:
    """Test SentenceTransformerResolver predict method."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_handles_empty_text_list(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that predict handles empty text list."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()

        # Act
        results = resolver.predict(texts=[], references=[])

        # Assert
        assert results == []

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_context_embeddings(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that context embeddings are cached to avoid recomputation."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512
        mock_transformer_instance.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])

        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_tokenizer_instance.tokenize.return_value = ["test"]

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer.return_value
        mock_candidate = Mock()
        mock_candidate.id = 1
        mock_candidate.identifier = "123"
        mock_candidate.data = {
            "name": "Paris",
            "feature_name": "city",
            "country_name": "France",
        }
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        resolver = SentenceTransformerResolver()

        def context_encode_calls() -> int:
            """How many times the transformer was asked to embed the context."""
            # Contexts are encoded as a batch of unique strings; candidate
            # descriptions go through the same mock, so match on the payload.
            return sum(
                1
                for call in mock_transformer_instance.encode.call_args_list
                if call.args and call.args[0] == ["Test"]
            )

        # Act - Call predict twice with same text
        resolver.predict(texts=["Test"], references=[[(0, 4)]])
        after_first = context_encode_calls()

        resolver.predict(texts=["Test"], references=[[(0, 4)]])
        after_second = context_encode_calls()

        # Assert - encode should not be called again for the same context
        # (though it may be called for candidates)
        assert "Test" in resolver.context_embeddings
        assert after_first == 1
        assert after_second == 1

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_candidate_embeddings(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that candidate embeddings are cached to avoid recomputation."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512
        mock_transformer_instance.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])

        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_tokenizer_instance.tokenize.return_value = ["test"]

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer.return_value
        mock_candidate = Mock()
        mock_candidate.id = 1
        mock_candidate.identifier = "123"
        mock_candidate.data = {
            "name": "Paris",
            "feature_name": "city",
            "country_name": "France",
        }
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        resolver = SentenceTransformerResolver()

        # Act - Call predict twice
        resolver.predict(texts=["Test"], references=[[(0, 4)]])
        resolver.predict(texts=["Test"], references=[[(0, 4)]])

        # Assert - Candidate with id=1 should be in cache
        assert 1 in resolver.candidate_embeddings

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_doc_tokens(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that document token counts are cached to avoid recomputation."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512
        mock_transformer_instance.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])

        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_tokenizer_instance.tokenize.return_value = ["test", "token"]

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer.return_value
        mock_candidate = Mock()
        mock_candidate.id = 1
        mock_candidate.identifier = "123"
        mock_candidate.data = {
            "name": "Paris",
            "feature_name": "city",
            "country_name": "France",
        }
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        resolver = SentenceTransformerResolver()

        # Act - Call predict with document containing multiple references
        text = "Test text"

        def doc_tokenize_calls() -> int:
            """How many times the full document text was tokenized."""
            # Sentence splitting tokenizes fragments; only the whole document
            # counts towards the cache being honoured.
            return sum(
                1
                for call in mock_tokenizer_instance.tokenize.call_args_list
                if call.args and call.args[0] == text
            )

        resolver.predict(texts=[text], references=[[(0, 4), (5, 9)]])

        # Assert - Token count for text should be cached, and the two
        # references in this one document share the single tokenization
        assert text in resolver.doc_tokens
        assert resolver.doc_tokens[text] == 2
        after_first = doc_tokenize_calls()
        assert after_first == 1

        # Call predict again with same text
        resolver.predict(texts=[text], references=[[(0, 4)]])

        # Assert - tokenize should not be called again for the document text
        # (it may be called for sentence tokenization, but not for full doc)
        assert text in resolver.doc_tokens
        assert doc_tokenize_calls() == after_first

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_doc_objects(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that spaCy doc objects are cached to avoid recomputation."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512
        mock_transformer_instance.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])

        mock_tokenizer_instance = mock_tokenizer.return_value
        # Return many tokens to trigger sentence splitting
        mock_tokenizer_instance.tokenize.return_value = ["token"] * 600

        # Mock spaCy
        mock_nlp_instance = Mock()
        mock_sent = Mock()
        mock_sent.start_char = 0
        mock_sent.end_char = 9
        mock_sent.text = "Test text"
        mock_doc = Mock()
        mock_doc.sents = [mock_sent]
        mock_nlp_instance.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp_instance

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer.return_value
        mock_candidate = Mock()
        mock_candidate.id = 1
        mock_candidate.identifier = "123"
        mock_candidate.data = {
            "name": "Paris",
            "feature_name": "city",
            "country_name": "France",
        }
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        resolver = SentenceTransformerResolver()

        # Act - Call predict with document containing multiple references
        text = "Test text"
        resolver.predict(texts=[text], references=[[(0, 4), (5, 9)]])

        # Assert - spaCy doc for text should be cached
        assert text in resolver.doc_objects

        # Act - Count spaCy calls before second predict
        nlp_call_count_first = mock_nlp_instance.call_count

        # Call predict again with same text
        resolver.predict(texts=[text], references=[[(0, 4)]])
        nlp_call_count_second = mock_nlp_instance.call_count

        # Assert - spaCy should not be called again for the same text
        assert nlp_call_count_second == nlp_call_count_first
        assert text in resolver.doc_objects


@pytest.mark.unit
class TestSentenceTransformerResolverHelperMethods:
    """Test SentenceTransformerResolver helper methods."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_equivalent_gazetteer_searches(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Repeated normalized candidate searches hit the resolver cache."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()
        candidate = SimpleNamespace(id=1)
        resolver.gazetteer.search.return_value = [candidate]

        first = resolver._search_candidates(' "Paris" ', "exact", tiers=1)
        second = resolver._search_candidates("Paris", "exact", tiers=1)

        assert first == second == (candidate,)
        resolver.gazetteer.search.assert_called_once_with("Paris", "exact", tiers=1)

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_candidate_descriptions(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """A feature description is generated once per resolver instance."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()
        candidate = SimpleNamespace(id=1)
        resolver._generate_description = Mock(return_value="Paris (city)")

        first = resolver._candidate_description(candidate)
        second = resolver._candidate_description(candidate)

        assert first == second == "Paris (city)"
        resolver._generate_description.assert_called_once_with(candidate)

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_generate_description_geonames(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _generate_description creates proper descriptions for geonames."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        mock_candidate = Mock()
        mock_candidate.data = {
            "name": "Paris",
            "feature_name": "city",
            "country_name": "France",
            "admin1_name": "Île-de-France",
        }

        # Act
        description = resolver._generate_description(mock_candidate)

        # Assert
        assert "Paris" in description
        assert "city" in description
        assert "France" in description

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_batch_of_only_empty_candidate_lists_scores_nothing(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """References without candidates get empty score lists, no tensor work."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()

        assert resolver._calculate_similarity_batches(["a", "b"], [[], []]) == [[], []]

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_generate_description_handles_missing_attributes(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _generate_description handles missing attributes gracefully."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        mock_candidate = Mock()
        mock_candidate.data = {
            "name": "Paris",
            # Missing feature_name and admin levels
        }

        # Act
        description = resolver._generate_description(mock_candidate)

        # Assert
        assert "Paris" in description
        # Should not crash, just include what's available

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_generate_description_includes_all_admin_levels(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _generate_description includes all available admin levels."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        mock_candidate = Mock()
        mock_candidate.data = {
            "name": "Paris",
            "feature_name": "city",
            "country_name": "France",
            "admin1_name": "Île-de-France",
            "admin2_name": "Paris",
        }

        # Act
        description = resolver._generate_description(mock_candidate)

        # Assert
        assert "Paris" in description
        assert "city" in description
        assert "France" in description
        assert "Île-de-France" in description
        # Should be in hierarchical order: level3, level2, level1
        assert "in" in description

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_generate_description_uses_custom_attribute_map(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _generate_description uses custom attribute_map."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "custom_name",
            "type": "custom_type",
            "level1": "custom_level1",
            "level2": "custom_level2",
            "level3": "custom_level3",
        }

        resolver = SentenceTransformerResolver(
            gazetteer_name="custom_gazetteer", attribute_map=custom_map
        )

        mock_candidate = Mock()
        mock_candidate.data = {
            "custom_name": "Paris",
            "custom_type": "city",
            "custom_level1": "France",
        }

        # Act
        description = resolver._generate_description(mock_candidate)

        # Assert - Should use custom attribute names
        assert "Paris" in description
        assert "city" in description
        assert "France" in description


@pytest.mark.unit
class TestSpacyModelDownload:
    """The fallback that installs the sentence splitter on first use."""

    @patch(
        "geoparser.modules.resolvers.sentencetransformer.load_spacy_model",
        load_spacy_model,
    )
    @patch("geoparser.modules._spacy.spacy.cli.download")
    @patch("geoparser.modules._spacy.spacy.load")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_downloads_then_reloads_the_same_model(
        self,
        mock_gazetteer,
        mock_transformer,
        mock_tokenizer,
        mock_spacy_load,
        mock_download,
    ):
        """
        A missing model is downloaded and then loaded again by name.

        Both the download and the reload must name the model that was asked
        for; reloading something else would leave the resolver splitting
        sentences with the wrong pipeline.
        """
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        reloaded = Mock()
        mock_spacy_load.side_effect = [OSError("not found"), reloaded]

        # Act
        resolver = SentenceTransformerResolver()

        # Assert
        mock_download.assert_called_once_with("xx_sent_ud_sm")
        assert mock_spacy_load.call_args_list == [
            call("xx_sent_ud_sm"),
            call("xx_sent_ud_sm"),
        ]
        assert resolver.nlp is reloaded


@pytest.mark.unit
class TestConfigIdentity:
    """What the resolver records as its configuration."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_a_custom_attribute_map_is_part_of_the_module_identity(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """
        The attribute map reaches the recorded config, so it changes the id.

        Predictions are stored against the module id. Dropping the map from
        the config would make two resolvers that describe candidates
        differently share an id, and their predictions would be conflated.
        """
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        custom_map = {
            "name": "N",
            "type": "T",
            "level1": "A",
            "level2": "B",
            "level3": "C",
        }

        # Act
        default = SentenceTransformerResolver()
        customized = SentenceTransformerResolver(attribute_map=custom_map)

        # Assert
        assert customized.config["attribute_map"] == custom_map
        assert default.config["attribute_map"] is None
        assert customized.id != default.id
