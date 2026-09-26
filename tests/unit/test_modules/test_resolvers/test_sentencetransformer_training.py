"""SentenceTransformerResolver training data (resolvers/_training.py)."""

from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestSentenceTransformerResolverPrepareTrainingData:
    """Test SentenceTransformerResolver _prepare_training_data method."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_creates_training_examples_from_referents(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data creates training examples from referents."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search to return candidates
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate1 = Mock()
        mock_candidate1.identifier = "123"
        mock_candidate1.data = {"name": "Paris", "feature_name": "city"}
        mock_candidate2 = Mock()
        mock_candidate2.identifier = "456"
        mock_candidate2.data = {"name": "Paris", "feature_name": "region"}
        mock_gazetteer_instance.search.return_value = [mock_candidate1, mock_candidate2]

        # Mock _extract_context
        with patch.object(
            resolver, "_extract_context", return_value="Paris is beautiful"
        ):
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]  # "Paris"
            referents = [[("geonames", "123")]]  # Matches candidate1

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            assert "sentence1" in training_data
            assert "sentence2" in training_data
            assert "label" in training_data
            assert len(training_data["sentence1"]) > 0
            assert len(training_data["sentence2"]) > 0
            assert len(training_data["label"]) > 0

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_creates_positive_and_negative_examples(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data creates both positive and negative examples."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search to return multiple candidates
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate1 = Mock()
        mock_candidate1.identifier = "123"
        mock_candidate1.data = {"name": "Paris", "feature_name": "city"}
        mock_candidate2 = Mock()
        mock_candidate2.identifier = "456"
        mock_candidate2.data = {"name": "Paris", "feature_name": "region"}
        mock_gazetteer_instance.search.return_value = [mock_candidate1, mock_candidate2]

        with patch.object(
            resolver, "_extract_context", return_value="Paris is beautiful"
        ):
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]  # Only matches candidate1

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should have 2 examples: 1 positive (label=1) and 1 negative (label=0)
            assert len(training_data["label"]) == 2
            assert 1 in training_data["label"]  # Positive example
            assert 0 in training_data["label"]  # Negative example

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_handles_multiple_references_in_document(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data handles multiple references per document."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(resolver, "_extract_context", return_value="Context"):
            texts = ["Paris and London"]
            references = [[(0, 5), (10, 16)]]  # "Paris", "London"
            referents = [[("geonames", "123"), ("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should have examples for both references
            assert len(training_data["sentence1"]) >= 2

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_handles_multiple_documents(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data handles multiple documents."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(resolver, "_extract_context", return_value="Context"):
            texts = ["Paris is beautiful.", "London is historic."]
            references = [[(0, 5)], [(0, 6)]]  # "Paris", "London"
            referents = [[("geonames", "123")], [("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should have examples from both documents
            assert len(training_data["sentence1"]) >= 2

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_extracts_context_for_each_reference(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data extracts context for each reference."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(
            resolver, "_extract_context", return_value="Extracted context"
        ) as mock_extract:
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # _extract_context should have been called
            mock_extract.assert_called()
            # All sentence1 entries should be the extracted context
            assert all(s1 == "Extracted context" for s1 in training_data["sentence1"])

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_generates_descriptions_for_candidates(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data generates descriptions for candidates."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "Paris", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with (
            patch.object(resolver, "_extract_context", return_value="Context"),
            patch.object(
                resolver, "_generate_description", return_value="Paris (city)"
            ) as mock_generate,
        ):
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # _generate_description should have been called
            mock_generate.assert_called()
            # All sentence2 entries should be the generated description
            assert all(s2 == "Paris (city)" for s2 in training_data["sentence2"])

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_returns_correct_data_structure(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data returns correct data structure."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(resolver, "_extract_context", return_value="Context"):
            texts = ["Paris"]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should be a dict with three keys
            assert isinstance(training_data, dict)
            assert set(training_data.keys()) == {"sentence1", "sentence2", "label"}
            # All lists should have the same length
            assert len(training_data["sentence1"]) == len(training_data["sentence2"])
            assert len(training_data["sentence1"]) == len(training_data["label"])
            # Labels should be 0 or 1
            assert all(label in [0, 1] for label in training_data["label"])
