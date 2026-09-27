"""SentenceTransformerResolver context windows (resolvers/_context.py)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestContextWindow:
    """Sizing the context each reference is embedded with."""

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_caches_measured_sentences(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Long-document sentence/token measurement is reused per text."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver()
        resolver._sentences = Mock(
            return_value=[
                SimpleNamespace(text="Paris", start_char=0, end_char=5),
            ]
        )
        resolver._sentence_tokens = Mock(return_value=3)

        first = resolver._measured_sentences("Paris is here")
        second = resolver._measured_sentences("Paris is here")

        assert first == second
        resolver._sentences.assert_called_once_with("Paris is here")
        resolver._sentence_tokens.assert_called_once_with(
            resolver._sentences.return_value[0]
        )

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_extract_context_returns_full_text_when_within_limit(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _extract_context returns full text when within token limit."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512

        mock_tokenizer_instance = mock_tokenizer.return_value
        # Short text - only 3 tokens
        mock_tokenizer_instance.tokenize.return_value = ["Paris", "is", "beautiful"]

        resolver = SentenceTransformerResolver()

        text = "Paris is beautiful"

        # Act
        context = resolver._extract_context(text, 0, 5)

        # Assert
        assert context == text

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_extract_context_expands_bidirectionally(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _extract_context expands context bidirectionally around reference."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512

        mock_tokenizer_instance = mock_tokenizer.return_value

        # Return different lengths for different calls
        def tokenize_side_effect(text):
            # Simulate that full text is too long
            if len(text) > 20:
                return ["token"] * 600  # Exceeds limit
            # But sentences are short
            return ["token"] * 3

        mock_tokenizer_instance.tokenize.side_effect = tokenize_side_effect

        # Mock spaCy sentence splitter
        mock_nlp = Mock()
        mock_sent1 = Mock()
        mock_sent1.start_char = 0
        mock_sent1.end_char = (
            26  # Cover the full text including "Paris" at position 10-15
        )
        mock_sent1.text = "I went to Paris yesterday."

        mock_doc = Mock()
        mock_doc.sents = [mock_sent1]
        mock_nlp.return_value = mock_doc

        resolver = SentenceTransformerResolver()
        resolver.nlp = mock_nlp

        text = "I went to Paris yesterday."

        # Act
        context = resolver._extract_context(text, 10, 15)  # "Paris"

        # Assert
        # Should at least include the sentence containing the reference
        assert isinstance(context, str)
        # Should contain the reference text
        assert "Paris" in context or context == text

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_extract_context_reports_reference_outside_every_sentence(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """A reference offset no sentence covers is reported clearly."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer_instance = mock_transformer.return_value
        mock_transformer_instance.get_max_seq_length.return_value = 512

        mock_tokenizer_instance = mock_tokenizer.return_value

        def tokenize_side_effect(text):
            # Force the sentence-splitting path by making the document too long
            return ["token"] * 600 if len(text) > 20 else ["token"] * 3

        mock_tokenizer_instance.tokenize.side_effect = tokenize_side_effect

        # A sentence that stops well before the requested offset, which is what
        # a bad reference span or an uncovered gap between sentences looks like.
        mock_sent = Mock()
        mock_sent.start_char = 0
        mock_sent.end_char = 10
        mock_sent.text = "I went to."

        mock_doc = Mock()
        mock_doc.sents = [mock_sent]
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc

        resolver = SentenceTransformerResolver()
        resolver.nlp = mock_nlp

        text = "I went to Paris yesterday and it was lovely."

        # Act & Assert - a clear message, not "None is not in list"
        with pytest.raises(ValueError, match="No sentence contains reference"):
            resolver._extract_context(text, 10, 15)

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_extract_context_requires_a_maximum_sequence_length(
        self, mock_gazetteer, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """A model that advertises no maximum length is reported clearly."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        mock_transformer.return_value.get_max_seq_length.return_value = None

        resolver = SentenceTransformerResolver()

        # Act & Assert
        with pytest.raises(ValueError, match="does not report a maximum sequence"):
            resolver._extract_context("Some text", 0, 4)
