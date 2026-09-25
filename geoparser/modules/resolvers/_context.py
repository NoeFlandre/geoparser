"""Context windows for SentenceTransformerResolver.

Sizing a reference's context needs only the resolver's tokenizer, its spaCy
sentence splitter and its model's sequence limit; this mixin holds that text
logic apart from the search loop and the training code.
"""

import spacy
import spacy.tokens
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from geoparser.modules.resolvers.context import Sentence, select_context


class ContextWindowMixin:
    """Build the text each reference is embedded with, within the token budget."""

    model_name: str
    transformer: SentenceTransformer
    tokenizer: PreTrainedTokenizerBase
    nlp: spacy.language.Language
    doc_tokens: dict[str, int]
    doc_objects: dict[str, spacy.tokens.Doc]
    measured_sentences: dict[str, tuple[Sentence, ...]]

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """
        Extract context around a single reference, respecting model token limits.

        The whole document is used when it fits. Otherwise the choice of which
        sentences to keep is made by :func:`~geoparser.modules.resolvers.context.select_context`
        over this document's measured sentences.

        Args:
            text: Full document text
            start: Start position of the reference
            end: End position of the reference

        Returns:
            Context string for the reference
        """
        token_limit = self._token_limit()

        if self._document_tokens(text) <= token_limit:
            return text

        return select_context(self._measured_sentences(text), start, end, token_limit)

    def _measured_sentences(self, text: str) -> tuple[Sentence, ...]:
        """
        The document's sentences, priced in encoder tokens.

        This is the adapter between the models this resolver loads and the
        plain arithmetic that sizes a context: spaCy supplies the spans, the
        tokenizer supplies the costs, and everything downstream sees neither.

        Args:
            text: Full document text

        Returns:
            One Sentence per sentence of the document, in order
        """
        if text not in self.measured_sentences:
            self.measured_sentences[text] = tuple(
                Sentence(
                    text=sent.text,
                    start=sent.start_char,
                    end=sent.end_char,
                    cost=self._sentence_tokens(sent),
                )
                for sent in self._sentences(text)
            )
        return self.measured_sentences[text]

    def _token_limit(self) -> int:
        """
        The number of tokens available for a context.

        Returns:
            The model's maximum sequence length, less the special tokens
            ([CLS] and [SEP] for BERT-like models)

        Raises:
            ValueError: If the model advertises no maximum sequence length
        """
        max_seq_length = self.transformer.get_max_seq_length()
        if max_seq_length is None:
            # pragma: no mutate start - wording only; a test pins the type and
            # that the message names the model.
            raise ValueError(
                f"Model '{self.model_name}' does not report a maximum sequence "
                "length, so reference context cannot be sized"
            )
            # pragma: no mutate end
        return max_seq_length - 2

    def _document_tokens(self, text: str) -> int:
        """
        The token count of a whole document, computed once per document.

        Args:
            text: Full document text

        Returns:
            Number of tokens in the document
        """
        if text not in self.doc_tokens:
            self.doc_tokens[text] = len(self.tokenizer.tokenize(text))
        return self.doc_tokens[text]

    def _sentences(self, text: str) -> list["spacy.tokens.Span"]:
        """
        The document's sentences, parsed once per document.

        Args:
            text: Full document text

        Returns:
            The document's sentence spans, in order
        """
        if text not in self.doc_objects:
            self.doc_objects[text] = self.nlp(text)
        return list(self.doc_objects[text].sents)

    def _sentence_tokens(self, sentence: "spacy.tokens.Span") -> int:
        """
        The token cost of one sentence.

        Args:
            sentence: The sentence to measure

        Returns:
            Number of tokens the encoder would spend on it
        """
        return len(self.tokenizer.tokenize(sentence.text))
