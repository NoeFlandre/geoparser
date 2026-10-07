"""Recognition with a caller-prepared Otter cross-encoder or bi-encoder.

This module does not download checkpoints or execute remote loading code. The
caller must prepare the model and its tokenizers before constructing the adapter.
"""

import typing as t
from collections.abc import Sequence
from itertools import pairwise
from numbers import Real

from geoparser.modules.recognizers.base import Recognizer


class PredictionOutputError(ValueError):
    """Invalid Otter output, with the original document index and raw evidence.

    ``document_index`` is ``None`` when the model's batch cannot be aligned to
    inputs. In that case ``raw_output`` contains the entire returned batch.
    Otherwise it contains the invalid document's returned entities.
    """

    def __init__(self, message: str, *, document_index: int | None, raw_output: object):
        self.document_index = document_index
        self.raw_output = raw_output
        super().__init__(message)


def _positive_integer(value: t.Any, name: str) -> int:
    if type(value) is not int or value < 1:
        message = f"{name} must be a positive integer."
        raise ValueError(message)
    return value


def _probability(value: t.Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        message = f"{name} must be a finite number between 0 and 1."
        raise TypeError(message)
    if not 0 <= value <= 1:
        message = f"{name} must be a finite number between 0 and 1."
        raise ValueError(message)
    return float(value)


def _nonblank_string(value: t.Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        message = f"{name} must be a nonblank string."
        raise ValueError(message)
    return value


def _entity_types(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        message = "entity_types must be a sequence of distinct entity names."
        raise TypeError(message)
    labels = tuple(_nonblank_string(value, "entity type") for value in values)
    if not labels or len(set(labels)) != len(labels):
        message = "entity_types must contain at least one distinct entity name."
        raise ValueError(message)
    return labels


def _ready_tokenizer(model: t.Any, attribute: str) -> t.Any:
    # Reading the public tokenizer property would invoke upstream's unpinned
    # remote loader. Its plain cache field must already contain the tokenizer.
    tokenizer = vars(model).get(attribute)
    if not callable(tokenizer):
        message = (
            f"Prepare Otter's {attribute} tokenizer from the reviewed local "
            "checkpoint before constructing the recognizer."
        )
        raise TypeError(message)
    return tokenizer


def _prepared_tokenizers(model: t.Any, architecture: str) -> tuple[t.Any, t.Any, int]:
    if architecture == "cross_encoder":
        return _ready_tokenizer(model, "_tokenizer"), None, 0
    if architecture != "bi_encoder":
        message = f"Unsupported Otter architecture: {architecture!r}."
        raise ValueError(message)
    text_tokenizer = _ready_tokenizer(model, "_token_tokenizer")
    type_tokenizer = _ready_tokenizer(model, "_type_tokenizer")
    type_limit = min(
        _positive_integer(type_tokenizer.model_max_length, "type tokenizer limit"),
        _positive_integer(
            model.type_config.max_position_embeddings, "type encoder limit"
        ),
    )
    return text_tokenizer, type_tokenizer, type_limit


def _check_lengths(
    tokenizer: t.Any,
    texts: list[str],
    indices: list[int],
    limit: int,
    kind: str,
) -> None:
    encodings = tokenizer(
        texts, padding=False, truncation=False, add_special_tokens=True
    )
    for index, tokens in zip(indices, encodings["input_ids"], strict=True):
        if len(tokens) > limit:
            message = (
                f"Otter {kind} at index {index} has {len(tokens)} tokens, "
                f"exceeding the limit of {limit}; truncation is not supported."
            )
            raise ValueError(message)


def _input_texts(texts: list[str]) -> list[tuple[int, str]]:
    _check_document_types(texts)
    return [(index, text) for index, text in enumerate(texts) if text.strip()]


def _check_document_types(texts: list[str]) -> None:
    if not isinstance(texts, list):
        message = "texts must be a list of strings."
        raise TypeError(message)
    for text in texts:
        if not isinstance(text, str):
            message = "Every document in texts must be a string."
            raise TypeError(message)


def _align_outputs(raw: object, indices: list[int], count: int) -> list[t.Any]:
    if not isinstance(raw, list) or len(raw) != len(indices):
        message = f"Otter must return one result list for each input index {indices}."
        raise PredictionOutputError(message, document_index=None, raw_output=raw)
    aligned: list[t.Any] = [[] for _ in range(count)]
    for position, entities in enumerate(raw):
        aligned[indices[position]] = entities
    return aligned


def _span_bounds(entity: dict, text: str) -> tuple[int, int]:
    start, end = entity["start"], entity["end"]
    if type(start) is not int or type(end) is not int:
        message = "Entity offsets must be integers, excluding booleans."
        raise ValueError(message)
    if not 0 <= start < end <= len(text):
        message = "Entity offsets must select a nonempty span in the original text."
        raise ValueError(message)
    return start, end


def _disjoint_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    for previous, current in pairwise(spans):
        if previous[1] > current[0]:
            message = "Otter returned distinct overlapping entity spans."
            raise ValueError(message)
    return spans


class OtterRecognizer(Recognizer):
    """Adapt a prepared Otter model to original-text character spans.

    Args:
        model: An already-loaded Otter model with its tokenizer caches filled.
        source: Caller-supplied artifact identity. This value is recorded for
            result separation; the adapter does not verify the model's origin.
        entity_types: Distinct natural-language entity types to recognize.
        threshold: Keep scores strictly above this value. ``None`` selects
            the model's configured prediction threshold.
        batch_size: Maximum number of texts in each upstream inference batch.

    The model's configured maximum sequence length includes special tokens
    and, for cross-encoders, the entity-type prompt. Overlength inputs raise
    before inference. No text is normalized, truncated, or split.

    ``last_raw_predictions`` retains the latest returned document entities in
    original input order. Save it before the next call for benchmark evidence.
    Empty documents have empty entries. A preflight or model failure leaves it
    empty. Instances are not intended for concurrent prediction calls.
    """

    NAME = "OtterRecognizer"
    DEFAULT_ENTITY_TYPES: t.ClassVar[tuple[str, ...]] = ("city", "country", "location")

    def __init__(
        self,
        model: t.Any,
        *,
        source: str,
        entity_types: Sequence[str] = DEFAULT_ENTITY_TYPES,
        threshold: float | None = None,
        batch_size: int = 8,
    ):
        self.model = model
        self.source = _nonblank_string(source, "source")
        self.entity_types = _entity_types(entity_types)
        self.architecture = model.config.architecture
        self._text_tokenizer, self._type_tokenizer, self._type_limit = (
            _prepared_tokenizers(model, self.architecture)
        )
        self.threshold = _probability(
            model.config.prediction_threshold if threshold is None else threshold,
            "threshold",
        )
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.max_seq_length = _positive_integer(
            model.config.max_seq_length, "max_seq_length"
        )
        self.max_span_length = _positive_integer(
            model.config.max_span_length, "max_span_length"
        )
        self.last_raw_predictions: list[t.Any] = []
        super().__init__(
            source=self.source,
            provenance="caller-supplied",
            architecture=self.architecture,
            entity_types=self.entity_types,
            threshold=self.threshold,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            max_span_length=self.max_span_length,
            type_max_seq_length=self._type_limit or None,
        )

    def predict(self, texts: list[str]) -> list[list[tuple[int, int]] | None]:
        """Return sorted, distinct spans, or raise with invalid-output evidence."""
        return list(self._predict_spans(texts))

    def predict_batch(self, texts: list[str]) -> list[set[tuple[int, int]]]:
        """Return the same spans as sets for set-based benchmark predictors."""
        return [set(spans) for spans in self._predict_spans(texts)]

    def _predict_spans(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        self.last_raw_predictions = []
        indexed = _input_texts(texts)
        self._preflight(indexed)
        raw = self._predict_raw(indexed)
        self.last_raw_predictions = _align_outputs(
            raw, [index for index, _ in indexed], len(texts)
        )
        return [
            self._document_spans(text, self.last_raw_predictions[index], index)
            for index, text in enumerate(texts)
        ]

    def _preflight(self, indexed: list[tuple[int, str]]) -> None:
        if not indexed:
            return
        self._check_label_lengths()
        prefix = (
            self.model.build_prompt(self.entity_types)
            if self.architecture == "cross_encoder"
            else ""
        )
        for offset in range(0, len(indexed), self.batch_size):
            self._check_text_batch(indexed[offset : offset + self.batch_size], prefix)

    def _check_label_lengths(self) -> None:
        if self._type_tokenizer is not None:
            _check_lengths(
                self._type_tokenizer,
                list(self.entity_types),
                list(range(len(self.entity_types))),
                self._type_limit,
                "entity type",
            )

    def _check_text_batch(self, batch: list[tuple[int, str]], prefix: str) -> None:
        _check_lengths(
            self._text_tokenizer,
            [prefix + text for _, text in batch],
            [index for index, _ in batch],
            self.max_seq_length,
            "document",
        )

    def _predict_raw(self, indexed: list[tuple[int, str]]) -> object:
        if not indexed:
            return []
        return self.model.predict(
            [text for _, text in indexed],
            labels=list(self.entity_types),
            threshold=self.threshold,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
        )

    def _document_spans(
        self, text: str, entities: object, index: int
    ) -> list[tuple[int, int]]:
        try:
            return self._validated_spans(text, entities)
        except (KeyError, TypeError, ValueError) as error:
            message = f"Invalid Otter output for document at index {index}: {error}"
            raise PredictionOutputError(
                message, document_index=index, raw_output=entities
            ) from error

    def _validated_spans(self, text: str, entities: object) -> list[tuple[int, int]]:
        if not isinstance(entities, list):
            message = "Each document result must be a list of entities."
            raise TypeError(message)
        candidates = [self._entity_span(text, entity) for entity in entities]
        spans = sorted({span for span, score in candidates if score > self.threshold})
        return _disjoint_spans(spans)

    def _entity_span(self, text: str, entity: object) -> tuple[tuple[int, int], float]:
        if not isinstance(entity, dict):
            message = "Each entity must be a dictionary."
            raise TypeError(message)
        start, end = _span_bounds(entity, text)
        if entity["text"] != text[start:end]:
            message = "Entity text does not match its offsets in the original text."
            raise ValueError(message)
        if entity["label"] not in self.entity_types:
            message = "Entity label was not requested."
            raise ValueError(message)
        return (start, end), _probability(entity["score"], "entity score")
