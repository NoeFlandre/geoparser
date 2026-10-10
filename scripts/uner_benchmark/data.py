"""Read UNER IOB2 sentences without changing their source text."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from scripts.panx_benchmark.data import Example, Span

SPLITS = frozenset({"train", "dev", "test"})
LABELS = frozenset(
    {"O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-OTH", "I-OTH"}
)


@dataclass(frozen=True)
class Sentence:
    """A sentence with its source IDs and standard recognition input."""

    identifier: str
    source_document_id: str | None
    sentence_id: str
    example: Example
    token_count: int

    @property
    def document_id(self) -> str:
        """Return the explicit source ID, or this sentence's fallback ID."""
        return (
            self.sentence_id
            if self.source_document_id is None
            else self.source_document_id
        )


@dataclass(frozen=True)
class Corpus:
    """Every sentence in one validated source configuration and split."""

    sentences: tuple[Sentence, ...]

    @property
    def sentence_count(self) -> int:
        """Return the full source sentence denominator."""
        return len(self.sentences)

    @property
    def document_count(self) -> int:
        """Count distinct source documents without joining their text."""
        return len(
            {
                (sentence.source_document_id is None, sentence.document_id)
                for sentence in self.sentences
            }
        )

    @property
    def token_count(self) -> int:
        """Count all tokens, including non-location tokens."""
        return sum(sentence.token_count for sentence in self.sentences)

    @property
    def location_count(self) -> int:
        """Count source LOC entities."""
        return sum(len(sentence.example.gold_spans) for sentence in self.sentences)

    @property
    def empty_text_count(self) -> int:
        """Count explicit empty source sentences."""
        return sum(sentence.example.text == "" for sentence in self.sentences)


@dataclass
class _Block:
    """One sentence's metadata and token rows."""

    metadata: dict[str, str] = field(default_factory=dict)
    rows: list[str] = field(default_factory=list)

    def comment(self, line: str) -> None:
        """Keep source metadata while rejecting misplaced or duplicate fields."""
        item = _metadata_item(line)
        if item is None:
            return
        key, value = item
        if self.rows:
            message = "Sentence metadata follows token rows without a boundary"
            raise ValueError(message)
        if key in self.metadata:
            message = f"duplicate metadata: {key}"
            raise ValueError(message)
        self.metadata[key] = value

    def consume(self, line: str) -> None:
        """Read a comment or preserve a complete token row."""
        if line.startswith("#"):
            self.comment(line)
        else:
            self.rows.append(line)


def _validate_metadata_comment(line: str) -> None:
    """Reject malformed reserved metadata instead of losing its boundaries."""
    fields = line[1:].split(maxsplit=1)
    if not fields:
        return
    prefixes = {
        "newdoc": "# newdoc id =",
        "sent_id": "# sent_id = ",
        "text": "# text = ",
    }
    expected = prefixes.get(fields[0])
    if expected is not None and not line.startswith(expected):
        message = "Malformed reserved metadata or document boundary"
        raise ValueError(message)


def _metadata_item(line: str) -> tuple[str, str] | None:
    """Read reserved metadata, including released no-space document IDs."""
    _validate_metadata_comment(line)
    key, separator, value = line[2:].partition(" =")
    if not separator or key not in {"text", "sent_id", "newdoc id"}:
        return None
    return key, value.removeprefix(" ")


def _blocks(source: str) -> Iterator[_Block]:
    """Yield nonempty sentence blocks and tolerate blank-line separators."""
    block = _Block()
    for raw_line in source.split("\n"):
        line = raw_line.removesuffix("\r")
        if line:
            block.consume(line)
        else:
            yield block
            block = _Block()
    yield block


def _required_metadata(block: _Block) -> tuple[str, str]:
    """Require explicit text and a nonempty source sentence ID."""
    sentence_id = block.metadata.get("sent_id", "")
    if not sentence_id:
        message = "Missing or empty sent_id"
        raise ValueError(message)
    if "text" not in block.metadata:
        message = "Missing source text metadata"
        raise ValueError(message)
    return sentence_id, block.metadata["text"]


def _row(line: str, expected_id: int) -> tuple[str, str]:
    """Validate the five release columns and sequential integer IDs."""
    columns = line.split("\t")
    if len(columns) != 5:
        message = "UNER token rows require five tab-separated columns"
        raise ValueError(message)
    if columns[0] != str(expected_id):
        message = "Unsupported or non-sequential token id"
        raise ValueError(message)
    token, label = columns[1:3]
    _validate_token_label(token, label)
    return token, label


def _validate_token_label(token: str, label: str) -> None:
    """Require a literal token and one of the released entity labels."""
    if not token or token.isspace():
        message = "UNER contains an empty token"
        raise ValueError(message)
    if label not in LABELS:
        message = f"Unknown UNER label: {label}"
        raise ValueError(message)


def _aligned_offset(text: str, token: str, cursor: int) -> Span:
    """Match the next literal token, permitting only intervening whitespace."""
    start = cursor + len(text[cursor:]) - len(text[cursor:].lstrip())
    if not text.startswith(token, start):
        message = f"Cannot align source token at character {start}"
        raise ValueError(message)
    return start, start + len(token)


def _validate_iob2(label: str, previous: str) -> None:
    """Reject orphan or cross-type inside labels for every entity class."""
    if label.startswith("I-") and previous not in {"B" + label[1:], label}:
        message = f"Malformed IOB2 transition: {previous} to {label}"
        raise ValueError(message)


@dataclass
class _Locations:
    """Accumulate strict location spans while keeping adjacent starts apart."""

    spans: list[Span] = field(default_factory=list)

    def consume(self, label: str, offset: Span) -> None:
        """Append a location or extend a validated inside label."""
        if label == "B-LOC":
            self.spans.append(offset)
        elif label == "I-LOC":
            self.spans[-1] = self.spans[-1][0], offset[1]


def _example(block: _Block, text: str, language: str) -> Example:
    """Convert strict IOB2 labels into original Python character spans."""
    locations = _Locations()
    previous, cursor = "O", 0
    for index, line in enumerate(block.rows, start=1):
        token, label = _row(line, index)
        _validate_iob2(label, previous)
        offset = _aligned_offset(text, token, cursor)
        locations.consume(label, offset)
        previous, cursor = label, offset[1]
    if text[cursor:].strip():
        message = "Source text has unmatched characters after the final token"
        raise ValueError(message)
    return Example(language, text, frozenset(locations.spans))


def _document_id(block: _Block, current: str | None) -> str | None:
    """Retain source document boundaries or use sentence IDs when absent."""
    document_id = block.metadata.get("newdoc id", current)
    if document_id == "":
        message = "Empty source document id"
        raise ValueError(message)
    return document_id


def _sentences(source: str, language: str, prefix: str) -> Iterator[Sentence]:
    """Build every sentence once and reject duplicate source IDs."""
    current_document = None
    seen: set[str] = set()
    for block in _blocks(source):
        if not block.rows and not block.metadata:
            continue
        sentence_id, text = _required_metadata(block)
        if sentence_id in seen:
            message = f"duplicate sentence id: {sentence_id}"
            raise ValueError(message)
        seen.add(sentence_id)
        current_document = _document_id(block, current_document)
        yield Sentence(
            f"{prefix}/{sentence_id}",
            current_document,
            sentence_id,
            _example(block, text, language),
            len(block.rows),
        )


def parse_iob2(source: str, *, language: str, configuration: str, split: str) -> Corpus:
    """Parse one split or fail before returning any partial benchmark corpus.

    Empty sentences and sentences without locations stay in the denominator.
    Token forms must align literally with the release's ``# text`` field.
    No normalization, whitespace reconstruction, orphan-label repair, entity
    linking, or coordinate assignment is performed.
    """
    if split not in SPLITS:
        message = f"Unknown UNER split: {split}"
        raise ValueError(message)
    return Corpus(tuple(_sentences(source, language, f"{configuration}/{split}")))
