"""Parse MultiCoNER II CoNLL-style sentences into exact character spans.

Each sentence starts with ``# id <sample-id>\\tdomain=<code>``, has one
``token _ _ TAG`` line per token, and ends at a blank line. Spans are Python
code-point offsets into the text made by joining the tokens with one space,
because the release carries no original whitespace. Malformed sentences are
returned as ``InvalidRecord`` values and counted. They are never dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scripts.multiconer_benchmark.label_policy import KNOWN_LABELS, label_for

Span = tuple[int, int]
_HEADER_PREFIXES = ("# id ", "# id\t")
_TAG = re.compile(r"^([BI])-(.+)$")
_COLUMN_SPLIT = re.compile(r"[ \t]+")


@dataclass(frozen=True, slots=True)
class Entity:
    """One fine-grained gold entity in character offsets, half-open."""

    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Sentence:
    """A valid sentence with its reconstructed text and retained gold."""

    sample_id: str
    domain: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    text: str
    offsets: tuple[Span, ...]
    entities: tuple[Entity, ...]

    def location_spans(self) -> frozenset[Span]:
        """Return the exact spans whose fine-grained label is a place."""
        return frozenset(
            (entity.start, entity.end)
            for entity in self.entities
            if label_for(entity.label) == "LOC"
        )


@dataclass(frozen=True, slots=True)
class InvalidRecord:
    """A sentence that was rejected, with the first line and reason it failed."""

    line: int
    sample_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class ParsedSource:
    """Every sentence block of one file, valid or invalid."""

    sentences: tuple[Sentence, ...]
    invalid: tuple[InvalidRecord, ...]

    @property
    def record_count(self) -> int:
        """Count every block, so valid and invalid records add up."""
        return len(self.sentences) + len(self.invalid)


@dataclass(slots=True)
class _Block:
    """Lines of one blank-line-delimited sentence block, with line numbers."""

    header: tuple[int, str] | None = None
    tokens: list[tuple[int, str]] = field(default_factory=list)


def reconstruct_text(tokens: tuple[str, ...] | list[str]) -> str:
    """Join tokens with one space; this is the shared text every system sees."""
    return " ".join(tokens)


def require_no_invalid_records(parsed: ParsedSource) -> None:
    """Refuse to freeze a source whose denominator would silently change."""
    if parsed.invalid:
        message = (
            f"source has {len(parsed.invalid)} invalid records; "
            "freeze only a clean source"
        )
        raise ValueError(message)


def parse_conll(text: str, *, expected_domain: str | None = None) -> ParsedSource:
    """Parse one split file and report every rejected block."""
    body = text[1:] if text.startswith("\ufeff") else text
    sentences: list[Sentence] = []
    invalid: list[InvalidRecord] = []
    seen_ids: set[str] = set()
    for block in _blocks(body.split("\n")):
        outcome = _parse_block(block, expected_domain, seen_ids)
        if isinstance(outcome, Sentence):
            sentences.append(outcome)
        else:
            invalid.append(outcome)
    return ParsedSource(tuple(sentences), tuple(invalid))


def _blocks(lines: list[str]) -> list[_Block]:
    """Group lines into blocks; a blank line or a new header closes a block."""
    blocks: list[_Block] = []
    current: _Block | None = None
    for number, raw in enumerate(lines, start=1):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip(" \t"):
            if current is not None:
                blocks.append(current)
                current = None
            continue
        if line.startswith(_HEADER_PREFIXES):
            if current is not None:
                blocks.append(current)
            current = _Block(header=(number, line))
            continue
        if current is None:
            current = _Block()
        current.tokens.append((number, line))
    if current is not None:
        blocks.append(current)
    return blocks


def _parse_block(
    block: _Block, expected_domain: str | None, seen_ids: set[str]
) -> Sentence | InvalidRecord:
    """Validate one block in order and stop at its first defect."""
    if block.header is None:
        return InvalidRecord(block.tokens[0][0], None, "missing sentence header")
    number, header = block.header
    checked = _read_header(number, header, expected_domain, seen_ids)
    if isinstance(checked, InvalidRecord):
        return checked
    sample_id, domain = checked
    if not block.tokens:
        return InvalidRecord(number, sample_id, "empty sentence")
    return _sentence_from_tokens(block.tokens, sample_id, domain)


def _read_header(
    number: int, header: str, expected_domain: str | None, seen_ids: set[str]
) -> tuple[str, str] | InvalidRecord:
    """Return the sample id and domain, or the first header defect."""
    fields = header.split()
    if len(fields) < 3:
        return InvalidRecord(number, None, "missing sample id")
    sample_id = fields[2]
    if len(fields) > 4:
        return InvalidRecord(number, sample_id, "malformed header")
    if len(fields) < 4 or not fields[3].startswith("domain=") or fields[3] == "domain=":
        return InvalidRecord(number, sample_id, "missing domain")
    domain = fields[3][len("domain=") :]
    if sample_id in seen_ids:
        return InvalidRecord(number, sample_id, "duplicate sample id")
    seen_ids.add(sample_id)
    if expected_domain is not None and domain != expected_domain:
        reason = f"domain mismatch: expected {expected_domain}, found {domain}"
        return InvalidRecord(number, sample_id, reason)
    return sample_id, domain


def _sentence_from_tokens(
    lines: list[tuple[int, str]], sample_id: str, domain: str
) -> Sentence | InvalidRecord:
    """Read token lines, check every column, then build spans."""
    tokens: list[str] = []
    tags: list[str] = []
    previous: str | None = None
    for number, line in lines:
        columns = _COLUMN_SPLIT.split(line.strip(" \t"))
        if len(columns) != 4:
            reason = f"wrong column count: expected 4, found {len(columns)}"
            return InvalidRecord(number, sample_id, reason)
        token, left, right, tag = columns
        if left != "_" or right != "_":
            return InvalidRecord(number, sample_id, "invalid separator columns")
        error, previous = _check_tag(tag, previous)
        if error is not None:
            return InvalidRecord(number, sample_id, error)
        tokens.append(token)
        tags.append(tag)
    offsets = _offsets(tokens)
    return Sentence(
        sample_id=sample_id,
        domain=domain,
        tokens=tuple(tokens),
        tags=tuple(tags),
        text=reconstruct_text(tokens),
        offsets=tuple(offsets),
        entities=tuple(_entities(tags, offsets)),
    )


def _check_tag(tag: str, previous: str | None) -> tuple[str | None, str | None]:
    """Validate one BIO tag; return the error and the label it leaves open."""
    if tag == "O":
        return None, None
    match = _TAG.match(tag)
    if match is None:
        return f"unknown tag format: {tag}", None
    kind, label = match.group(1), match.group(2)
    if label not in KNOWN_LABELS:
        return f"unknown entity type: {label}", None
    if kind == "I" and previous != label:
        return "I- tag does not continue an entity", None
    return None, label


def _offsets(tokens: list[str]) -> list[Span]:
    """Return half-open code-point offsets for the space-joined text."""
    offsets: list[Span] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            cursor += 1
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    return offsets


def _entities(tags: list[str], offsets: list[Span]) -> list[Entity]:
    """Group validated BIO tags into character-offset entities."""
    entities: list[Entity] = []
    open_label: str | None = None
    open_start = 0
    open_end = 0

    def close() -> None:
        if open_label is not None:
            entities.append(
                Entity(open_label, offsets[open_start][0], offsets[open_end][1])
            )

    for index, tag in enumerate(tags):
        if tag == "O":
            close()
            open_label = None
            continue
        kind, label = tag[0], tag[2:]
        if kind == "B" or label != open_label:
            close()
            open_label, open_start = label, index
        open_end = index
    close()
    return entities
