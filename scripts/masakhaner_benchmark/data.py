"""Parse MasakhaNER 2.0 token files and map LOC tags to exact character spans.

The upstream files hold one token and one tag per line, separated by a single
space, with a blank line between sentences. The tags are BIO tags over four
entity types. Only LOC is a location. PER, ORG and DATE are not scored here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

LABEL_NAMES = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-DATE",
    "I-DATE",
)
LOCATION_BEGIN = "B-LOC"
LOCATION_INSIDE = "I-LOC"
DEVELOPMENT_SPLIT = "validation"
EVALUATION_SPLIT = "test"
_SPLIT_BY_PURPOSE = {
    "development": DEVELOPMENT_SPLIT,
    "evaluation": EVALUATION_SPLIT,
}

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Sentence:
    """One pre-tokenized sentence with its upstream BIO tags."""

    tokens: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Example:
    """One sentence with gold location spans in Python character offsets."""

    language: str
    text: str
    gold_spans: frozenset[Span]
    malformed_location_tags: int = 0


def git_blob_sha1(content: bytes) -> str:
    """Return the git object id of a file body, as GitHub reports blob ids."""
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def split_for(purpose: str) -> str:
    """Return the only split that a purpose may read.

    Training data is never scored, and development and evaluation never share a
    split. An unknown purpose fails rather than defaulting to a split.
    """
    try:
        return _SPLIT_BY_PURPOSE[purpose]
    except KeyError:
        message = (
            f"Unknown purpose {purpose!r}; use 'development' or 'evaluation'. "
            "The training split is never used for scoring."
        )
        raise ValueError(message) from None


def parse_sentences(lines: Iterable[str]) -> list[Sentence]:
    """Group token-tag lines into sentences separated by blank lines.

    Each non-blank line must hold exactly one token and one known tag, separated
    by a single space. Any other shape is rejected with its line number, so a
    format change upstream cannot silently shift the tokens.
    """
    sentences: list[Sentence] = []
    tokens: list[str] = []
    tags: list[str] = []
    for number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip(" \t\r\n")
        if not line.strip():
            if tokens:
                sentences.append(Sentence(tuple(tokens), tuple(tags)))
            tokens, tags = [], []
            continue
        token, tag = _token_and_tag(line, number)
        tokens.append(token)
        tags.append(tag)
    if tokens:
        sentences.append(Sentence(tuple(tokens), tuple(tags)))
    return sentences


def _token_and_tag(line: str, number: int) -> tuple[str, str]:
    """Split one token line and validate both fields."""
    fields = line.split(" ")
    if len(fields) != 2 or not fields[0]:
        message = f"Line {number} must hold one token and one tag, separated by a space"
        raise ValueError(message)
    token, tag = fields
    if tag not in LABEL_NAMES:
        message = f"Line {number} has an unknown tag: {tag!r}"
        raise ValueError(message)
    return token, tag


def token_offsets(tokens: Sequence[str]) -> list[Span]:
    """Return half-open offsets of tokens joined by one space each."""
    offsets: list[Span] = []
    cursor = 0
    for index, token in enumerate(tokens):
        start = cursor + int(index > 0)
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def location_spans(
    tags: Sequence[str], offsets: Sequence[Span]
) -> tuple[set[Span], int]:
    """Join LOC begin and inside tags into spans and count orphan inside tags.

    An inside tag with no open span starts one and is counted as malformed. This
    matches the WikiANN handling used by the PAN-X benchmark.
    """
    spans: set[Span] = set()
    open_span: Span | None = None
    malformed = 0
    for tag, (start, end) in zip(tags, offsets, strict=True):
        if tag == LOCATION_INSIDE and open_span is not None:
            open_span = (open_span[0], end)
            continue
        if open_span is not None:
            spans.add(open_span)
            open_span = None
        if tag == LOCATION_INSIDE:
            malformed += 1
        if tag in (LOCATION_BEGIN, LOCATION_INSIDE):
            open_span = (start, end)
    if open_span is not None:
        spans.add(open_span)
    return spans, malformed


def example_from_sentence(language: str, sentence: Sentence) -> Example:
    """Join tokens with single spaces and map LOC tags to character spans.

    The text is not normalized. Offsets count Python code points, so combining
    marks in the tokens count as characters. The text is the tokens joined by
    spaces, not the original article text.
    """
    text = " ".join(sentence.tokens)
    spans, malformed = location_spans(sentence.tags, token_offsets(sentence.tokens))
    return Example(language, text, frozenset(spans), malformed)


def examples_from_text(language: str, content: str) -> list[Example]:
    """Parse one split file body into examples for one language.

    The body is split on newline characters only. Other separators that Python
    treats as line breaks can occur inside Unicode tokens and must not split them.
    """
    return [
        example_from_sentence(language, sentence)
        for sentence in parse_sentences(content.split("\n"))
    ]
