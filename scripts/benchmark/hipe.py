"""
Read HIPE-2022 TSV files into benchmark documents.

HIPE (Ehrmann et al., *Extended Overview of HIPE-2022*) annotates historical
newspapers in several languages, token by token, and links each named entity
to Wikidata. The parser rebuilds each document's text from its tokens, keeps
the entities whose literal coarse type is ``loc``, and places them with the
coordinates of the Wikidata item they are linked to.

Like the GeoVirus reader, this module holds no models, torch or gazetteer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from scripts.benchmark.corpus import Document, GoldSpan

DOCUMENT_ID = "# hipe2022:document_id = "
TAG_COLUMN = "NE-COARSE-LIT"
LINK_COLUMN = "NEL-LIT"
MISC_COLUMN = "MISC"
LOCATION = "loc"

Coordinates = Mapping[str, tuple[float, float]]


@dataclass
class _Entity:
    """An entity being read, with its character offsets and Wikidata link."""

    start: int
    end: int
    qid: str


@dataclass
class _Block:
    """One document's text and location entities, as they are read."""

    identifier: str
    text: str = ""
    entities: list[_Entity] = field(default_factory=list)
    space_pending: bool = False


def _column_indices(header: list[str]) -> tuple[int, int, int]:
    """Find the TSV columns used for entity labels, links and spacing."""
    return (
        header.index(TAG_COLUMN),
        header.index(LINK_COLUMN),
        header.index(MISC_COLUMN),
    )


def _skip_line(line: str, block: _Block | None) -> bool:
    """Whether a line is a comment, separator or outside a document block."""
    return not line or line.startswith("#") or block is None


def _new_block(line: str) -> _Block:
    """Start a HIPE document block from its metadata line."""
    return _Block(line[len(DOCUMENT_ID) :].strip())


def _append_token(
    block: _Block,
    columns: list[str],
    tag_index: int,
    link_index: int,
    misc_index: int,
    current: _Entity | None,
) -> _Entity | None:
    """Append one token and update the open location entity, if any."""
    if block.space_pending:
        block.text += " "
    start = len(block.text)
    block.text += columns[0]
    block.space_pending = "NoSpaceAfter" not in columns[misc_index]

    label = columns[tag_index].lower()
    if label == f"i-{LOCATION}" and current is not None:
        current.end = len(block.text)
    elif label == f"b-{LOCATION}":
        current = _Entity(start, len(block.text), columns[link_index])
        block.entities.append(current)
    else:
        current = None
    return current


def _blocks(path: Path) -> Iterator[_Block]:
    """Yield every document in the file with its rebuilt text and entities."""
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        tag, link, misc = _column_indices(header)
        yield from _read_document_blocks(handle, tag, link, misc)


def _read_document_blocks(
    handle: TextIO, tag: int, link: int, misc: int
) -> Iterator[_Block]:
    """Rebuild document boundaries and token spans from the TSV body."""
    block: _Block | None = None
    current: _Entity | None = None
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        if line.startswith(DOCUMENT_ID):
            if block is not None:
                yield block
            block = _new_block(line)
            current = None
            continue
        current = _append_line_token(block, line, tag, link, misc, current)
    if block is not None:
        yield block


def _append_line_token(
    block: _Block | None,
    line: str,
    tag: int,
    link: int,
    misc: int,
    current: _Entity | None,
) -> _Entity | None:
    """Append a token only when the line is data inside a document."""
    if block is None:
        return current
    if _skip_line(line, block):
        return current
    return _append_token(block, line.split("\t"), tag, link, misc, current)


def hipe_qids(path: Path) -> set[str]:
    """Return the Wikidata items the file's location entities link to."""
    return {
        entity.qid
        for block in _blocks(path)
        for entity in block.entities
        if entity.qid.startswith("Q")
    }


def parse_hipe(
    path: Path, coordinates: Coordinates, *, limit: int | None = None
) -> list[Document]:
    """
    Read a HIPE TSV file into documents with gold spans.

    A location linked to NIL, or to an item Wikidata gives no coordinates for,
    is dropped: distance cannot be scored without a place on Earth.

    Args:
        path: The HIPE TSV file
        coordinates: Wikidata QID to (latitude, longitude)
        limit: Keep only the first this many documents, for a shorter run

    Returns:
        The parsed documents, in corpus order
    """
    documents: list[Document] = []
    for block in _blocks(path):
        document = _document_from_block(block, coordinates)
        if document is not None:
            documents.append(document)
        if limit is not None and len(documents) >= limit:
            break
    return documents


def _document_from_block(block: _Block, coordinates: Coordinates) -> Document | None:
    """Keep the coordinate-backed location spans from one parsed block."""
    gold = tuple(
        GoldSpan(
            entity.start,
            entity.end,
            block.text[entity.start : entity.end],
            *coordinates[entity.qid],
        )
        for entity in block.entities
        if entity.qid in coordinates
    )
    if not gold:
        return None
    return Document(block.identifier, block.text, gold)
