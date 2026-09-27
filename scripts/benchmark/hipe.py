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


def _blocks(path: Path) -> Iterator[_Block]:
    """Yield every document in the file with its rebuilt text and entities."""
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        tag, link, misc = (
            header.index(TAG_COLUMN),
            header.index(LINK_COLUMN),
            header.index(MISC_COLUMN),
        )
        block: _Block | None = None
        current: _Entity | None = None
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith(DOCUMENT_ID):
                if block is not None:
                    yield block
                block = _Block(line[len(DOCUMENT_ID) :].strip())
                current = None
                continue
            if not line or line.startswith("#") or block is None:
                continue
            columns = line.split("\t")
            if block.space_pending:
                block.text += " "
            start = len(block.text)
            block.text += columns[0]
            block.space_pending = "NoSpaceAfter" not in columns[misc]

            label = columns[tag].lower()
            if label == f"i-{LOCATION}" and current is not None:
                current.end = len(block.text)
            elif label == f"b-{LOCATION}":
                current = _Entity(start, len(block.text), columns[link])
                block.entities.append(current)
            else:
                current = None
        if block is not None:
            yield block


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
        if gold:
            documents.append(Document(block.identifier, block.text, gold))
        if limit is not None and len(documents) >= limit:
            break
    return documents
