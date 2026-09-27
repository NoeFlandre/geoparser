"""
Read the GeoVirus benchmark corpus.

GeoVirus (Gritta et al., *A Pragmatic Guide to Geoparsing Evaluation*) is 229
WikiNews articles whose gold toponyms carry coordinates and a Wikipedia page
but no gazetteer identifiers. That is why the harness around this module scores
distance rather than identifier equality: coordinates are the only thing two
different gazetteers can be compared on.

This module is deliberately free of models, torch and the gazetteer, so the
corpus can be parsed and the parsing tested without any of them.
"""

from __future__ import annotations

import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

CORPUS_URL = (
    "https://raw.githubusercontent.com/milangritta/"
    "Pragmatic-Guide-to-Geoparsing-Evaluation/master/data/Corpora/GeoVirus.xml"
)
CORPUS_NAME = "GeoVirus"

# GeoVirus records every offset one character further along than the text it
# ships: ``start`` 170 for "Pandi" indexes "andi,". Measured before being
# compensated for -- all 2167 gold spans in all 229 articles align at -1, and
# no other shift in [-3, 3] aligns a single one.
OFFSET_SHIFT = -1


@dataclass(frozen=True)
class GoldSpan:
    """One gold toponym: where it is written, and where on Earth it is."""

    start: int
    end: int
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Document:
    """One corpus article and its gold toponyms."""

    identifier: str
    text: str
    gold: tuple[GoldSpan, ...]


def download_corpus(cache_path: Path, *, url: str = CORPUS_URL) -> Path:
    """
    Return the corpus file, fetching it once into the cache.

    The corpus is fetched to a temporary name and moved into place, so an
    interrupted download cannot leave a truncated file that later runs would
    silently parse as a smaller corpus.

    Args:
        cache_path: Where the downloaded XML is kept between runs
        url: Where to fetch the corpus from if it is not cached

    Returns:
        The path to the corpus file on disk
    """
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL of the benchmark corpus
        partial.write_bytes(response.read())
    partial.replace(cache_path)
    return cache_path


def corpus_digest(path: Path) -> str:
    """
    Return a short content digest of the corpus file.

    A checkpoint is only safe to resume from if the corpus has not changed
    underneath it, and the file name alone cannot say that.

    Args:
        path: The corpus XML file

    Returns:
        The first 16 characters of the file's SHA-256 digest
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def parse_corpus(path: Path, *, limit: int | None = None) -> list[Document]:
    """
    Read GeoVirus into documents with gold spans.

    A span that does not match the text it points at after the shift is
    dropped, because it could only ever count as a miss and would depress
    every pipeline alike.

    Args:
        path: The corpus XML file
        limit: Keep only the first this many articles, for a shorter run

    Returns:
        The parsed documents, in corpus order
    """
    root = ET.parse(path).getroot()  # noqa: S314 - parses the benchmark corpus this script downloads itself
    documents: list[Document] = []
    for index, article in enumerate(root.findall("article")):
        text = article.findtext("text") or ""
        if not text:
            continue
        gold = tuple(_gold_spans(article, text))
        if gold:
            documents.append(Document(str(index), text, gold))
        if limit is not None and len(documents) >= limit:
            break
    return documents


def _gold_spans(article: ET.Element, text: str):
    """Yield the gold spans of one article that line up with its text."""
    for location in article.findall("./locations/location"):
        name = (location.findtext("name") or "").strip()
        latitude = location.findtext("lat")
        longitude = location.findtext("lon")
        if latitude is None or longitude is None:
            continue
        try:
            start = int(location.findtext("start") or "") + OFFSET_SHIFT
            end = int(location.findtext("end") or "") + OFFSET_SHIFT
        except ValueError:
            continue
        if start < 0 or text[start:end] != name:
            continue
        yield GoldSpan(start, end, name, float(latitude), float(longitude))


def gold_toponym_count(documents: list[Document]) -> int:
    """Return how many gold toponyms the documents hold in total."""
    return sum(len(document.gold) for document in documents)
