"""
Read NewsLi from the UniTopRank data release.

NewsLi (Hu et al., *UniTopRank*, IJGIS 2026) is Wikinews in eleven languages
-- Arabic, German, Spanish, Persian, Japanese, Polish, Romanian, Serbian,
Tamil, Turkish and Ukrainian -- derived from the multilingual entity-linking
data of the Mewsli family, keeping the entities whose Wikidata item has a
GeoNames ID, with that place's coordinates. The release is one zip: per
language, a JSON of gold spans keyed by article, and one text file per
article. It is read in place, so a node never unpacks tens of thousands of
small files onto a quota-limited home.

German, Serbian and Spanish hold 7k to 14k articles each, hours of work per
pipeline, so every language is capped at MAX_DOCUMENTS articles, taken in
identifier order so the selection is deterministic.

Like the other readers, this module holds no models, torch or gazetteer.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.benchmark.corpus import Document, GoldSpan

MAX_DOCUMENTS = 500
STRICT_RAW_VIEW = "strict-raw-v1"
EXACT_FULL_RECORD_DEDUP_VIEW = "newsli-exact-full-record-dedup-v1"


def _article(
    archive: zipfile.ZipFile,
    names: set[str],
    language: str,
    identifier: str,
    entries: list[dict],
) -> tuple[Document | None, int]:
    """
    Build one article when its text file exists and has valid gold spans.

    Returns the article, or None, with the number of gold entries it dropped.
    """
    member = f"{language}_geotoponyms/{identifier}.txt"
    if member not in names:
        return None, len(entries)
    text = archive.read(member).decode("utf-8")
    spans = tuple(_aligned(entries, text))
    dropped = len(entries) - len(spans)
    if not spans:
        return None, dropped
    return Document(identifier, text, spans), dropped


def _articles(
    archive: zipfile.ZipFile,
    language: str,
    gold: dict[str, list[dict]],
    limit: int,
) -> tuple[list[Document], int]:
    """Read articles in stable identifier order, up to the requested cap."""
    names = set(archive.namelist())
    documents = []
    dropped = 0
    for identifier in sorted(gold):
        document, lost = _article(
            archive, names, language, identifier, gold[identifier]
        )
        dropped += lost
        if document is not None:
            documents.append(document)
        if len(documents) >= limit:
            break
    return documents, dropped


def read_newsli(
    path: Path, language: str, *, limit: int | None = None
) -> tuple[list[Document], int]:
    """
    Read one language of the NewsLi release, and count the gold it dropped.

    A gold entry is dropped when its span does not select its own surface form,
    or when its article has no text file. The count covers only the articles
    read under the cap, so a capped load reports the drops it actually scored
    past.

    Args:
        path: The release zip
        language: Two-letter code of the language to read
        limit: Keep only the first this many documents, on top of the cap

    Returns:
        The parsed documents, in identifier order, and the dropped gold entries

    Raises:
        KeyError: When the release holds no such language
    """
    cap = min(MAX_DOCUMENTS, limit) if limit is not None else MAX_DOCUMENTS
    with zipfile.ZipFile(path) as archive:
        gold = json.loads(archive.read(f"{language}_geotoponyms.json"))
        return _articles(archive, language, gold, cap)


def parse_newsli(
    path: Path, language: str, *, limit: int | None = None
) -> list[Document]:
    """
    Read one language of the NewsLi release into documents with gold spans.

    Args:
        path: The release zip
        language: Two-letter code of the language to read
        limit: Keep only the first this many documents, on top of the cap

    Returns:
        The parsed documents, in identifier order

    Raises:
        KeyError: When the release holds no such language
    """
    return read_newsli(path, language, limit=limit)[0]


def deduplicate_exact_full_records(
    documents: list[Document],
) -> tuple[list[Document], int, int]:
    """Collapse repeated full gold records without changing source documents.

    Equality follows the published NewsLi view: start, end, name, latitude,
    longitude. The first occurrence stays in source order. Nonidentical records
    remain in the output.

    Returns the new document list, excess records removed, and groups collapsed.
    """
    transformed: list[Document] = []
    removed_records = 0
    collapsed_groups = 0
    for document in documents:
        seen: set[tuple[int, int, str, float, float]] = set()
        duplicate_groups: set[tuple[int, int, str, float, float]] = set()
        retained: list[GoldSpan] = []
        for span in document.gold:
            key = (span.start, span.end, span.name, span.latitude, span.longitude)
            if key in seen:
                removed_records += 1
                if key not in duplicate_groups:
                    duplicate_groups.add(key)
                    collapsed_groups += 1
                continue
            seen.add(key)
            retained.append(span)
        transformed.append(
            Document(document.identifier, document.text, tuple(retained))
        )
    return transformed, removed_records, collapsed_groups


def _aligned(entries: list[dict], text: str):
    """Yield the gold spans that select their own surface form."""
    for entry in entries:
        start, end, name = entry["start"], entry["end"], entry["LOC"]
        if text[start:end] == name:
            yield GoldSpan(start, end, name, float(entry["lat"]), float(entry["lon"]))
