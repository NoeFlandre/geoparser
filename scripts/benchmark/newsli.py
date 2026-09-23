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
    cap = min(MAX_DOCUMENTS, limit) if limit is not None else MAX_DOCUMENTS
    documents: list[Document] = []
    with zipfile.ZipFile(path) as archive:
        gold = json.loads(archive.read(f"{language}_geotoponyms.json"))
        names = set(archive.namelist())
        for identifier in sorted(gold):
            member = f"{language}_geotoponyms/{identifier}.txt"
            if member not in names:
                continue
            text = archive.read(member).decode("utf-8")
            spans = tuple(_aligned(gold[identifier], text))
            if spans:
                documents.append(Document(identifier, text, spans))
            if len(documents) >= cap:
                break
    return documents


def _aligned(entries: list[dict], text: str):
    """Yield the gold spans that select their own surface form."""
    for entry in entries:
        start, end, name = entry["start"], entry["end"], entry["LOC"]
        if text[start:end] == name:
            yield GoldSpan(start, end, name, float(entry["lat"]), float(entry["lon"]))
