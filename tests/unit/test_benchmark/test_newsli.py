"""
Tests for reading NewsLi from the UniTopRank data release.

NewsLi is Wikinews in eleven languages with toponyms linked to GeoNames
coordinates (Hu et al., UniTopRank, IJGIS 2026). The release is one zip
holding, per language, a JSON of gold spans and one text file per article;
it is read in place, so the node never unpacks tens of thousands of files.
"""

import json
import zipfile

import pytest

from scripts.benchmark.newsli import MAX_DOCUMENTS, parse_newsli


def write_release(tmp_path, language="ro", documents=None):
    """Write a miniature release zip and return its path."""
    documents = documents or {
        "ro-2": ("Merg la Craiova.", [(8, 15, "Craiova", 44.33, 23.81)]),
        "ro-1": ("Din Egipt vin.", [(4, 9, "Egipt", 27.0, 29.0)]),
    }
    path = tmp_path / "data.zip"
    gold = {
        identifier: [
            {"start": s, "end": e, "LOC": name, "lat": lat, "lon": lon}
            for s, e, name, lat, lon in spans
        ]
        for identifier, (_, spans) in documents.items()
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{language}_geotoponyms.json", json.dumps(gold))
        for identifier, (text, _) in documents.items():
            if text is not None:
                archive.writestr(f"{language}_geotoponyms/{identifier}.txt", text)
    return path


class TestParseNewsli:
    """Turning the release into documents with gold spans."""

    def test_reads_text_and_spans(self, tmp_path):
        """Each article becomes a document whose spans select their names."""
        documents = parse_newsli(write_release(tmp_path), "ro")

        for document in documents:
            for span in document.gold:
                assert document.text[span.start : span.end] == span.name

    def test_orders_documents_by_identifier(self, tmp_path):
        """Order is deterministic, independent of the JSON's key order."""
        documents = parse_newsli(write_release(tmp_path), "ro")

        assert [d.identifier for d in documents] == ["ro-1", "ro-2"]

    def test_keeps_coordinates(self, tmp_path):
        """The gold place is where the release says it is."""
        (first, _) = parse_newsli(write_release(tmp_path), "ro")

        assert [(s.name, s.latitude, s.longitude) for s in first.gold] == [
            ("Egipt", 27.0, 29.0)
        ]

    def test_drops_a_span_that_does_not_match_its_text(self, tmp_path):
        """A misaligned span could only count as a miss for everyone."""
        path = write_release(
            tmp_path,
            documents={"ro-1": ("Din Egipt vin.", [(0, 5, "Egipt", 27.0, 29.0)])},
        )

        assert parse_newsli(path, "ro") == []

    def test_skips_an_article_whose_text_is_missing(self, tmp_path):
        """A gold entry without its text file is not a document."""
        path = write_release(
            tmp_path,
            documents={
                "ro-1": ("Din Egipt vin.", [(4, 9, "Egipt", 27.0, 29.0)]),
                "ro-9": (None, [(0, 1, "x", 0.0, 0.0)]),
            },
        )

        assert [d.identifier for d in parse_newsli(path, "ro")] == ["ro-1"]

    def test_caps_a_large_language(self, tmp_path):
        """At most MAX_DOCUMENTS articles, the first by identifier."""
        documents = {
            f"ro-{i:05d}": ("Egipt.", [(0, 5, "Egipt", 27.0, 29.0)])
            for i in range(MAX_DOCUMENTS + 5)
        }

        parsed = parse_newsli(write_release(tmp_path, documents=documents), "ro")

        assert len(parsed) == MAX_DOCUMENTS
        assert parsed[-1].identifier == f"ro-{MAX_DOCUMENTS - 1:05d}"

    def test_limit_shortens_further(self, tmp_path):
        """--limit applies on top of the cap."""
        assert len(parse_newsli(write_release(tmp_path), "ro", limit=1)) == 1

    def test_an_unknown_language_is_an_error(self, tmp_path):
        """A language the release does not hold fails loudly."""
        with pytest.raises(KeyError):
            parse_newsli(write_release(tmp_path), "xx")
