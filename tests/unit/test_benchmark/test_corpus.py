"""
Tests for reading the GeoVirus corpus.

The offset correction is the part worth pinning: uncorrected it costs every
pipeline nearly all of its recall, and it does so silently, producing a
confident benchmark of the wrong thing.
"""

import pytest

from scripts.benchmark.corpus import (
    OFFSET_SHIFT,
    corpus_digest,
    download_corpus,
    gold_toponym_count,
    parse_corpus,
)

ARTICLE = """<articles>
  <article>
    <source>https://example.invalid/a</source>
    <text>Cases were reported in Pandi and in Bulacan today.</text>
    <locations>
      <location>
        <name>Pandi</name><start>24</start><end>29</end>
        <lat>14.87</lat><lon>120.95</lon>
      </location>
      <location>
        <name>Bulacan</name><start>37</start><end>44</end>
        <lat>15.0</lat><lon>121.08</lon>
      </location>
    </locations>
  </article>
</articles>
"""


def write(tmp_path, xml, name="corpus.xml"):
    """Write a corpus file and return its path."""
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return path


class TestParseCorpus:
    """Turning the shipped XML into documents with usable spans."""

    def test_applies_the_one_character_offset_shift(self, tmp_path):
        """Test that the corrected offsets select the toponym text."""
        documents = parse_corpus(write(tmp_path, ARTICLE))

        document = documents[0]
        assert OFFSET_SHIFT == -1
        for span in document.gold:
            assert document.text[span.start : span.end] == span.name

    def test_keeps_coordinates_and_names(self, tmp_path):
        """Test that the gold data survives parsing."""
        (document,) = parse_corpus(write(tmp_path, ARTICLE))

        assert [(s.name, s.latitude, s.longitude) for s in document.gold] == [
            ("Pandi", 14.87, 120.95),
            ("Bulacan", 15.0, 121.08),
        ]

    def test_drops_a_span_whose_offsets_do_not_match(self, tmp_path):
        """Test that a misaligned span is dropped rather than scored as a miss."""
        xml = ARTICLE.replace(
            "<start>24</start><end>29</end>", "<start>2</start><end>7</end>"
        )

        (document,) = parse_corpus(write(tmp_path, xml))

        assert [span.name for span in document.gold] == ["Bulacan"]

    def test_drops_a_span_without_coordinates(self, tmp_path):
        """Test that a span with no location cannot be scored."""
        xml = ARTICLE.replace("<lat>14.87</lat><lon>120.95</lon>", "")

        (document,) = parse_corpus(write(tmp_path, xml))

        assert [span.name for span in document.gold] == ["Bulacan"]

    def test_drops_a_span_with_unparsable_offsets(self, tmp_path):
        """Test that a non-numeric offset does not crash the parse."""
        xml = ARTICLE.replace("<start>24</start>", "<start>x</start>")

        (document,) = parse_corpus(write(tmp_path, xml))

        assert [span.name for span in document.gold] == ["Bulacan"]

    def test_drops_a_span_shifted_before_the_text(self, tmp_path):
        """Test that offset zero cannot become a negative index."""
        xml = ARTICLE.replace(
            "<start>24</start><end>29</end>", "<start>0</start><end>5</end>"
        )

        (document,) = parse_corpus(write(tmp_path, xml))

        assert [span.name for span in document.gold] == ["Bulacan"]

    def test_skips_an_article_without_text(self, tmp_path):
        """Test that an empty article is not a document."""
        xml = ARTICLE.replace(
            "<text>Cases were reported in Pandi and in Bulacan today.</text>",
            "<text></text>",
        )

        assert parse_corpus(write(tmp_path, xml)) == []

    def test_skips_an_article_without_usable_gold(self, tmp_path):
        """Test that an article contributes nothing when no span survives."""
        xml = ARTICLE.replace("<locations>", "<locations2>").replace(
            "</locations>", "</locations2>"
        )

        assert parse_corpus(write(tmp_path, xml)) == []

    def test_limit_stops_after_the_requested_articles(self, tmp_path):
        """Test that a limit shortens the run."""
        xml = ARTICLE.replace("</articles>", ARTICLE.split("<articles>")[1])

        assert len(parse_corpus(write(tmp_path, xml), limit=1)) == 1
        assert len(parse_corpus(write(tmp_path, xml))) == 2

    def test_counts_gold_toponyms(self, tmp_path):
        """Test the total used in the report header."""
        assert gold_toponym_count(parse_corpus(write(tmp_path, ARTICLE))) == 2


class TestCorpusFile:
    """Fetching and identifying the corpus file."""

    def test_digest_changes_with_content(self, tmp_path):
        """Test that the digest distinguishes two corpora."""
        one = write(tmp_path, ARTICLE, "one.xml")
        other = write(tmp_path, ARTICLE.replace("Pandi", "Pandj"), "other.xml")

        assert corpus_digest(one) != corpus_digest(other)

    def test_digest_is_stable(self, tmp_path):
        """Test that the same bytes give the same digest."""
        path = write(tmp_path, ARTICLE)

        assert corpus_digest(path) == corpus_digest(path)

    def test_download_keeps_an_existing_file(self, tmp_path):
        """Test that a cached corpus is not fetched again."""
        path = write(tmp_path, ARTICLE)

        # An unreachable URL: reaching the network at all would fail the test.
        assert download_corpus(path, url="https://example.invalid") == path

    def test_download_writes_through_a_partial_file(self, tmp_path, monkeypatch):
        """Test that an interrupted fetch cannot leave a truncated corpus."""
        import scripts.benchmark.corpus as module

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                raise OSError("connection reset")

        monkeypatch.setattr(
            module.urllib.request, "urlopen", lambda *a, **k: Response()
        )
        target = tmp_path / "sub" / "corpus.xml"

        with pytest.raises(OSError, match="connection reset"):
            download_corpus(target)

        assert not target.exists()

    def test_download_fetches_when_missing(self, tmp_path, monkeypatch):
        """Test the successful fetch path."""
        import scripts.benchmark.corpus as module

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return ARTICLE.encode()

        monkeypatch.setattr(
            module.urllib.request, "urlopen", lambda *a, **k: Response()
        )
        target = tmp_path / "sub" / "corpus.xml"

        assert download_corpus(target) == target
        assert gold_toponym_count(parse_corpus(target)) == 2
