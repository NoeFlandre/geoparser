"""
Tests for the corpus registry.

Every corpus the CLI offers must load to the same shape, so the runner and
the report never need to know which one they are scoring.
"""

import pytest

from scripts.benchmark import corpora
from scripts.benchmark.corpus import Document, GoldSpan


class TestRegistry:
    """Which corpora exist and what they declare."""

    def test_geovirus_is_the_default(self):
        """Test that the historical benchmark stays the default."""
        assert corpora.DEFAULT == ("geovirus",)
        assert "geovirus" in corpora.CORPORA

    def test_offers_multilingual_hipe_corpora(self):
        """Test that non-English corpora are registered."""
        languages = {spec.language for spec in corpora.CORPORA.values()}

        assert {"de", "fr", "en"} <= languages
        assert len(languages) >= 4

    def test_hipe_urls_point_at_test_splits(self):
        """Test that no training data is benchmarked on."""
        for name, spec in corpora.CORPORA.items():
            if name.startswith(("hipe2020", "newseye", "topres")):
                assert "-test-" in spec.url, name
                assert "masked" not in spec.url, name

    def test_unknown_corpus_is_an_error(self, tmp_path):
        """Test that a typo does not silently fall back."""
        with pytest.raises(KeyError):
            corpora.load("nope", tmp_path)


class TestLoad:
    """Loading a HIPE corpus through the registry."""

    def test_loads_a_hipe_corpus_from_cache(self, tmp_path, monkeypatch):
        """Test that a cached TSV is parsed with committed coordinates."""
        spec = corpora.CORPORA["hipe2020-fr"]
        (tmp_path / spec.filename).write_text(
            "TOKEN\tNE-COARSE-LIT\tA\tB\tC\tD\tE\tNEL-LIT\tF\tMISC\n"
            "# hipe2022:document_id = d\n"
            "Paris\tB-loc\tO\tO\tO\tO\tO\tQ90\t_\t_\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            corpora, "load_coordinates", lambda qids, cache, **_: {"Q90": (48.85, 2.35)}
        )

        loaded = corpora.load("hipe2020-fr", tmp_path)

        assert loaded.name == "hipe2020-fr"
        assert loaded.language == "fr"
        assert loaded.documents == [
            Document("d", "Paris", (GoldSpan(0, 5, "Paris", 48.85, 2.35),))
        ]
        assert len(loaded.digest) == 16

    def test_digest_changes_with_coordinates(self, tmp_path, monkeypatch):
        """Test that a checkpoint cannot resume across a gold change."""
        spec = corpora.CORPORA["hipe2020-fr"]
        (tmp_path / spec.filename).write_text(
            "TOKEN\tNE-COARSE-LIT\tA\tB\tC\tD\tE\tNEL-LIT\tF\tMISC\n"
            "# hipe2022:document_id = d\n"
            "Paris\tB-loc\tO\tO\tO\tO\tO\tQ90\t_\t_\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            corpora, "load_coordinates", lambda qids, cache, **_: {"Q90": (48.85, 2.35)}
        )
        first = corpora.load("hipe2020-fr", tmp_path).digest
        monkeypatch.setattr(
            corpora, "load_coordinates", lambda qids, cache, **_: {"Q90": (1.0, 2.0)}
        )

        assert corpora.load("hipe2020-fr", tmp_path).digest != first


class TestNewsliRegistry:
    """NewsLi's eleven languages are registered."""

    def test_registers_every_newsli_language(self):
        """ar, de, es, fa, ja, pl, ro, sr, ta, tr and uk."""
        newsli = {
            spec.language
            for name, spec in corpora.CORPORA.items()
            if name.startswith("newsli-")
        }

        assert newsli == {
            "ar",
            "de",
            "es",
            "fa",
            "ja",
            "pl",
            "ro",
            "sr",
            "ta",
            "tr",
            "uk",
        }

    def test_loads_newsli_from_one_shared_release(self, tmp_path, monkeypatch):
        """Every language reads the same zip, cached beside the corpus folders."""
        from tests.unit.test_benchmark.test_newsli import write_release

        release = write_release(tmp_path)
        fetched = []

        def download(cache_path, *, url):
            fetched.append(cache_path)
            return release

        monkeypatch.setattr(corpora.corpus, "download_corpus", download)

        loaded = corpora.load("newsli-ro", tmp_path / "newsli-ro")

        assert fetched == [tmp_path / corpora.NEWSLI_RELEASE]
        assert loaded.language == "ro"
        assert [d.identifier for d in loaded.documents] == ["ro-1", "ro-2"]
        assert len(loaded.digest) == 16
