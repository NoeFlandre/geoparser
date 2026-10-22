"""
Tests for the corpus registry.

Every corpus the CLI offers must load to the same shape, so the runner and
the report never need to know which one they are scoring.
"""

import hashlib
import io
from dataclasses import replace

import pytest

from scripts.benchmark import corpora
from scripts.benchmark.corpus import Document, GoldSpan


def pin_release(monkeypatch, release):
    """Pin the release to the miniature zip, so its own size and digest match."""
    data = release.read_bytes()
    monkeypatch.setattr(corpora, "NEWSLI_RELEASE_BYTES", len(data))
    monkeypatch.setattr(
        corpora,
        "NEWSLI_RELEASE_MD5",
        hashlib.md5(data, usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        corpora, "NEWSLI_RELEASE_SHA256", hashlib.sha256(data).hexdigest()
    )


def pin_source(monkeypatch, name, source_bytes):
    spec = corpora.CORPORA[name]
    monkeypatch.setitem(
        corpora.CORPORA,
        name,
        replace(
            spec,
            expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        ),
    )


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

    def test_geovirus_and_hipe_sources_have_full_sha256_pins(self):
        sources = {
            name: spec.expected_source_sha256
            for name, spec in corpora.CORPORA.items()
            if spec.kind in {corpora.GEOVIRUS, corpora.HIPE}
        }

        assert sources == corpora.GEOVIRUS_HIPE_SOURCE_SHA256
        assert all(
            source_sha256 is not None
            and len(source_sha256) == 64
            and set(source_sha256) <= set("0123456789abcdef")
            for source_sha256 in sources.values()
        )

    def test_unknown_corpus_is_an_error(self, tmp_path):
        """Test that a typo does not silently fall back."""
        with pytest.raises(KeyError):
            corpora.load("nope", tmp_path)


class TestLoad:
    """Loading a HIPE corpus through the registry."""

    @pytest.mark.parametrize("kind", [corpora.GEOVIRUS, corpora.HIPE])
    @pytest.mark.parametrize("fetched", [False, True], ids=["cached", "fetched"])
    @pytest.mark.parametrize("matches", [False, True], ids=["changed", "pinned"])
    def test_source_hash_is_checked_before_parsing(
        self, tmp_path, monkeypatch, kind, fetched, matches
    ):
        name = "geovirus" if kind == corpora.GEOVIRUS else "hipe2020-en"
        original_spec = corpora.CORPORA[name]
        filename = original_spec.filename
        source_bytes = b"<source />"
        expected_bytes = source_bytes if matches else b"<different source />"
        expected_hash = hashlib.sha256(expected_bytes).hexdigest()
        spec = replace(
            original_spec,
            documents=None,
            gold_spans=None,
            expected_source_sha256=expected_hash,
        )
        monkeypatch.setitem(corpora.CORPORA, name, spec)
        cache_path = tmp_path / filename
        if fetched:
            monkeypatch.setattr(
                corpora.corpus.urllib.request,
                "urlopen",
                lambda *args, **kwargs: io.BytesIO(source_bytes),
            )
        else:
            cache_path.write_bytes(source_bytes)

        parser_calls = []
        if kind == corpora.GEOVIRUS:
            monkeypatch.setattr(
                corpora.corpus,
                "parse_corpus",
                lambda *args, **kwargs: parser_calls.append("parse") or [],
            )
        else:
            monkeypatch.setattr(
                corpora,
                "hipe_qids",
                lambda path: parser_calls.append("qids") or set(),
            )
            monkeypatch.setattr(
                corpora,
                "load_coordinates",
                lambda *args, **kwargs: parser_calls.append("coordinates") or {},
            )
            monkeypatch.setattr(
                corpora,
                "parse_hipe",
                lambda *args, **kwargs: parser_calls.append("parse") or [],
            )

        if matches:
            loaded = corpora.load(name, tmp_path)
            assert (
                loaded.source_content_sha256 == hashlib.sha256(source_bytes).hexdigest()
            )
            assert parser_calls
        else:
            with pytest.raises(ValueError, match=f"{name}.*SHA-256 mismatch"):
                corpora.load(name, tmp_path)
            assert parser_calls == []

    def test_loads_a_hipe_corpus_from_cache(self, tmp_path, monkeypatch):
        """Test that a cached TSV is parsed with committed coordinates."""
        spec = corpora.CORPORA["hipe2020-fr"]
        source = (
            "TOKEN\tNE-COARSE-LIT\tA\tB\tC\tD\tE\tNEL-LIT\tF\tMISC\n"
            "# hipe2022:document_id = d\n"
            "Paris\tB-loc\tO\tO\tO\tO\tO\tQ90\t_\t_\n"
        )
        source_bytes = source.encode("utf-8")
        (tmp_path / spec.filename).write_bytes(source_bytes)
        pin_source(monkeypatch, spec.name, source_bytes)
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
        source = (
            "TOKEN\tNE-COARSE-LIT\tA\tB\tC\tD\tE\tNEL-LIT\tF\tMISC\n"
            "# hipe2022:document_id = d\n"
            "Paris\tB-loc\tO\tO\tO\tO\tO\tQ90\t_\t_\n"
        )
        source_bytes = source.encode("utf-8")
        (tmp_path / spec.filename).write_bytes(source_bytes)
        pin_source(monkeypatch, spec.name, source_bytes)
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
        pin_release(monkeypatch, release)
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

    @pytest.mark.parametrize("wrong_pin", ["bytes", "digest", "sha256"])
    def test_refuses_a_cached_release_that_is_not_the_pinned_file(
        self, tmp_path, monkeypatch, wrong_pin
    ):
        """A stale or altered zip stops the load before any article is parsed."""
        from tests.unit.test_benchmark.test_newsli import write_release

        release = write_release(tmp_path)
        pin_release(monkeypatch, release)
        if wrong_pin == "bytes":
            monkeypatch.setattr(
                corpora, "NEWSLI_RELEASE_BYTES", corpora.NEWSLI_RELEASE_BYTES + 1
            )
        elif wrong_pin == "digest":
            monkeypatch.setattr(corpora, "NEWSLI_RELEASE_MD5", "0" * 32)
        else:
            monkeypatch.setattr(corpora, "NEWSLI_RELEASE_SHA256", "0" * 64)
        monkeypatch.setattr(
            corpora.corpus, "download_corpus", lambda cache_path, *, url: release
        )

        with pytest.raises(ValueError, match="pinned UniTopRank release"):
            corpora.load("newsli-ro", tmp_path / "newsli-ro")


class TestExpectedTotals:
    """What a load of each corpus must hold, given the caps that apply to it."""

    @pytest.mark.parametrize(
        ("name", "limit", "expected"),
        [
            ("geovirus", None, (229, 2167)),
            ("geovirus", 500, (229, 2167)),
            ("geovirus", 9, (9, None)),
            ("newsli-de", None, (500, None)),
            ("newsli-de", 40, (40, None)),
            ("newsli-fa", None, (71, 326)),
            ("newsli-fa", 10, (10, None)),
            ("newsli-fa", 1000, (71, 326)),
            ("newsli-pl", None, (186, 196)),
            ("hipe2020-fr", None, (43, 800)),
            ("hipe2020-fr", 5, (5, None)),
            ("topres19th-en", None, (110, 880)),
        ],
    )
    def test_a_full_load_holds_the_source_totals_and_a_capped_one_its_cap(
        self, name, limit, expected
    ):
        """Full loads compare with the source; capped loads compare documents only."""
        assert corpora.expected_totals(corpora.CORPORA[name], limit) == expected

    def test_a_load_carries_its_expected_totals(self, tmp_path, monkeypatch):
        """Test that the totals travel with the loaded corpus to the gate."""
        from tests.unit.test_benchmark.test_newsli import write_release

        release = write_release(tmp_path)
        pin_release(monkeypatch, release)
        monkeypatch.setattr(
            corpora.corpus, "download_corpus", lambda cache_path, *, url: release
        )

        loaded = corpora.load("newsli-ro", tmp_path / "newsli-ro")

        assert (loaded.expected_documents, loaded.expected_gold) == (226, 385)
