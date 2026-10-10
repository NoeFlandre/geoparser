"""
The corpora the benchmark can score, and one way to load any of them.

Every corpus loads to the same ``LoadedCorpus``, so the runner and the report
never need to know which one they are scoring. GeoVirus carries its own
coordinates; the HIPE-2022 corpora link toponyms to Wikidata, placed through a
committed coordinate cache (see ``wikidata``).

Only the unmasked test splits are registered: the models under test may have
seen training data, and the masked files have their gold removed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from scripts.benchmark import corpus
from scripts.benchmark.corpus import Document
from scripts.benchmark.hipe import hipe_qids, parse_hipe
from scripts.benchmark.newsli import MAX_DOCUMENTS, parse_newsli
from scripts.benchmark.wikidata import load_coordinates

HIPE_BASE_URL = (
    "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1"
)
COORDINATE_CACHE = Path(__file__).with_name("data") / "wikidata-coordinates.json"

GEOVIRUS = "geovirus"
HIPE = "hipe"
NEWSLI = "newsli"
# The UniTopRank data release, which carries NewsLi (Apache-2.0). One file
# serves every language, so it is cached once beside the corpus folders.
NEWSLI_URL = "https://ndownloader.figshare.com/files/59465342"
NEWSLI_RELEASE = "unitoprank-data.zip"
# Text files per NewsLi language in the release, as the inventory snapshot
# records them. Each text file is one article, so a full load is expected to
# hold one document per file. An article that yields no aligned gold span is
# then reported as a source-count problem. The release has not been read for
# this check, so the per-language counts of aligned articles are unverified.
NEWSLI_TEXT_FILES = {
    "ar": 757,
    "de": 13174,
    "es": 8054,
    "fa": 85,
    "ja": 2868,
    "pl": 321,
    "ro": 241,
    "sr": 13950,
    "ta": 873,
    "tr": 805,
    "uk": 282,
}
NEWSLI_LANGUAGES = tuple(NEWSLI_TEXT_FILES)
# GeoVirus holds 229 articles and 2167 gold toponyms. Both figures are recorded
# in this repository, and a parse of the upstream file matched them on 2026-10-10.
GEOVIRUS_DOCUMENTS = 229
GEOVIRUS_GOLD_SPANS = 2167


@dataclass(frozen=True)
class CorpusSpec:
    """Where a corpus comes from, how to read it, and the totals it should hold."""

    name: str
    language: str
    url: str
    kind: str
    documents: int | None = None
    gold_spans: int | None = None

    @property
    def filename(self) -> str:
        """The name the downloaded file is cached under."""
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class LoadedCorpus:
    """A corpus ready to score, with a digest over everything its gold rests on."""

    name: str
    language: str
    documents: list[Document]
    digest: str
    expected_documents: int | None = None
    expected_gold: int | None = None


def _hipe(dataset: str, language: str) -> CorpusSpec:
    """Return the spec of one HIPE-2022 test split."""
    return CorpusSpec(
        name=f"{dataset}-{language}",
        language=language,
        url=(
            f"{HIPE_BASE_URL}/{dataset}/{language}/"
            f"HIPE-2022-v2.1-{dataset}-test-{language}.tsv"
        ),
        kind=HIPE,
    )


CORPORA: dict[str, CorpusSpec] = {
    spec.name: spec
    for spec in (
        CorpusSpec(
            GEOVIRUS,
            "en",
            corpus.CORPUS_URL,
            GEOVIRUS,
            documents=GEOVIRUS_DOCUMENTS,
            gold_spans=GEOVIRUS_GOLD_SPANS,
        ),
        _hipe("hipe2020", "de"),
        _hipe("hipe2020", "fr"),
        _hipe("hipe2020", "en"),
        _hipe("newseye", "de"),
        _hipe("newseye", "fr"),
        _hipe("newseye", "fi"),
        _hipe("newseye", "sv"),
        _hipe("topres19th", "en"),
        *(
            CorpusSpec(
                f"newsli-{language}",
                language,
                NEWSLI_URL,
                NEWSLI,
                documents=NEWSLI_TEXT_FILES[language],
            )
            for language in NEWSLI_LANGUAGES
        ),
    )
}
DEFAULT = (GEOVIRUS,)


def expected_totals(
    spec: CorpusSpec, limit: int | None
) -> tuple[int | None, int | None]:
    """
    Return the document and gold totals a load of one corpus must hold.

    A full load holds what the source publishes. A capped load holds its cap in
    documents: NewsLi keeps MAX_DOCUMENTS articles per language, and ``limit``
    keeps the first N. The gold total of a capped load is not known in advance,
    so it is None and is not compared. A corpus whose source publishes no total
    gives None for both, and the gate reports that it was not compared.

    Args:
        spec: The registered corpus
        limit: The ``--limit`` of the run, if any

    Returns:
        The expected document count and gold span count, each None when not compared
    """
    cap = _effective_cap(spec, limit)
    if spec.documents is None or cap is None or cap >= spec.documents:
        return spec.documents, spec.gold_spans
    return cap, None


def _effective_cap(spec: CorpusSpec, limit: int | None) -> int | None:
    """Return the smaller of the harness cap and the run's limit, if either applies."""
    caps = [cap for cap in (_harness_cap(spec), limit) if cap is not None]
    return min(caps, default=None)


def _harness_cap(spec: CorpusSpec) -> int | None:
    """Return the cap the harness itself applies to a corpus, if it has one."""
    return MAX_DOCUMENTS if spec.kind == NEWSLI else None


def load(
    name: str,
    cache_dir: Path,
    *,
    limit: int | None = None,
    coordinate_cache: Path = COORDINATE_CACHE,
) -> LoadedCorpus:
    """
    Download (once) and parse one registered corpus.

    Args:
        name: A key of CORPORA
        cache_dir: Where downloaded corpus files are kept between runs
        limit: Keep only the first this many documents
        coordinate_cache: The Wikidata coordinate cache for HIPE corpora

    Returns:
        The corpus, parsed, with the totals it is expected to hold

    Raises:
        KeyError: When the name is not registered
    """
    spec = CORPORA[name]
    loaded = _load_parsed(spec, cache_dir, limit, coordinate_cache)
    documents, gold = expected_totals(spec, limit)
    return replace(loaded, expected_documents=documents, expected_gold=gold)


def _load_parsed(
    spec: CorpusSpec,
    cache_dir: Path,
    limit: int | None,
    coordinate_cache: Path,
) -> LoadedCorpus:
    """Download and parse one corpus, before its expected totals are attached."""
    if spec.kind == NEWSLI:
        return _load_newsli(spec, cache_dir, limit)
    path = corpus.download_corpus(cache_dir / spec.filename, url=spec.url)
    if spec.kind == GEOVIRUS:
        return LoadedCorpus(
            spec.name,
            spec.language,
            corpus.parse_corpus(path, limit=limit),
            corpus.corpus_digest(path),
        )
    coordinates = load_coordinates(hipe_qids(path), coordinate_cache)
    digest = hashlib.sha256(path.read_bytes())
    digest.update(json.dumps(sorted(coordinates.items())).encode())
    return LoadedCorpus(
        spec.name,
        spec.language,
        parse_hipe(path, coordinates, limit=limit),
        digest.hexdigest()[:16],
    )


def _load_newsli(spec: CorpusSpec, cache_dir: Path, limit: int | None) -> LoadedCorpus:
    """Load one NewsLi language from the shared release zip."""
    release = corpus.download_corpus(cache_dir.parent / NEWSLI_RELEASE, url=spec.url)
    digest = hashlib.sha256(release.read_bytes())
    digest.update(spec.language.encode())
    return LoadedCorpus(
        spec.name,
        spec.language,
        parse_newsli(release, spec.language, limit=limit),
        digest.hexdigest()[:16],
    )
