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

from scripts.benchmark import corpus, corpus_checks, missing_matrix
from scripts.benchmark.corpus import Document
from scripts.benchmark.hipe import hipe_qids, parse_hipe
from scripts.benchmark.newsli import (
    EXACT_FULL_RECORD_DEDUP_VIEW,
    MAX_DOCUMENTS,
    STRICT_RAW_VIEW,
    deduplicate_exact_full_records,
    read_newsli,
)
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
# The figshare file as published (data.zip, version 10). A cached copy must match
# both values, or the load stops before any article is read.
NEWSLI_RELEASE_BYTES = 63356728
NEWSLI_RELEASE_MD5 = "8e6f82a0931cb909e6f18217e80ce1f3"
NEWSLI_RELEASE_SHA256 = (
    "d4672d28045868cd64c52febedb2f62ae85288b9572e70bb7026b410f355d192"
)
# Totals per NewsLi language, as the checked-in report of the 2026-09-23 run
# recorded them at commit bbb8a7b (benchmark-evidence/2026-09-23-newsli). A
# document is an article the adapter keeps: its text file exists and its gold
# has an aligned span. The raw text-file count is not used, because it also
# counts the articles the adapter drops. Each entry is (documents, gold spans).
# A language at the 500-article cap has no gold total for a full load, since
# its recorded gold belongs to the capped subset, so the gold total is None.
NEWSLI_RECORDED_TOTALS: dict[str, tuple[int, int | None]] = {
    "ar": (500, None),
    "de": (500, None),
    "es": (500, None),
    "fa": (71, 326),
    "ja": (500, None),
    "pl": (186, 196),
    "ro": (226, 385),
    "sr": (500, None),
    "ta": (500, None),
    "tr": (500, None),
    "uk": (164, 258),
}
NEWSLI_LANGUAGES = tuple(NEWSLI_RECORDED_TOTALS)
# GeoVirus holds 229 articles and 2167 gold toponyms. Both figures are recorded
# in this repository, and a parse of the upstream file matched them on 2026-10-10.
GEOVIRUS_DOCUMENTS = 229
GEOVIRUS_GOLD_SPANS = 2167
GEOVIRUS_HIPE_SOURCE_SHA256: dict[str, str] = {
    GEOVIRUS: "2f8a928b9af4a44232493b37c5324bcb14a06873a4d97315cf3b2c7653ea06b1",
    "hipe2020-de": "39a27cae920561005c5a366b923a827bf7e0b35fbbae47943667dcf1865da550",
    "hipe2020-fr": "c3e10f1c9a9b12bae71a40d8148516da3979315a405911474ecfee304bb688a6",
    "hipe2020-en": "081e5b3648c26e7fb4f3825d0ab34fc60c9f021b06943750ad9973cb0458c311",
    "newseye-de": "e208286fc47f364e5d60bf6a077852a5007ca514500f7bbb4bfc645755efe719",
    "newseye-fr": "9afece6528a8dc4da51b00a73fdce0f67a0e14449c05c5e230e4c707da3ecfcc",
    "newseye-fi": "115b3299226b97a1b34db2ee4a544530bced616f084f7e60327c7e375b46682a",
    "newseye-sv": "6fd7634b1aa82432798e1858df2061249ce4a6e81fde9a422346a987460c7faa",
    "topres19th-en": "f54fb9ab5600f9f50126327ac9bef05c6c20ebd6472c1c9b1ffbd0cff465adeb",
}


@dataclass(frozen=True)
class CorpusSpec:
    """Describe a corpus source, its expected hash, and its recorded totals."""

    name: str
    language: str
    url: str
    kind: str
    documents: int | None = None
    gold_spans: int | None = None
    expected_source_sha256: str | None = None

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
    # Gold entries the adapter read and dropped, for sources that drop any.
    dropped_gold: int = 0
    # Capped GeoVirus/HIPE runs also validate the complete recorded source.
    integrity_report: corpus_checks.CorpusReport | None = None
    # The explicit benchmark CLI limit, if this load is a smoke/subset run.
    limit: int | None = None
    scoring_view: str = STRICT_RAW_VIEW
    pre_transform_gold_toponyms: int | None = None
    source_archive_sha256: str | None = None
    exact_duplicate_records: int = 0
    duplicate_records_removed: int = 0
    duplicate_groups_collapsed: int = 0
    source_url: str | None = None
    source_content_sha256: str | None = None
    coordinate_cache_sha256: str | None = None


# Totals per HIPE-2022 test split, as the checked-in multilingual report of the
# 2026-09-23 run recorded them (benchmark-evidence/2026-09-23-multilingual). Each
# entry is (documents, gold spans). Parsing the upstream files on 2026-10-10
# with the current parser and the committed coordinate cache reproduces every
# one of these totals, so a truncated split is refused.
HIPE_RECORDED_TOTALS: dict[str, tuple[int, int]] = {
    "hipe2020-de": (48, 558),
    "hipe2020-fr": (43, 800),
    "hipe2020-en": (38, 158),
    "newseye-de": (8, 628),
    "newseye-fr": (33, 665),
    "newseye-fi": (18, 214),
    "newseye-sv": (18, 265),
    "topres19th-en": (110, 880),
}


def _hipe(dataset: str, language: str) -> CorpusSpec:
    """Return the spec of one HIPE-2022 test split."""
    name = f"{dataset}-{language}"
    documents, gold_spans = HIPE_RECORDED_TOTALS[name]
    return CorpusSpec(
        name=name,
        language=language,
        url=(
            f"{HIPE_BASE_URL}/{dataset}/{language}/"
            f"HIPE-2022-v2.1-{dataset}-test-{language}.tsv"
        ),
        kind=HIPE,
        documents=documents,
        gold_spans=gold_spans,
        expected_source_sha256=GEOVIRUS_HIPE_SOURCE_SHA256[name],
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
            expected_source_sha256=GEOVIRUS_HIPE_SOURCE_SHA256[GEOVIRUS],
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
                documents=NEWSLI_RECORDED_TOTALS[language][0],
                gold_spans=NEWSLI_RECORDED_TOTALS[language][1],
                expected_source_sha256=NEWSLI_RELEASE_SHA256,
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

    A full load holds the totals recorded for the corpus: the source's own for
    GeoVirus, and the adapter-output totals of the checked-in reports for NewsLi
    and HIPE. A capped load holds its cap in documents: NewsLi keeps
    MAX_DOCUMENTS articles per language, and ``limit`` keeps the first N. The
    gold total of a capped load is not known in advance, so it is None and is
    not compared. A corpus with no recorded total gives None for both, and the
    gate reports that it was not compared.

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
    scoring_view: str = STRICT_RAW_VIEW,
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
    _validate_scoring_view(spec, scoring_view)
    documents, gold = expected_totals(spec, limit)
    loaded = _load_expected_slice(
        spec, cache_dir, limit, coordinate_cache, scoring_view, gold
    )
    if scoring_view == EXACT_FULL_RECORD_DEDUP_VIEW:
        documents, gold = _exact_dedup_expected_totals(spec, limit)
    return replace(
        loaded,
        expected_documents=documents,
        expected_gold=gold,
        limit=limit,
        scoring_view=scoring_view,
    )


def _validate_scoring_view(spec: CorpusSpec, scoring_view: str) -> None:
    if scoring_view not in {STRICT_RAW_VIEW, EXACT_FULL_RECORD_DEDUP_VIEW}:
        msg = f"Unknown scoring view: {scoring_view}"
        raise ValueError(msg)
    if scoring_view == EXACT_FULL_RECORD_DEDUP_VIEW and (
        spec.kind != NEWSLI
        or spec.language not in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
    ):
        msg = (
            "NewsLi exact-record dedup is defined only for the seven affected "
            "NewsLi languages."
        )
        raise ValueError(msg)


def _load_expected_slice(
    spec: CorpusSpec,
    cache_dir: Path,
    limit: int | None,
    coordinate_cache: Path,
    scoring_view: str,
    gold: int | None,
) -> LoadedCorpus:
    if limit is not None and gold is None and spec.kind != NEWSLI:
        # A cap makes the subset's gold total unknown, but must not also hide
        # gold lost while parsing the registered source. Validate its pinned
        # full-load totals before selecting the documents to score.
        full = _load_parsed(spec, cache_dir, None, coordinate_cache, scoring_view)
        return replace(
            full,
            documents=full.documents[:limit],
            integrity_report=corpus_checks.check_corpus(
                full.documents,
                expected_documents=spec.documents,
                expected_gold=spec.gold_spans,
                dropped_gold=full.dropped_gold,
            ),
        )
    return _load_parsed(spec, cache_dir, limit, coordinate_cache, scoring_view)


def _load_parsed(
    spec: CorpusSpec,
    cache_dir: Path,
    limit: int | None,
    coordinate_cache: Path,
    scoring_view: str = STRICT_RAW_VIEW,
) -> LoadedCorpus:
    """Download and parse one corpus, before its expected totals are attached."""
    if spec.kind == NEWSLI:
        return _load_newsli(spec, cache_dir, limit, scoring_view)
    path = corpus.download_corpus(cache_dir / spec.filename, url=spec.url)
    source_sha256 = _verify_source_sha256(spec, path)
    if spec.kind == GEOVIRUS:
        return LoadedCorpus(
            spec.name,
            spec.language,
            corpus.parse_corpus(path, limit=limit),
            corpus.corpus_digest(path),
            source_url=spec.url,
            source_content_sha256=source_sha256,
        )
    coordinates = load_coordinates(hipe_qids(path), coordinate_cache)
    digest = hashlib.sha256(path.read_bytes())
    digest.update(json.dumps(sorted(coordinates.items())).encode())
    return LoadedCorpus(
        spec.name,
        spec.language,
        parse_hipe(path, coordinates, limit=limit),
        digest.hexdigest()[:16],
        source_url=spec.url,
        source_content_sha256=source_sha256,
        coordinate_cache_sha256=hashlib.sha256(
            coordinate_cache.read_bytes()
        ).hexdigest(),
    )


def _verify_source_sha256(spec: CorpusSpec, path: Path) -> str:
    """Refuse a cached or downloaded source that differs from its registry pin."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = spec.expected_source_sha256
    if expected is None:
        message = f"{spec.name} has no expected source SHA-256 pin"
        raise ValueError(message)
    if actual != expected:
        message = (
            f"{spec.name} source SHA-256 mismatch: expected {expected}, found {actual}"
        )
        raise ValueError(message)
    return actual


def _check_newsli_release(data: bytes, path: Path) -> None:
    """Refuse a release that is not the pinned file, cached or freshly fetched."""
    if (
        len(data) != NEWSLI_RELEASE_BYTES
        or hashlib.md5(data, usedforsecurity=False).hexdigest() != NEWSLI_RELEASE_MD5
        or hashlib.sha256(data).hexdigest() != NEWSLI_RELEASE_SHA256
    ):
        message = (
            f"{path} is not the pinned UniTopRank release; delete it and load again"
        )
        raise ValueError(message)


def _load_newsli(
    spec: CorpusSpec,
    cache_dir: Path,
    limit: int | None,
    scoring_view: str = STRICT_RAW_VIEW,
) -> LoadedCorpus:
    """Load one NewsLi language from the shared release zip."""
    release = corpus.download_corpus(cache_dir.parent / NEWSLI_RELEASE, url=spec.url)
    data = release.read_bytes()
    _check_newsli_release(data, release)
    digest = hashlib.sha256(data)
    source_archive_sha256 = digest.hexdigest()
    digest.update(spec.language.encode())
    documents, dropped = read_newsli(release, spec.language, limit=limit)
    pre_transform_gold = sum(len(document.gold) for document in documents)
    deduplicated, exact_duplicates, collapsed_groups = deduplicate_exact_full_records(
        documents
    )
    removed_records = 0
    if scoring_view == EXACT_FULL_RECORD_DEDUP_VIEW:
        documents = deduplicated
        removed_records = exact_duplicates
        fixed = _newsli_matrix_cell(spec.name)
        if len(documents) == fixed.documents and (
            removed_records
            != missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[spec.language]
        ):
            msg = (
                f"{spec.name} removed {removed_records} exact duplicate records; "
                "the pinned protocol audit expects "
                f"{missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[spec.language]}."
            )
            raise ValueError(msg)
    digest.update(scoring_view.encode())
    return LoadedCorpus(
        spec.name,
        spec.language,
        documents,
        digest.hexdigest()[:16],
        dropped_gold=dropped,
        scoring_view=scoring_view,
        pre_transform_gold_toponyms=pre_transform_gold,
        source_archive_sha256=source_archive_sha256,
        exact_duplicate_records=exact_duplicates,
        duplicate_records_removed=removed_records,
        duplicate_groups_collapsed=collapsed_groups,
        source_url=spec.url,
        source_content_sha256=source_archive_sha256,
    )


def _newsli_matrix_cell(name: str) -> missing_matrix.CorpusLanguageCell:
    """Return the frozen NewsLi subset used by the published dedup audit."""
    for cell in missing_matrix.MATRIX:
        if missing_matrix.cell_name(cell) == name:
            return cell
    msg = f"NewsLi corpus is outside the frozen matrix: {name}"
    raise KeyError(msg)


def _exact_dedup_expected_totals(
    spec: CorpusSpec, limit: int | None
) -> tuple[int | None, int | None]:
    """Return the reviewed transformed totals for one NewsLi dedup slice."""
    fixed = _newsli_matrix_cell(spec.name)
    if limit is not None and limit < fixed.documents:
        return limit, None
    return (
        fixed.documents,
        fixed.gold_toponyms
        - missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[spec.language],
    )
