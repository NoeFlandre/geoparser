"""The NewsLi transform has its own identity and never edits strict raw."""

import hashlib

import pytest

from scripts.benchmark import checkpoint, corpora, runner
from scripts.benchmark.corpus import Document, GoldSpan
from scripts.benchmark.newsli import (
    EXACT_FULL_RECORD_DEDUP_VIEW,
    STRICT_RAW_VIEW,
    deduplicate_exact_full_records,
)


def _pin_release(monkeypatch, release):
    """Pin the miniature zip to the reader's strict source metadata."""
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


def test_exact_full_record_dedup_keeps_first_occurrence_and_other_records():
    duplicate = GoldSpan(0, 5, "Paris", 48.8, 2.3)
    different_coordinates = GoldSpan(0, 5, "Paris", 1.0, 2.3)
    different_name = GoldSpan(0, 5, "PARIS", 48.8, 2.3)
    source = [
        Document(
            "ar-1",
            "Paris",
            (duplicate, duplicate, different_coordinates, different_name),
        )
    ]

    transformed, removed, groups = deduplicate_exact_full_records(source)

    assert source[0].gold == (
        duplicate,
        duplicate,
        different_coordinates,
        different_name,
    )
    assert transformed[0].gold == (duplicate, different_coordinates, different_name)
    assert (removed, groups) == (1, 1)


def test_dedup_view_is_restricted_to_the_seven_audited_newsli_languages(tmp_path):
    with pytest.raises(ValueError, match="seven affected NewsLi languages"):
        corpora.load(
            "newsli-ro",
            tmp_path,
            scoring_view=EXACT_FULL_RECORD_DEDUP_VIEW,
        )
    with pytest.raises(ValueError, match="seven affected NewsLi languages"):
        corpora.load(
            "geovirus",
            tmp_path,
            scoring_view=EXACT_FULL_RECORD_DEDUP_VIEW,
        )


def test_raw_and_dedup_have_distinct_digest_and_keep_transform_counts(
    tmp_path, monkeypatch
):
    from tests.unit.test_benchmark.test_newsli import write_release

    duplicate = (4, 9, "Egipt", 27.0, 29.0)
    release = write_release(
        tmp_path,
        language="ar",
        documents={"ar-1": ("Din Egipt vin.", [duplicate, duplicate])},
    )
    _pin_release(monkeypatch, release)
    monkeypatch.setattr(
        corpora.corpus, "download_corpus", lambda _cache_path, *, url: release
    )

    raw = corpora.load("newsli-ar", tmp_path / "raw", scoring_view=STRICT_RAW_VIEW)
    dedup = corpora.load(
        "newsli-ar",
        tmp_path / "dedup",
        scoring_view=EXACT_FULL_RECORD_DEDUP_VIEW,
    )

    assert (
        len(raw.documents[0].gold),
        len(dedup.documents[0].gold),
        raw.digest == dedup.digest,
        raw.exact_duplicate_records,
        raw.duplicate_records_removed,
        dedup.duplicate_records_removed,
        dedup.pre_transform_gold_toponyms,
    ) == (2, 1, False, 1, 0, 1, 2)


def test_raw_protocol_retains_duplicates_and_records_release_hash(
    tmp_path, monkeypatch
):
    from tests.unit.test_benchmark.test_newsli import write_release

    duplicate = (4, 9, "Egipt", 27.0, 29.0)
    release = write_release(
        tmp_path,
        language="ar",
        documents={"ar-1": ("Din Egipt vin.", [duplicate, duplicate])},
    )
    _pin_release(monkeypatch, release)
    monkeypatch.setattr(
        corpora.corpus, "download_corpus", lambda _cache_path, *, url: release
    )

    loaded = corpora.load("newsli-ar", tmp_path / "raw")

    assert (
        loaded.source_archive_sha256,
        loaded.exact_duplicate_records,
        loaded.duplicate_records_removed,
    ) == (hashlib.sha256(release.read_bytes()).hexdigest(), 1, 0)


def test_duplicate_reference_rows_remain_raw_but_metrics_score_span_identity_once():
    duplicate = GoldSpan(4, 9, "Egipt", 27.0, 29.0)
    raw_documents = [Document("ar-1", "Din Egipt vin.", (duplicate, duplicate))]
    deduplicated, removed, _groups = deduplicate_exact_full_records(raw_documents)

    def score(documents):
        state = checkpoint.Checkpoint(
            checkpoint.RunIdentity(
                pipeline="gliner-minilm",
                corpus_digest="corpus-pin",
                gazetteer="geonames",
                min_similarity=0.0,
                limit=None,
                commit="code-pin",
            )
        )
        state.record(
            runner.RECOGNITION,
            "ar-1",
            [
                {
                    "start": 4,
                    "end": 9,
                    "identifier": None,
                    "document_id": "ar-1",
                    "latitude": None,
                    "longitude": None,
                }
            ],
        )
        state.record(
            runner.RESOLUTION,
            "ar-1",
            [
                {
                    "start": 4,
                    "end": 9,
                    "identifier": "geonames:123",
                    "document_id": "ar-1",
                    "latitude": 27.0,
                    "longitude": 29.0,
                }
            ],
        )
        return runner.score("gliner-minilm", documents, state, models={}, device="cpu")

    raw_result = score(raw_documents)
    dedup_result = score(deduplicated)

    assert (
        len(raw_documents[0].gold),
        removed,
        len(deduplicated[0].gold),
        runner.scoring_span_count(raw_documents),
        runner.scoring_span_count(deduplicated),
        raw_result.recognition == dedup_result.recognition,
        raw_result.resolution == dedup_result.resolution,
    ) == (
        2,
        1,
        1,
        1,
        1,
        True,
        True,
    )
