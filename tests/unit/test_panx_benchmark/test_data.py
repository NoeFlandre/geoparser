import sys
from types import SimpleNamespace

import pytest

from scripts.panx_benchmark import data
from scripts.panx_benchmark.constants import DATASET_ID, DATASET_REVISION, MODELS


def test_pinned_language_and_wikiann_intersection_is_explicit():
    languages = data.target_languages()
    manifest = data.split_manifest()

    assert len(languages) == 85
    assert len(set(languages)) == 85
    assert len(manifest["eligible_languages"]) == 82
    assert set(manifest["missing_target_languages"]) == {"ha", "xh", "zu"}
    assert set(manifest["eligible_languages"]) | set(
        manifest["missing_target_languages"]
    ) == set(languages)
    assert manifest["total_test_examples"] == 423_100


def test_location_bio_tags_align_to_joined_unicode_text():
    example = data.example_from_row(
        "pt",
        {
            "tokens": ["São", "Paulo", "and", "Rio"],
            "ner_tags": [5, 6, 0, 5],
            "langs": ["pt"] * 4,
        },
    )

    assert example.text == "São Paulo and Rio"
    assert example.gold_spans == {(0, 9), (14, 17)}
    assert example.malformed_location_tags == 0


def test_orphan_iob2_location_tag_is_kept_and_reported():
    example = data.example_from_row(
        "en",
        {"tokens": ["New", "York"], "ner_tags": [6, 6], "langs": ["en", "en"]},
    )

    assert example.gold_spans == {(0, 8)}
    assert example.malformed_location_tags == 1


def test_example_rejects_misaligned_tokens_and_tags():
    with pytest.raises(ValueError, match="counts do not match"):
        data.example_from_row(
            "en", {"tokens": ["Paris"], "ner_tags": [], "langs": ["en"]}
        )


def test_example_rejects_a_row_with_another_language():
    with pytest.raises(ValueError, match="different language"):
        data.example_from_row(
            "en", {"tokens": ["Paris"], "ner_tags": [5], "langs": ["fr"]}
        )


def test_example_rejects_unknown_tag_ids():
    with pytest.raises(ValueError, match="Unknown WikiANN tag"):
        data.example_from_row(
            "en", {"tokens": ["Paris"], "ner_tags": [99], "langs": ["en"]}
        )


def test_loader_checks_revision_counts_and_limits_rows(monkeypatch, tmp_path):
    manifest = data.split_manifest()
    counts = manifest["test_examples_by_language"]
    calls = []

    class Split:
        def __init__(self, language):
            self.language = language

        def __len__(self):
            return counts[self.language]

        def __getitem__(self, index):
            assert index == 0
            return {
                "tokens": [self.language],
                "ner_tags": [5],
                "langs": [self.language],
            }

    def fake_load_dataset(dataset_id, *, name, split, revision, cache_dir):
        calls.append((dataset_id, name, split, revision, cache_dir))
        return Split(name)

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )

    loaded = data.load_test_examples(tmp_path, limit_per_language=1)

    assert loaded.evaluated_example_count == 82
    assert loaded.source_example_count == 423_100
    assert loaded.limit_per_language == 1
    assert len(calls) == 82
    assert all(
        call[0] == DATASET_ID and call[2] == "test" and call[3] == DATASET_REVISION
        for call in calls
    )


def test_loader_fails_closed_if_pinned_split_count_changes(monkeypatch, tmp_path):
    class ChangedSplit:
        def __len__(self):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: ChangedSplit()),
    )

    with pytest.raises(ValueError, match="snapshot records"):
        data.load_test_examples(tmp_path, limit_per_language=1)


def test_target_language_manifest_carries_upstream_commit():
    manifest = data.read_json(data.TARGET_LANGUAGES_PATH)

    assert manifest["source_repository"] == "NoeFlandre/osm-polygon-wikidata-only"
    assert manifest["source_revision"] == ("c6b503908b4687517f546cecce40c621d4ee56ac")
    assert manifest["source_path"] == "docs/sentence-splitting.md"


def test_model_specs_record_xlmr_transfer_languages_and_revision():
    xlmr = next(spec for spec in MODELS if spec.key == "xlmr_ner_hrl")

    assert xlmr.revision == "253f557bd8249b8515114cfd7f71974fe5fa4d2f"
    assert xlmr.documented_languages is not None
    assert len(xlmr.documented_languages) == 10
    assert "WikiANN" in xlmr.overlap_note
