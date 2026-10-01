import sys
from types import SimpleNamespace

import pytest

from scripts.panx_benchmark import data
from scripts.panx_benchmark.constants import DATASET_ID, DATASET_REVISION, MODELS


def test_target_language_manifest_has_85_unique_codes():
    languages = data.target_languages()

    assert (len(languages), len(set(languages))) == (85, 85)


def test_wikiann_test_split_intersection_names_missing_targets():
    manifest = data.split_manifest()

    assert (
        len(manifest["eligible_languages"]),
        set(manifest["missing_target_languages"]),
    ) == (
        82,
        {"ha", "xh", "zu"},
    )


def test_wikiann_manifest_records_full_test_row_count():
    assert data.split_manifest()["total_test_examples"] == 423_100


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_id", "another-dataset", "different dataset"),
        ("dataset_revision", "another-revision", "different revision"),
        ("split", "train", "pinned test split"),
    ],
)
def test_split_manifest_rejects_a_different_dataset_snapshot(
    monkeypatch, field, value, message
):
    manifest = data.read_json(data.TEST_SPLITS_PATH)
    manifest[field] = value
    monkeypatch.setattr(data, "read_json", lambda _path: manifest)

    with pytest.raises(ValueError, match=message):
        data.split_manifest()


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


@pytest.fixture
def fake_wikiann_loader(monkeypatch):
    counts = data.split_manifest()["test_examples_by_language"]
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
    return calls


def test_loader_checks_revision_counts_and_limits_rows(fake_wikiann_loader, tmp_path):
    calls = fake_wikiann_loader

    loaded = data.load_test_examples(tmp_path, limit_per_language=1)

    assert (
        loaded.evaluated_example_count,
        loaded.source_example_count,
        loaded.limit_per_language,
        len(calls),
    ) == (82, 423_100, 1, 82)
    assert {call[0] for call in calls} == {DATASET_ID}
    assert {call[2:4] for call in calls} == {("test", DATASET_REVISION)}


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


def test_xlmr_model_revision_is_pinned():
    xlmr = next(spec for spec in MODELS if spec.key == "xlmr_ner_hrl")

    assert xlmr.revision == "253f557bd8249b8515114cfd7f71974fe5fa4d2f"


def test_xlmr_model_documents_ten_finetuned_languages():
    xlmr = next(spec for spec in MODELS if spec.key == "xlmr_ner_hrl")

    assert len(xlmr.documented_languages or ()) == 10


def test_xlmr_overlap_note_names_wikiann():
    xlmr = next(spec for spec in MODELS if spec.key == "xlmr_ner_hrl")

    assert "WikiANN" in xlmr.overlap_note
