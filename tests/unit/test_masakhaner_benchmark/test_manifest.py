import copy

import pytest

from scripts.masakhaner_benchmark.data import LABEL_NAMES
from scripts.masakhaner_benchmark.manifest import (
    MANIFEST_PATH,
    read_manifest,
    validate_manifest,
)


@pytest.fixture
def manifest():
    return copy.deepcopy(read_manifest())


def _language(manifest, config):
    return next(row for row in manifest["languages"] if row["config"] == config)


def test_checked_in_manifest_is_valid_and_lists_twenty_configurations():
    manifest = read_manifest(MANIFEST_PATH)

    assert len(manifest["languages"]) == 20


def test_dataset_revisions_are_the_verified_commit_ids():
    dataset = read_manifest()["dataset"]

    assert dataset["huggingface_dataset_id"] == "masakhane/masakhaner2"
    assert dataset["huggingface_revision"] == (
        "60512e89e68841b6b5ed1be59caf97b169f0d27a"
    )
    assert dataset["github_commit"] == "ba5843cd08aa491d5f96a5e809e71eb9ec461391"


def test_label_names_in_the_manifest_match_the_parser():
    assert tuple(read_manifest()["label_names"]) == LABEL_NAMES


def test_the_license_conflict_is_recorded_and_not_resolved():
    manifest = read_manifest()
    sources = {row["source"]: row["value"] for row in manifest["licenses"]}

    assert "conflicting" in manifest["license_status"]
    assert sources["Hugging Face dataset metadata (license tag)"] == "afl-3.0"
    assert sources["GitHub MasakhaNER2.0/README.md"] == "CC-BY-4.0-NC"


@pytest.mark.parametrize(
    ("config", "counts", "test_blob"),
    [
        ("hau", (5716, 816, 1633), "85e294f04d2b400bef400c99a04cce5988e98f3f"),
        ("ibo", (7634, 1090, 2181), None),
        ("xho", (5718, 817, 1633), None),
        ("yor", (6877, 983, 1964), None),
        ("zul", (5848, 836, 1670), None),
    ],
)
def test_covered_configurations_record_the_readme_split_counts(
    config, counts, test_blob
):
    row = _language(read_manifest(), config)

    assert (
        row["readme_counts"]["train"],
        row["readme_counts"]["validation"],
        row["readme_counts"]["test"],
    ) == counts
    if test_blob is not None:
        assert row["files"]["test"]["git_blob_sha"] == test_blob
        assert row["files"]["test"]["size_bytes"] == 358507


@pytest.mark.parametrize(
    ("config", "counts"),
    [
        # Swapping these two rows leaves the totals unchanged, so pin each one.
        ("pcm", (5646, 806, 1294)),
        ("nya", (6250, 893, 1785)),
    ],
)
def test_pidgin_and_chichewa_counts_are_not_swapped(config, counts):
    row = _language(read_manifest(), config)

    assert (
        row["readme_counts"]["train"],
        row["readme_counts"]["validation"],
        row["readme_counts"]["test"],
    ) == counts


def test_all_configurations_together_match_the_readme_totals():
    rows = read_manifest()["languages"]
    totals = {
        split: sum(row["readme_counts"][split] for row in rows)
        for split in ("train", "validation", "test")
    }

    assert totals == {"train": 106766, "validation": 15282, "test": 30550}


def test_validation_split_is_the_upstream_dev_file():
    row = _language(read_manifest(), "hau")

    assert row["files"]["validation"]["path"] == "MasakhaNER2.0/data/hau/dev.txt"
    assert row["files"]["test"]["path"] == "MasakhaNER2.0/data/hau/test.txt"


def test_annotation_provenance_is_kept_with_the_guideline_pin():
    provenance = read_manifest()["annotation_provenance"]

    assert provenance["process_reference"] == "arXiv:2103.11811"
    assert provenance["guideline_git_blob_sha"] == (
        "3bdcb9914b5e18cbeb07fcd6b146c3b43ea0ac5f"
    )


def test_a_malformed_revision_is_rejected(manifest):
    manifest["dataset"]["huggingface_revision"] = "main"

    with pytest.raises(ValueError, match="Hugging Face revision"):
        validate_manifest(manifest)


def test_a_wrong_schema_version_is_rejected(manifest):
    manifest["schema_version"] = 2

    with pytest.raises(ValueError, match="schema version"):
        validate_manifest(manifest)


def test_a_changed_label_set_is_rejected(manifest):
    manifest["label_names"] = manifest["label_names"][:-1]

    with pytest.raises(ValueError, match="label names"):
        validate_manifest(manifest)


def test_a_duplicate_configuration_is_rejected(manifest):
    manifest["languages"][1]["config"] = manifest["languages"][0]["config"]

    with pytest.raises(ValueError, match="Duplicate config"):
        validate_manifest(manifest)


def test_two_configurations_cannot_claim_one_iso_639_1_code(manifest):
    _language(manifest, "xho")["iso639_1"] = "ha"

    with pytest.raises(ValueError, match="Duplicate iso639_1"):
        validate_manifest(manifest)


def test_a_file_pinned_to_another_configuration_is_rejected(manifest):
    _language(manifest, "hau")["files"]["test"]["path"] = (
        "MasakhaNER2.0/data/ibo/test.txt"
    )

    with pytest.raises(ValueError, match="unexpected path"):
        validate_manifest(manifest)


def test_a_blob_pin_that_is_not_a_git_id_is_rejected(manifest):
    _language(manifest, "hau")["files"]["train"]["git_blob_sha"] = "abc"

    with pytest.raises(ValueError, match="invalid blob pin"):
        validate_manifest(manifest)


def test_a_malformed_github_commit_is_rejected(manifest):
    manifest["dataset"]["github_commit"] = "master"

    with pytest.raises(ValueError, match="GitHub commit must be"):
        validate_manifest(manifest)


def test_the_manifest_must_list_twenty_configurations(manifest):
    manifest["languages"] = manifest["languages"][:-1]

    with pytest.raises(ValueError, match="must list 20 configurations"):
        validate_manifest(manifest)


def test_a_configuration_must_record_all_three_split_counts(manifest):
    del _language(manifest, "hau")["readme_counts"]["test"]

    with pytest.raises(ValueError, match="must record train, validation and test"):
        validate_manifest(manifest)


def test_a_configuration_must_pin_all_three_split_files(manifest):
    del _language(manifest, "hau")["files"]["test"]

    with pytest.raises(ValueError, match="must pin train, validation and test"):
        validate_manifest(manifest)


def test_a_manifest_file_must_hold_a_json_object(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="Expected a JSON object"):
        read_manifest(path)
