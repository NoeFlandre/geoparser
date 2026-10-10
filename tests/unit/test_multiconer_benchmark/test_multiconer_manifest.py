"""Pinned MultiCoNER II metadata: revision, license, counts and digest policy."""

import copy
import json

import pytest

from scripts.multiconer_benchmark.manifest import (
    DEFAULT_MANIFEST_PATH,
    intersection_languages,
    load_manifest,
    validate_manifest,
)

# Viewer num_rows from the Hub dataset-viewer size endpoint at the pinned revision.
VIEWER_ROWS = {
    "bn": 30074,
    "de": 30442,
    "en": 267629,
    "es": 264207,
    "fa": 236344,
    "fr": 267191,
    "hi": 28545,
    "it": 265318,
    "pt": 246813,
    "sv": 248409,
    "uk": 255576,
    "zh": 30530,
}


def test_manifest_pins_the_verified_revision_and_license():
    manifest = load_manifest()
    assert manifest["dataset"]["id"] == "MultiCoNER/multiconer_v2"
    assert manifest["dataset"]["revision"] == "4be2d62c912977ee26ed14d2553a4fe17ca3d980"
    assert manifest["license"]["spdx"] == "CC-BY-4.0"
    assert manifest["clean_noisy_breakdown"]["supported_by_release"] is False


def test_intersection_with_target_languages_is_exactly_the_twelve_dataset_codes():
    assert set(intersection_languages(load_manifest())) == set(VIEWER_ROWS)
    assert "multi" not in intersection_languages(load_manifest())


def test_multilingual_configuration_is_excluded_with_a_recorded_reason():
    excluded = load_manifest()["excluded_configurations"]["multi"]
    assert excluded["viewer_num_rows"] == 538387
    assert "Aggregate" in excluded["reason"]


@pytest.mark.parametrize("code", sorted(VIEWER_ROWS))
def test_official_split_counts_sum_to_the_independent_viewer_total(code):
    language = load_manifest()["languages"][code]
    splits = language["splits"]
    assert splits["train"] + splits["dev"] + splits["test"] == VIEWER_ROWS[code]
    assert language["viewer_num_rows"] == VIEWER_ROWS[code]


def test_english_test_file_digest_is_the_lfs_sha256_at_the_pinned_revision():
    test_file = load_manifest()["languages"]["en"]["files"]["test"]
    assert test_file["path"] == "EN-English/en_test.conll"
    assert test_file["bytes"] == 62002874
    assert test_file["digest_algorithm"] == "sha256"
    assert (
        test_file["sha256"]
        == "be7e85f1542a9643de5f01cc99465ec0cb9d96a20bdc3c58418d391e194aff9a"
    )


def test_non_lfs_test_files_claim_only_a_git_blob_digest():
    bengali = load_manifest()["languages"]["bn"]["files"]["test"]
    assert bengali["digest_algorithm"] == "git-blob-sha1"
    assert bengali["digest"] == "f76dc9355ca0336fefd724b7dd004475d04091bc"
    assert bengali["sha256"] is None


def test_manifest_rejects_a_moved_revision():
    payload = load_manifest()
    payload = copy.deepcopy(payload)
    payload["dataset"]["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        validate_manifest(payload)


def test_manifest_rejects_another_full_commit():
    payload = copy.deepcopy(load_manifest())
    payload["dataset"]["revision"] = "a" * 40
    with pytest.raises(ValueError, match="revision"):
        validate_manifest(payload)


def test_manifest_rejects_a_count_that_does_not_match_the_viewer():
    payload = copy.deepcopy(load_manifest())
    payload["languages"]["en"]["splits"]["test"] += 1
    with pytest.raises(ValueError, match="viewer"):
        validate_manifest(payload)


def test_manifest_rejects_a_sha256_claim_on_a_non_lfs_file():
    payload = copy.deepcopy(load_manifest())
    payload["languages"]["bn"]["files"]["test"]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="sha256"):
        validate_manifest(payload)


def test_default_manifest_is_packaged_beside_the_module():
    assert DEFAULT_MANIFEST_PATH.name == "multiconer_manifest.json"
    json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))


def _mutated(change):
    payload = copy.deepcopy(load_manifest())
    change(payload)
    return payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p["dataset"].update(id="Other/dataset"), "different dataset"),
        (lambda p: p["license"].update(spdx="MIT"), "CC-BY-4.0"),
        (lambda p: p["languages"].update(multi=p["languages"]["en"]), "MULTI"),
        (lambda p: p["languages"].update(xx=p["languages"]["en"]), "85-code"),
        (lambda p: p["languages"]["de"]["splits"].update(dev="507"), "integer"),
        (
            lambda p: p["languages"]["de"]["files"]["dev"].update(path="wrong.conll"),
            "file path",
        ),
        (lambda p: p["languages"]["de"]["files"]["dev"].update(bytes=0), "byte size"),
        (
            lambda p: p["languages"]["de"]["files"]["dev"].update(
                digest_algorithm="md5"
            ),
            "unknown digest algorithm",
        ),
        (
            lambda p: p["languages"]["de"]["files"]["dev"].update(digest="xyz"),
            "40 hex",
        ),
        (
            lambda p: p["languages"]["en"]["files"]["test"].update(sha256="0" * 64),
            "LFS digest",
        ),
    ],
)
def test_manifest_rejects_each_inconsistent_pin(change, message):
    with pytest.raises(ValueError, match=message):
        validate_manifest(_mutated(change))
