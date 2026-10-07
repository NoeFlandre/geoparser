"""Test source pins and split isolation without remote data."""

import hashlib

import pytest
from pydantic import ValidationError

from scripts.uner_benchmark.inventory import (
    Dataset,
    File,
    inventory_summary,
    load_local,
)

PAYLOAD = b"# sent_id = one\n# text = Paris\n1\tParis\tB-LOC\t-\t-\n"
# Calculated independently with git hash-object, then fixed in the fixture.


def dataset(**updates):
    values = {
        "configuration": "en_fixture",
        "repository": "UNER_English-EWT",
        "revision": "8ed072de2fd24022cc62458997fe96a8fe191ea4",
        "source_language": "en",
        "target_language": "en",
        "domains": ["web"],
        "annotation_provenance": "Human UNER annotation",
        "domain_source": "Synthetic test metadata",
        "mapping_note": "Synthetic test mapping",
        "license_note": "Synthetic test license source",
        "license": "CC-BY-SA-4.0",
        "license_sources": [
            {
                "path": "LICENSE.txt",
                "git_blob_sha1": "547fb999d1fc888c7071f50a0cf6b7338eeae2fe",
            }
        ],
        "splits": {
            "test": {
                "path": "en_fixture-ud-test.iob2",
                "git_blob_sha1": "2192002ee27e73bfd0ecd4ac26f976dec4ef7786",
            }
        },
    }
    values.update(updates)
    return Dataset.model_validate(values)


def test_load_checks_pin_preserves_ids_and_records_sha256(tmp_path):
    spec = dataset()
    path = tmp_path / spec.repository / "en_fixture-ud-test.iob2"
    path.parent.mkdir()
    path.write_bytes(PAYLOAD)
    loaded = load_local(spec, "test", tmp_path)
    assert loaded.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert loaded.corpus.sentences[0].identifier == "en_fixture/test/one"
    assert loaded.corpus.sentences[0].example.gold_spans == {(0, 5)}
    assert (loaded.corpus.sentence_count, loaded.path) == (1, path)


def test_load_rejects_malformed_tail_after_verified_prefix(tmp_path):
    payload = PAYLOAD + (b"\n# sent_id = broken\n# text = London\n1\tLondon\tB-LOC\n")
    spec = dataset(
        splits={
            "test": {
                "path": "en_fixture-ud-test.iob2",
                # Independently verified with git hash-object --stdin.
                "git_blob_sha1": "5a466fe3eca74da7a46cae20a5d46b5a9d9c4bb1",
            }
        }
    )
    path = tmp_path / spec.repository / "en_fixture-ud-test.iob2"
    path.parent.mkdir()
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="five tab-separated columns"):
        load_local(spec, "test", tmp_path)


def test_checksum_mismatch_rejected_before_parsing(tmp_path):
    spec = dataset()
    path = tmp_path / spec.repository / "en_fixture-ud-test.iob2"
    path.parent.mkdir()
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        load_local(spec, "test", tmp_path)


def test_missing_split_does_not_fall_back_to_test(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_local(dataset(), "dev", tmp_path)


def test_missing_file_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_local(dataset(), "test", tmp_path)


@pytest.mark.parametrize(
    "path", ["../data", "/data", "C:/data", "a\\b", "a/../b", "", "./file"]
)
def test_source_paths_cannot_escape_local_cache(path):
    with pytest.raises(ValidationError):
        File(path=path, git_blob_sha1="a" * 40)


@pytest.mark.parametrize("value", ["main", "a" * 39, "A" * 40, "z" * 40])
def test_revisions_require_full_git_commit_hashes(value):
    with pytest.raises(ValidationError):
        dataset(revision=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("configuration", "../en"),
        ("repository", "../repo"),
        ("target_language", "hr"),
        ("license", ""),
        ("domains", []),
        ("splits", {}),
        ("license_sources", []),
    ],
)
def test_incomplete_and_unsafe_metadata_rejected(field, value):
    with pytest.raises(ValidationError):
        dataset(**{field: value})


def test_configurations_keep_variants_but_share_explicit_target_language():
    specs = [
        dataset(),
        dataset(configuration="nb_ndt", source_language="nb", target_language="no"),
        dataset(configuration="nn_ndt", source_language="nn", target_language="no"),
    ]
    summary = inventory_summary(specs, "test")
    assert summary["covered_target_languages"] == ["en", "no"]
    assert len(summary["missing_target_languages"]) == 83
    assert summary["selected_configurations"] == ["en_fixture", "nb_ndt", "nn_ndt"]
    assert (summary["task"], summary["training_overlap"]) == ("recognition", "unknown")


def test_split_inventory_does_not_claim_test_only_as_development():
    summary = inventory_summary([dataset()], "dev")
    assert summary["covered_target_languages"] == []
    assert len(summary["missing_target_languages"]) == 85
    assert summary["configurations"][0]["status"] == "split_unavailable"
    assert summary["selected_configurations"] == []


def test_outside_target_languages_remain_visible():
    summary = inventory_summary(
        [dataset(configuration="tl_trg", source_language="tl", target_language=None)],
        "test",
    )
    assert summary["configurations"][0]["status"] == "outside_target_inventory"
    assert summary["covered_target_languages"] == []


def test_duplicate_configuration_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        inventory_summary([dataset(), dataset()], "test")


def test_unavailable_license_blocks_selection():
    summary = inventory_summary([dataset(license="unknown")], "test")
    assert summary["configurations"][0]["status"] == "license_unverified"
    assert summary["selected_configurations"] == []


def test_manifest_reads_only_explicit_local_metadata(tmp_path):
    import json

    from scripts.uner_benchmark.inventory import read_manifest

    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "audit_status": "complete",
                "expected_configuration_count": 1,
                "datasets": [dataset().model_dump(mode="json")],
            }
        )
    )
    assert read_manifest(path) == (dataset(),)


def test_nested_source_paths_and_immutable_urls():
    spec = dataset()
    source = File(path="nested/test.iob2", git_blob_sha1="a" * 40)
    assert (
        spec.url(source)
        == "https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/8ed072de2fd24022cc62458997fe96a8fe191ea4/nested/test.iob2"
    )


@pytest.mark.parametrize("sha", ["main", "a" * 39, "A" * 40, "g" * 40])
def test_file_pins_require_full_lowercase_git_object_ids(sha):
    with pytest.raises(ValidationError):
        File(path="test.iob2", git_blob_sha1=sha)


def test_extra_unreviewed_metadata_is_not_silently_accepted():
    with pytest.raises(ValidationError):
        dataset(coordinates=[0, 0])


def test_double_slash_absolute_source_path_is_rejected():
    with pytest.raises(ValidationError):
        File(path="//data", git_blob_sha1="a" * 40)


@pytest.mark.parametrize("path", ["en_fixture-ud-test.iob2", "copied-test.iob2"])
def test_split_paths_and_source_payloads_cannot_alias(path):
    test_file = dataset().splits["test"].model_dump()
    dev_file = {**test_file, "path": path}
    with pytest.raises(ValidationError, match="split"):
        dataset(splits={"test": test_file, "dev": dev_file})


def test_incomplete_release_inventory_cannot_be_loaded(tmp_path):
    import json

    from scripts.uner_benchmark.inventory import read_manifest

    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "audit_status": "incomplete",
                "datasets": [dataset().model_dump(mode="json")],
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        read_manifest(path)


def test_manifest_rejects_incomplete_configuration_count(tmp_path):
    import json

    from scripts.uner_benchmark.inventory import read_manifest

    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "audit_status": "complete",
                "expected_configuration_count": 30,
                "datasets": [dataset().model_dump(mode="json")],
            }
        )
    )
    with pytest.raises(ValueError, match="count"):
        read_manifest(path)


@pytest.mark.parametrize("value", [[], None, "text", 5])
def test_manifest_requires_a_json_object(tmp_path, value):
    import json

    from scripts.uner_benchmark.inventory import read_manifest

    path = tmp_path / "sources.json"
    path.write_text(json.dumps(value))
    with pytest.raises(TypeError, match="object"):
        read_manifest(path)


def test_unknown_license_can_explicitly_have_no_license_source():
    spec = dataset(license="unknown", license_sources=[])
    assert spec.status("test") == "license_unverified"
    assert spec.license_sources == ()


def test_official_inventory_accounts_for_all_thirty_configurations():
    from scripts.uner_benchmark.inventory import read_manifest

    specs = read_manifest()
    assert {s.configuration for s in specs} == {
        "ceb_gja",
        "cs_pud",
        "da_ddt",
        "de_pud",
        "el_gdt",
        "en_ewt",
        "en_pud",
        "he_htb",
        "hr_set",
        "id_pud",
        "ja_pud",
        "ko_pud",
        "nno_norne",
        "nob_norne",
        "pt_bosque",
        "pt_pud",
        "qaf_arabizi",
        "ro_legalnero",
        "ru_pud",
        "sk_snk",
        "sl_ssj",
        "sr_set",
        "sv_lines",
        "sv_pud",
        "sv_talbanken",
        "tl_trg",
        "tl_ugnayan",
        "zh_gsd",
        "zh_gsdsimp",
        "zh_pud",
    }
    assert len({s.repository for s in specs}) == 29
    assert {s.configuration for s in specs if s.license == "unknown"} == {
        "el_gdt",
        "ko_pud",
    }
    assert {s.configuration for s in specs if s.target_language is None} == {
        "hr_set",
        "qaf_arabizi",
        "tl_trg",
        "tl_ugnayan",
    }


def test_source_presence_and_license_eligibility_are_distinct():
    from scripts.uner_benchmark.inventory import read_manifest

    summary = inventory_summary(read_manifest(), "test")
    assert summary["present_target_languages"] == [
        "ceb",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "he",
        "id",
        "ja",
        "ko",
        "no",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sr",
        "sv",
        "zh",
    ]
    assert summary["unavailable_target_languages"] == ["el", "ko"]
    assert (
        len(summary["covered_target_languages"]),
        len(summary["missing_target_languages"]),
        len(summary["selected_configurations"]),
    ) == (17, 66, 24)
    assert set(summary["missing_target_languages"]).isdisjoint({"el", "ko"})


def test_sources_preserve_variants_licenses_and_test_only_split():
    from scripts.uner_benchmark.inventory import read_manifest

    specs = {s.configuration: s for s in read_manifest()}
    assert (specs["nno_norne"].target_language, specs["nob_norne"].target_language) == (
        "no",
        "no",
    )
    assert (
        specs["en_pud"].license,
        specs["sv_pud"].license,
        specs["tl_ugnayan"].license,
    ) == ("CC-BY-SA-3.0", "CC-BY-SA-4.0", "CC-BY-NC-SA-4.0")
    assert (
        set(specs["ro_legalnero"].splits),
        specs["ro_legalnero"].splits["test"].path,
    ) == ({"test"}, "ro_legalnero.iob2")


def test_domains_count_source_configurations_without_claiming_sentence_shares():
    summary = inventory_summary(
        [dataset(), dataset(configuration="other", domains=["web", "news"])], "test"
    )
    assert summary["domain_configuration_counts"] == {"news": 1, "web": 2}
