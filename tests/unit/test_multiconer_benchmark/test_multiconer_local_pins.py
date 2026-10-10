"""Local split files must match the pinned size and digest before they are parsed."""

import copy

import pytest

from scripts.multiconer_benchmark.manifest import (
    _local_digest,
    check_local_file,
    load_manifest,
)

HELLO = b"hello\n"
HELLO_GIT_BLOB = "ce013625030ba8dba906f756967f9e9ca394464a"
HELLO_SHA256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def _manifest_with_en_test(bytes_, digest_algorithm, digest):
    manifest = copy.deepcopy(load_manifest())
    entry = manifest["languages"]["en"]["files"]["test"]
    entry.update(bytes=bytes_, digest_algorithm=digest_algorithm, digest=digest)
    return manifest


def test_git_blob_digest_matches_git_hash_object():
    assert _local_digest(HELLO, "git-blob-sha1") == HELLO_GIT_BLOB


def test_lfs_digest_is_the_plain_sha256():
    assert _local_digest(HELLO, "sha256") == HELLO_SHA256


def test_a_file_matching_its_pinned_size_and_digest_is_accepted(tmp_path):
    path = tmp_path / "en_test.conll"
    path.write_bytes(HELLO)
    manifest = _manifest_with_en_test(len(HELLO), "git-blob-sha1", HELLO_GIT_BLOB)
    check_local_file(path, "en", "test", manifest)


def test_a_file_of_another_size_is_refused_before_it_is_parsed(tmp_path):
    path = tmp_path / "en_test.conll"
    path.write_bytes(HELLO)
    manifest = _manifest_with_en_test(999, "git-blob-sha1", HELLO_GIT_BLOB)
    with pytest.raises(ValueError, match="has 6 bytes; the pinned en test file"):
        check_local_file(path, "en", "test", manifest)


def test_a_file_with_the_right_size_and_another_digest_is_refused(tmp_path):
    path = tmp_path / "en_test.conll"
    path.write_bytes(HELLO)
    manifest = _manifest_with_en_test(len(HELLO), "sha256", HELLO_GIT_BLOB)
    with pytest.raises(ValueError, match="does not match the pinned digest"):
        check_local_file(path, "en", "test", manifest)


def test_a_language_without_a_pinned_split_is_refused(tmp_path):
    path = tmp_path / "multi_test.conll"
    path.write_bytes(HELLO)
    with pytest.raises(ValueError, match="has no pinned split"):
        check_local_file(path, "multi", "test", load_manifest())
