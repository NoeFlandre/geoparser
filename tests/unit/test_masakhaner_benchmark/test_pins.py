from scripts.masakhaner_benchmark.pins import (
    MISMATCH,
    MISSING,
    OK,
    check_directory,
    check_file,
)

HELLO = b"hello\n"
HELLO_SHA = "ce013625030ba8dba906f756967f9e9ca394464a"


def test_a_file_matching_its_blob_id_and_size_is_ok(tmp_path):
    path = tmp_path / "train.txt"
    path.write_bytes(HELLO)

    assert check_file(path, HELLO_SHA, len(HELLO)) == OK


def test_a_missing_file_is_reported_as_missing(tmp_path):
    assert check_file(tmp_path / "absent.txt", HELLO_SHA, len(HELLO)) == MISSING


def test_a_file_with_the_wrong_size_or_content_is_a_mismatch(tmp_path):
    path = tmp_path / "dev.txt"
    path.write_bytes(HELLO)

    assert check_file(path, HELLO_SHA, len(HELLO) + 1) == MISMATCH
    assert check_file(path, "0" * 40, len(HELLO)) == MISMATCH


def test_directory_check_reports_each_pinned_split(tmp_path):
    (tmp_path / "xyz").mkdir()
    (tmp_path / "xyz" / "train.txt").write_bytes(HELLO)
    (tmp_path / "xyz" / "test.txt").write_bytes(b"other\n")
    languages = [
        {
            "config": "xyz",
            "files": {
                "train": {
                    "path": "MasakhaNER2.0/data/xyz/train.txt",
                    "git_blob_sha": HELLO_SHA,
                    "size_bytes": len(HELLO),
                },
                "validation": {
                    "path": "MasakhaNER2.0/data/xyz/dev.txt",
                    "git_blob_sha": HELLO_SHA,
                    "size_bytes": len(HELLO),
                },
                "test": {
                    "path": "MasakhaNER2.0/data/xyz/test.txt",
                    "git_blob_sha": HELLO_SHA,
                    "size_bytes": len(HELLO),
                },
            },
        }
    ]

    rows = check_directory(tmp_path, languages)

    assert {row["split"]: row["status"] for row in rows} == {
        "train": OK,
        "validation": MISSING,
        "test": MISMATCH,
    }
