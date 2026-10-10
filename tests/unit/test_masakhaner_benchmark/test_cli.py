import pytest

from scripts.masakhaner_benchmark import __main__ as cli
from scripts.masakhaner_benchmark.data import git_blob_sha1


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_inventory_writes_the_report_to_the_given_directory(tmp_path):
    output = tmp_path / "inventory"

    assert cli.main(["inventory", "--output-dir", str(output)]) == 0

    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()


def test_inventory_defaults_to_a_dated_evidence_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["inventory"]) == 0

    default = tmp_path / "benchmark-evidence" / "masakhaner" / "inventory-2026-10-09"
    assert (default / "report.md").is_file()


def test_verify_fails_when_the_pinned_files_are_absent(tmp_path, capsys):
    assert cli.main(["verify", "--data-dir", str(tmp_path)]) == 1

    output = capsys.readouterr().out
    assert "missing  bam train" in output


def test_verify_passes_when_every_pinned_file_matches(tmp_path, monkeypatch):
    content = b"Kano B-LOC\n\n"
    (tmp_path / "xyz").mkdir()
    for name in ("train.txt", "dev.txt", "test.txt"):
        (tmp_path / "xyz" / name).write_bytes(content)
    pin = {
        "path": "MasakhaNER2.0/data/xyz/train.txt",
        "git_blob_sha": git_blob_sha1(content),
        "size_bytes": len(content),
    }
    languages = [
        {
            "config": "xyz",
            "files": {
                "train": pin,
                "validation": {**pin, "path": "MasakhaNER2.0/data/xyz/dev.txt"},
                "test": {**pin, "path": "MasakhaNER2.0/data/xyz/test.txt"},
            },
        }
    ]
    monkeypatch.setattr(cli, "read_manifest", lambda: {"languages": languages})

    assert cli.main(["verify", "--data-dir", str(tmp_path)]) == 0
