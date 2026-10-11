import json

import pytest

from scripts.masakhaner_benchmark.coverage import coverage_rows, coverage_summary
from scripts.masakhaner_benchmark.manifest import read_manifest
from scripts.masakhaner_benchmark.report import CAVEATS, render_markdown, write_report


@pytest.fixture(scope="module")
def manifest():
    return read_manifest()


def test_report_files_are_written_and_the_json_reads_back(tmp_path, manifest):
    json_path, markdown_path = write_report(tmp_path / "out", manifest)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert markdown_path.name == "report.md"
    assert payload["summary"]["covered_codes"] == ["ha", "ig", "xh", "yo", "zu"]
    assert payload["caveats"] == list(CAVEATS)
    assert payload["license_status"] == manifest["license_status"]


def test_rewriting_the_report_gives_identical_bytes(tmp_path, manifest):
    first_json, first_md = write_report(tmp_path / "a", manifest)
    second_json, second_md = write_report(tmp_path / "b", manifest)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()


def test_markdown_lists_each_covered_target_with_its_counts(manifest):
    rows = coverage_rows(manifest)
    text = render_markdown(manifest, rows, coverage_summary(manifest, rows))

    assert "| ha | covered | hau | missing | 5716 | 816 | 1633 |" in text
    assert "| am | missing | - | present | - | - | - |" in text
    assert "- Missing: 80 of 85." in text
    assert "Status: conflicting" in text


def test_markdown_names_each_excluded_configuration_and_why(manifest):
    rows = coverage_rows(manifest)
    text = render_markdown(manifest, rows, coverage_summary(manifest, rows))

    assert "| bbj | bbj | - | Ghomala | no ISO 639-1 code" in text
    assert "| swa | swa | sw | Kiswahili | its ISO 639-1 code" in text
