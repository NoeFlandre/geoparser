from pathlib import Path

import pytest

from scripts import changelog


@pytest.mark.parametrize(
    ("tag", "expected"), [("0.6.0", "0.6.0"), ("0.6.0rc2", "0.6.0")]
)
def test_release_notes_select_the_stable_section(tag: str, expected: str) -> None:
    text = "## [0.6.0] - 2026-09-30\n\nRelease notes.\n\n## [0.5.0]\n\nOlder."

    notes = changelog.extract_release_notes(text, tag)

    assert changelog._stable_version(tag) == expected
    assert notes == "Release notes."


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("## [0.5.0]\n\nOlder.", "No changelog section for 0.6.0"),
        ("## [0.6.0]\n\n## [0.5.0]\n\nOlder.", "section for 0.6.0 is empty"),
    ],
)
def test_release_notes_reject_missing_or_empty_sections(
    text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        changelog.extract_release_notes(text, "0.6.0")


def test_changelog_cli_prints_the_selected_notes(tmp_path: Path, capsys) -> None:
    path = tmp_path / "CHANGELOG.md"
    path.write_text("## [1.0.0]\n\nShipped.\n", encoding="utf-8")

    exit_code = changelog.main(["1.0.0", "--changelog", str(path)])

    assert exit_code == 0
    assert capsys.readouterr().out == "Shipped.\n"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing-file", "No such file or directory"),
        ("missing-section", "No changelog section for 1.0.0"),
    ],
)
def test_changelog_cli_reports_read_and_lookup_errors(
    tmp_path: Path, capsys, failure: str, message: str
) -> None:
    path = tmp_path / "CHANGELOG.md"
    if failure == "missing-section":
        path.write_text("## [0.5.0]\n\nOld.\n", encoding="utf-8")

    exit_code = changelog.main(["1.0.0", "--changelog", str(path)])

    assert exit_code == 1
    assert message in capsys.readouterr().err
