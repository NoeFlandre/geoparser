from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from tests.conftest import PROJECT_ROOT


def test_release_candidate_extracts_stable_release_notes(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n### Changed\n- Upcoming.\n\n"
        "## [0.6.0] - 2026-09-26\n\n"
        "### Added\n- Curated release note.\n\n"
        "## [0.5.0] - 2025-01-01\n\n### Fixed\n- Old note.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "changelog.py"),
            "0.6.0rc1",
            "--changelog",
            str(changelog),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "### Added\n- Curated release note.\n"


def test_release_changelog_rejects_missing_version_section(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "changelog.py"),
            "0.7.0",
            "--changelog",
            str(changelog),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "No changelog section for 0.7.0" in result.stderr


def test_existing_github_release_keeps_handwritten_notes() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    publish_script = next(
        step["run"]
        for step in workflow["jobs"]["github-release"]["steps"]
        if step.get("name") == "Publish the GitHub Release"
    )
    existing_release_script, new_release_script = publish_script.split("else", 1)

    assert "gh release edit" not in existing_release_script
    assert "--notes-file release-notes/release-notes.md" in new_release_script
