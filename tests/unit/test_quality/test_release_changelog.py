from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
