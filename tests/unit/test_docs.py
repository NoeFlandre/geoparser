"""Regression checks for public spaCy transformer documentation."""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = Path(__file__).resolve()
MODEL = "en_core_web_trf"
PLUGIN = "spacy-curated-transformers"
PIN = "spacy-curated-transformers>=0.3.1,<1"
ALTERNATIVE = "en_core_web_lg"


def _tracked_text_files() -> list[Path]:
    """Return tracked text files that can contain public examples."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    files = []
    for name in result.stdout.split("\0"):
        if not name:
            continue
        path = REPO_ROOT / name
        try:
            path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files.append(path)
    return files


def _transformer_sources() -> list[Path]:
    """Find tracked public sources that present the transformer model."""
    return sorted(
        path
        for path in _tracked_text_files()
        if path != GUARD
        and "tests" not in path.relative_to(REPO_ROOT).parts
        and MODEL in path.read_text(encoding="utf-8")
    )


@pytest.mark.unit
class TestTransformerDocumentation:
    """Every public transformer example explains its runtime requirement."""

    def test_transformer_sources_are_discovered(self):
        """Ensure this guard cannot pass by scanning nothing."""
        assert _transformer_sources()

    def test_markdown_pages_explain_plugin_and_fallback(self):
        """Markdown examples include the prerequisite and safe alternative."""
        pages = [path for path in _transformer_sources() if path.suffix == ".md"]

        assert pages
        for path in pages:
            text = path.read_text(encoding="utf-8")
            assert PLUGIN in text, path
            assert "Python 3.14" in text, path
            assert ALTERNATIVE in text, path

    def test_non_markdown_examples_pin_plugin_version(self):
        """Notebook and build examples use the spaCy-compatible plugin line."""
        sources = [path for path in _transformer_sources() if path.suffix != ".md"]

        assert sources
        for path in sources:
            assert PIN in path.read_text(encoding="utf-8"), path
