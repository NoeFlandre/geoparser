"""Regression checks for public spaCy transformer documentation."""

from pathlib import Path

import pytest

MODEL = "en_core_web_trf"
PLUGIN = "spacy-curated-transformers"
PIN = "spacy-curated-transformers>=0.3.1,<1"
ALTERNATIVE = "en_core_web_lg"
# The surfaces a reader meets before running anything: the landing page, the
# published documentation, and the runnable demo.
PUBLIC_ROOTS = ("README.md", "docs", "demo")


def _repository_root() -> Path:
    """Return the checkout that carries the public documentation.

    Walking up beats a fixed parent count because mutmut runs the suite from a
    copied ``mutants`` tree that holds only the mutated package. That copy has
    no documentation to scan, and no repository metadata either, so asking the
    version control system for the file list fails there outright.
    """
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / root).exists() for root in PUBLIC_ROOTS):
            return candidate
    raise RuntimeError(f"no checkout above {__file__} contains {PUBLIC_ROOTS}")


def _public_text_files() -> list[Path]:
    """Return readable text files on the project's public surfaces."""
    root = _repository_root()
    candidates: list[Path] = []
    for name in PUBLIC_ROOTS:
        source = root / name
        candidates.extend([source] if source.is_file() else sorted(source.rglob("*")))
    files = []
    for path in candidates:
        try:
            path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files.append(path)
    return files


def _transformer_sources() -> list[Path]:
    """Find public sources that present the transformer model."""
    return sorted(
        path
        for path in _public_text_files()
        if MODEL in path.read_text(encoding="utf-8")
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
