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


def _fence_marker(line: str) -> str:
    """Return the backtick or tilde run that opens or closes a code fence."""
    stripped = line.lstrip(" ")
    for char in ("`", "~"):
        run = len(stripped) - len(stripped.lstrip(char))
        if run >= 3:
            return char * run
    return ""


def _outside_code_fences(text: str) -> str:
    """Return a Markdown document with its fenced code blocks blanked out.

    Fenced lines become empty rather than disappearing so that line numbers and
    paragraph boundaries survive, which keeps a failure message pointing at the
    same place a reader sees in the rendered page.
    """
    prose: list[str] = []
    fence = ""
    for line in text.splitlines():
        visible, fence = _line_outside_fence(line, fence)
        prose.append(visible)
    return "\n".join(prose)


def _line_outside_fence(line: str, fence: str) -> tuple[str, str]:
    """Return visible prose and the updated fence marker for one line."""
    marker = _fence_marker(line)
    if not fence:
        if marker:
            return "", marker
        return line, ""
    if _closes_fence(marker, fence):
        return "", ""
    return "", fence


def _closes_fence(marker: str, fence: str) -> bool:
    """Whether a marker of the same kind closes the current fence run."""
    return bool(marker) and marker[0] == fence[0] and len(marker) >= len(fence)


def _markdown_transformer_pages() -> list[Path]:
    """Find public Markdown pages that mention the transformer model."""
    return [path for path in _transformer_sources() if path.suffix == ".md"]


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

    def test_markdown_transformer_pages_exist(self):
        """Keep a nonempty set of public Markdown transformer examples."""
        assert _markdown_transformer_pages()

    def test_markdown_examples_name_the_plugin(self):
        for path in _markdown_transformer_pages():
            assert PLUGIN in path.read_text(encoding="utf-8"), path

    def test_markdown_examples_state_the_python_version(self):
        for path in _markdown_transformer_pages():
            assert "Python 3.14" in path.read_text(encoding="utf-8"), path

    def test_markdown_examples_name_the_fallback_model(self):
        for path in _markdown_transformer_pages():
            assert ALTERNATIVE in path.read_text(encoding="utf-8"), path

    def test_code_fences_are_blanked_out(self):
        """Pin the stripper so the placement guard cannot pass vacuously."""
        page = "\n".join(
            [
                "before",
                "``` python",
                "inside",
                "```",
                "after",
            ]
        )

        assert _outside_code_fences(page) == "before\n\n\n\nafter"

    def test_markdown_prerequisite_examples_exist(self):
        """Keep a nonempty set of public prose pages with model guidance."""
        assert _markdown_transformer_pages()

    def test_prose_prerequisite_names_the_plugin(self):
        for path in _markdown_transformer_pages():
            prose = _outside_code_fences(path.read_text(encoding="utf-8"))
            assert PLUGIN in prose, path

    def test_prose_prerequisite_names_the_supported_python_version(self):
        for path in _markdown_transformer_pages():
            prose = _outside_code_fences(path.read_text(encoding="utf-8"))
            assert "Python 3.14" in prose, path

    def test_prose_prerequisite_explains_requested_language_fallback(self):
        for path in _markdown_transformer_pages():
            prose = _outside_code_fences(path.read_text(encoding="utf-8"))
            assert "requested language" in prose, path

    def test_prose_prerequisite_names_the_fallback_model(self):
        for path in _markdown_transformer_pages():
            prose = _outside_code_fences(path.read_text(encoding="utf-8"))
            assert ALTERNATIVE in prose, path

    def test_non_markdown_transformer_examples_exist(self):
        assert any(path.suffix != ".md" for path in _transformer_sources())

    @pytest.mark.parametrize(
        "source",
        [path for path in _transformer_sources() if path.suffix != ".md"],
    )
    def test_non_markdown_examples_pin_plugin_version(self, source):
        """Notebook and build examples use the spaCy-compatible plugin line."""
        assert PIN in source.read_text(encoding="utf-8"), source
