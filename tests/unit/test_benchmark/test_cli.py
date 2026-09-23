"""
Tests for the benchmark command line.

Several corpora share one output directory, so each gets its own folder:
checkpoints of two corpora must never be mistaken for one another.
"""

from scripts.benchmark.__main__ import build_parser, corpus_output_dir


class TestCorpusOption:
    """Choosing which corpora to score."""

    def test_defaults_to_no_explicit_corpus(self):
        """Test that omitting --corpus leaves the registry default to apply."""
        assert build_parser().parse_args([]).corpus is None

    def test_accepts_several_corpora(self):
        """Test that --corpus repeats."""
        arguments = build_parser().parse_args(
            ["--corpus", "hipe2020-de", "--corpus", "newseye-fi"]
        )

        assert arguments.corpus == ["hipe2020-de", "newseye-fi"]

    def test_accepts_all(self):
        """Test that every registered corpus can be asked for at once."""
        assert build_parser().parse_args(["--corpus", "all"]).corpus == ["all"]


class TestCorpusOutputDir:
    """Where each corpus's checkpoints and report go."""

    def test_each_corpus_gets_its_own_folder(self, tmp_path):
        """Test that two corpora cannot share a checkpoint path."""
        assert corpus_output_dir(tmp_path, "hipe2020-de") == tmp_path / "hipe2020-de"
        assert corpus_output_dir(tmp_path, "geovirus") != corpus_output_dir(
            tmp_path, "newseye-fi"
        )
