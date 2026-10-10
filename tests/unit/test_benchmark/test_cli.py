"""
Tests for the benchmark command line.

Several corpora share one output directory, so each gets its own folder:
checkpoints of two corpora must never be mistaken for one another.
"""

import argparse
import os

import pytest

from scripts.benchmark import __main__ as benchmark_cli
from scripts.benchmark import corpora, pipelines, runner
from scripts.benchmark.__main__ import build_parser, corpus_output_dir
from scripts.benchmark.corpora import LoadedCorpus
from scripts.benchmark.corpus import Document, GoldSpan
from scripts.benchmark.report import PipelineResult


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


def test_score_requested_pipelines_uses_defaults_and_explicit_choices(
    monkeypatch, tmp_path
):
    run = benchmark_cli._CorpusRun(
        LoadedCorpus("fixture", "en", [], "digest"),
        argparse.Namespace(pipeline=None, phase=None),
        tmp_path,
        "cpu",
        "commit",
        {},
    )
    scored = []

    def score_pipeline(run, name, phases):
        scored.append((name, phases))
        return name

    monkeypatch.setattr(benchmark_cli, "_score_pipeline", score_pipeline)

    defaults = benchmark_cli._score_requested_pipelines(run)
    run.arguments.pipeline = ["swapped", "hybrid"]
    run.arguments.phase = [runner.RESOLUTION]
    selected = benchmark_cli._score_requested_pipelines(run)

    assert defaults == list(pipelines.DEFAULT_PIPELINES)
    assert selected == ["swapped", "hybrid"]
    assert scored == [
        (name, [runner.RECOGNITION, runner.RESOLUTION])
        for name in pipelines.DEFAULT_PIPELINES
    ] + [("swapped", [runner.RESOLUTION]), ("hybrid", [runner.RESOLUTION])]


def _execute_score_pipeline(monkeypatch, tmp_path):
    """Run both phases with a controlled checkpoint and scoring backend."""
    document = Document("doc", "Zurich", ())
    loaded = LoadedCorpus("fixture", "en", [document], "digest")
    arguments = argparse.Namespace(
        min_similarity=0.4, limit=7, chunk_size=3, pipeline=None, phase=None
    )
    run = benchmark_cli._CorpusRun(
        loaded, arguments, tmp_path, "cpu", "commit", {"host": "fixture"}
    )
    state = object()
    identities = []
    phases = []

    def load(path, identity):
        identities.append((path, identity))
        return state, ["schema changed"]

    def run_phase(
        phase, pipeline_name, documents, checkpoint, checkpoint_path, settings
    ):
        phases.append((phase, pipeline_name, documents, checkpoint, settings))
        return {phase: f"model-{phase}"}

    expected = PipelineResult(name="swapped", device="cpu")
    monkeypatch.setattr(benchmark_cli.ckpt, "load", load)
    monkeypatch.setattr(benchmark_cli.runner, "run_phase", run_phase)
    monkeypatch.setattr(benchmark_cli.runner, "score", lambda *args, **kwargs: expected)

    result = benchmark_cli._score_pipeline(
        run, "swapped", [runner.RECOGNITION, runner.RESOLUTION]
    )
    return result, expected, identities, phases, state, run


def test_score_pipeline_resumes_requested_phases(monkeypatch, tmp_path):
    result, expected, _identities, phases, state, _run = _execute_score_pipeline(
        monkeypatch, tmp_path
    )

    assert result is expected
    assert [phases[0][0], phases[1][0]] == [runner.RECOGNITION, runner.RESOLUTION]
    assert phases[0][3] is state
    assert phases[1][3] is state


def test_score_pipeline_checkpoint_tracks_pipeline_and_corpus(monkeypatch, tmp_path):
    _result, _expected, identities, _phases, _state, _run = _execute_score_pipeline(
        monkeypatch, tmp_path
    )
    path, identity = identities[0]

    assert path == tmp_path / "checkpoint-swapped.json"
    assert identity.pipeline == "swapped"
    assert identity.corpus_digest == "digest"
    assert identity.commit == "commit"


def test_score_pipeline_passes_similarity_and_execution_options(monkeypatch, tmp_path):
    _result, _expected, identities, phases, _state, run = _execute_score_pipeline(
        monkeypatch, tmp_path
    )
    identity = identities[0][1]

    assert identity.min_similarity == run.arguments.min_similarity
    assert identity.limit == run.arguments.limit
    assert phases[0][4] == runner.PhaseSettings(
        device="cpu", min_similarity=0.4, chunk_size=3
    )
    assert phases[1][4] == phases[0][4]


def _corpus_report_run(tmp_path):
    """Make a representative run and one result for corpus reports."""
    document = Document("doc", "Zurich", ())
    loaded = LoadedCorpus("fixture", "en", [document], "digest")
    arguments = argparse.Namespace(min_similarity=0.25)
    run = benchmark_cli._CorpusRun(
        loaded, arguments, tmp_path, "cpu", "commit", {"host": "fixture"}
    )
    result = PipelineResult(
        name="swapped",
        models={"recognizer": "model@revision"},
        recognition={"f1": 0.75},
        resolution={"auc": 0.2},
        elapsed_seconds=2.5,
        device="cpu",
    )
    return run, result


def test_corpus_report_preserves_environment_and_pipeline_metrics(tmp_path):
    run, result = _corpus_report_run(tmp_path)
    payload = benchmark_cli._corpus_report_payload(run, 3, [result])

    assert payload["environment"] == {"host": "fixture"}
    assert payload["pipelines"] == [
        {
            "name": "swapped",
            "device": "cpu",
            "models": {"recognizer": "model@revision"},
            "recognition": {"f1": 0.75},
            "resolution": {"auc": 0.2},
            "elapsed_seconds": 2.5,
        }
    ]


def test_corpus_summary_row_records_counts_and_scores(tmp_path):
    run, result = _corpus_report_run(tmp_path)
    rows = benchmark_cli._summary_rows(run, 3, [result])

    assert rows == [
        {
            "corpus": "fixture",
            "language": "en",
            "documents": 1,
            "gold_toponyms": 3,
            "pipeline": "swapped",
            "recognition": {"f1": 0.75},
            "resolution": {"auc": 0.2},
            "elapsed_seconds": 2.5,
        }
    ]


def test_corpus_report_writer_emits_markdown_and_json(tmp_path):
    run, result = _corpus_report_run(tmp_path)
    payload = benchmark_cli._corpus_report_payload(run, 3, [result])
    benchmark_cli._write_corpus_reports(tmp_path, "# fixture report", payload)

    assert (tmp_path / "benchmark-report.md").read_text() == "# fixture report"
    assert '"commit": "commit"' in (tmp_path / "benchmark-report.json").read_text()


def _run_default_corpus_main(monkeypatch, tmp_path, capsys):
    """Run the default corpus path with loading and model scoring stubbed."""
    gold = GoldSpan(0, 4, "city", 47.0, 8.0)
    second_gold = GoldSpan(5, 9, "town", 47.1, 8.1)
    loaded = LoadedCorpus(
        "geovirus", "en", [Document("doc", "city town", (gold, second_gold))], "digest"
    )
    result = PipelineResult(name="swapped", device="cpu")
    load_calls = []

    monkeypatch.setattr(benchmark_cli.pipelines, "resolve_device", lambda value: value)
    monkeypatch.setattr(benchmark_cli.pipelines, "describe_device", lambda value: value)
    monkeypatch.setattr(
        benchmark_cli.provenance, "source_commit", lambda root: "commit"
    )
    monkeypatch.setattr(
        benchmark_cli.provenance,
        "environment",
        lambda job_id: {"job_id": job_id},
    )

    def load(name, folder, *, limit):
        load_calls.append((name, folder, limit))
        return loaded

    monkeypatch.setattr(benchmark_cli.corpora, "load", load)
    monkeypatch.setattr(
        benchmark_cli, "_score_requested_pipelines", lambda run: [result]
    )
    monkeypatch.delenv("GEOPARSER_DB_PATH", raising=False)
    monkeypatch.setenv("OAR_JOB_ID", "job-17")

    exit_code = benchmark_cli.main(
        ["--output-dir", str(tmp_path), "--limit", "9", "--device", "cpu"]
    )

    corpus_dir = tmp_path / "geovirus"
    output = capsys.readouterr().out
    return exit_code, load_calls, corpus_dir, output


def test_main_loads_default_corpus_and_writes_report_files(
    monkeypatch, tmp_path, capsys
):
    exit_code, load_calls, corpus_dir, _output = _run_default_corpus_main(
        monkeypatch, tmp_path, capsys
    )

    assert exit_code == 0
    assert load_calls == [("geovirus", corpus_dir, 9)]
    assert (
        corpus_dir.is_dir(),
        (corpus_dir / "benchmark-report.md").is_file(),
        (tmp_path / "summary.json").is_file(),
    ) == (True, True, True)


def test_main_reports_provenance_and_corpus_summary(monkeypatch, tmp_path, capsys):
    _exit_code, _load_calls, _corpus_dir, output = _run_default_corpus_main(
        monkeypatch, tmp_path, capsys
    )

    assert '"job_id": "job-17"' in (tmp_path / "summary.json").read_text()
    assert "Gold toponyms: 2" in output


def test_main_expands_all_corpora_once_and_preserves_database_override(
    monkeypatch, tmp_path
):
    gold = GoldSpan(0, 4, "city", 47.0, 8.0)
    loaded = LoadedCorpus("fixture", "en", [Document("doc", "city", (gold,))], "digest")
    calls = []
    monkeypatch.setattr(benchmark_cli.pipelines, "resolve_device", lambda value: value)
    monkeypatch.setattr(benchmark_cli.pipelines, "describe_device", lambda value: value)
    monkeypatch.setattr(
        benchmark_cli.provenance, "source_commit", lambda root: "commit"
    )
    monkeypatch.setattr(benchmark_cli.provenance, "environment", lambda job: {})
    monkeypatch.setattr(
        benchmark_cli.corpora, "CORPORA", {"alpha": object(), "beta": object()}
    )
    monkeypatch.setattr(
        benchmark_cli.corpora,
        "load",
        lambda name, folder, *, limit: calls.append(name) or loaded,
    )
    monkeypatch.setattr(benchmark_cli, "_score_requested_pipelines", lambda run: [])
    monkeypatch.setenv("GEOPARSER_DB_PATH", "existing.sqlite")

    assert benchmark_cli.main(["--output-dir", str(tmp_path), "--corpus", "all"]) == 0
    assert calls == ["alpha", "beta"]
    assert os.environ["GEOPARSER_DB_PATH"] == "existing.sqlite"


def _malformed_corpus() -> LoadedCorpus:
    """A corpus whose one gold span has a latitude no place on Earth has."""
    bad = GoldSpan(0, 4, "city", 123.0, 8.0)
    return LoadedCorpus(
        "geovirus", "en", [Document("doc", "city town", (bad,))], "digest"
    )


class TestCorpusChecksGate:
    """A corpus the offline checks reject is never scored."""

    def test_clean_corpus_passes_the_gate(self):
        """Test that a corpus without problems is not rejected."""
        clean = LoadedCorpus(
            "fixture",
            "en",
            [Document("doc", "city", (GoldSpan(0, 4, "city", 47.0, 8.0),))],
            "digest",
        )

        benchmark_cli.require_clean_corpus(clean)

    def test_malformed_corpus_is_rejected_naming_the_corpus_and_problem(self):
        """Test that the rejection says which corpus failed and why."""
        with pytest.raises(benchmark_cli.CorpusRejectedError) as raised:
            benchmark_cli.require_clean_corpus(_malformed_corpus())

        message = str(raised.value)
        assert message.startswith("geovirus rejected by corpus checks")
        assert "coordinate in doc" in message

    def test_malformed_corpus_stops_the_run_before_any_model_is_called(
        self, monkeypatch, tmp_path, capsys
    ):
        """Test that a rejected corpus fails the run and no phase builds a model."""
        phase_calls = []
        monkeypatch.setattr(benchmark_cli.pipelines, "resolve_device", lambda v: v)
        monkeypatch.setattr(benchmark_cli.pipelines, "describe_device", lambda v: v)
        monkeypatch.setattr(
            benchmark_cli.provenance, "source_commit", lambda root: "commit"
        )
        monkeypatch.setattr(benchmark_cli.provenance, "environment", lambda job: {})
        monkeypatch.setattr(
            benchmark_cli.corpora,
            "load",
            lambda name, folder, *, limit: _malformed_corpus(),
        )
        monkeypatch.setattr(
            benchmark_cli.runner,
            "run_phase",
            lambda *args, **kwargs: phase_calls.append(args) or {},
        )
        monkeypatch.delenv("GEOPARSER_DB_PATH", raising=False)

        exit_code = benchmark_cli.main(
            ["--output-dir", str(tmp_path), "--device", "cpu"]
        )

        assert exit_code == 1
        assert phase_calls == []
        assert not (tmp_path / "geovirus" / "benchmark-report.md").exists()
        assert not (tmp_path / "summary.json").exists()
        assert "rejected by corpus checks, no model was run" in capsys.readouterr().err


def test_a_corpus_with_no_documents_is_rejected_before_any_model_runs():
    """An empty load is never written up as a successful benchmark."""
    with pytest.raises(benchmark_cli.CorpusRejectedError) as raised:
        benchmark_cli.require_clean_corpus(LoadedCorpus("fixture", "en", [], "digest"))

    assert "empty_corpus in corpus: no documents" in str(raised.value)


def test_a_corpus_with_no_gold_spans_is_rejected_before_any_model_runs():
    """Documents with no gold give nothing to score, so the run is refused."""
    no_gold = LoadedCorpus("fixture", "en", [Document("doc", "city", ())], "digest")

    with pytest.raises(benchmark_cli.CorpusRejectedError) as raised:
        benchmark_cli.require_clean_corpus(no_gold)

    assert "empty_corpus in corpus: no gold spans" in str(raised.value)


GEOVIRUS_ARTICLES = 229
GEOVIRUS_GOLD_SPANS = 2167


def corpus_gold(loaded: LoadedCorpus) -> int:
    """The number of gold spans a loaded corpus holds."""
    return sum(len(document.gold) for document in loaded.documents)


def _geovirus_article(index: int) -> str:
    """One well-formed article; the first 106 hold ten toponyms, the rest nine."""
    names = [f"Place{number}" for number in range(10 if index < 106 else 9)]
    locations = []
    start = 0
    for name in names:
        # The corpus records each offset one character further along, so the
        # written start is one more than the true position of the name.
        locations.append(
            f"<location><name>{name}</name><start>{start + 1}</start>"
            f"<end>{start + 1 + len(name)}</end>"
            "<lat>10.0</lat><lon>20.0</lon></location>"
        )
        start += len(name) + 1
    return (
        f"<article><text>{' '.join(names)}</text>"
        f"<locations>{''.join(locations)}</locations></article>"
    )


def _write_geovirus(folder, articles: int) -> None:
    """Write a GeoVirus file holding the first `articles` articles of the source."""
    folder.mkdir(parents=True, exist_ok=True)
    body = "".join(_geovirus_article(index) for index in range(articles))
    (folder / "GeoVirus.xml").write_text(
        f"<articles>{body}</articles>", encoding="utf-8"
    )


class TestSourceTotalsGate:
    """A GeoVirus file that is not the whole source is refused, not benchmarked."""

    def test_the_complete_source_is_compared_and_accepted(self, tmp_path, capsys):
        """229 articles and 2167 toponyms match the source, so the run goes ahead."""
        _write_geovirus(tmp_path, GEOVIRUS_ARTICLES)
        loaded = corpora.load("geovirus", tmp_path)

        benchmark_cli.require_clean_corpus(loaded)

        assert (len(loaded.documents), corpus_gold(loaded)) == (
            GEOVIRUS_ARTICLES,
            GEOVIRUS_GOLD_SPANS,
        )
        assert (
            "compared with source totals of 229 documents, 2167 gold spans"
            in capsys.readouterr().out
        )

    def test_a_capped_run_is_compared_on_documents_and_says_gold_is_not(
        self, tmp_path, capsys
    ):
        """With --limit the document cap is checked, and the skipped gold total is named."""
        _write_geovirus(tmp_path, GEOVIRUS_ARTICLES)
        loaded = corpora.load("geovirus", tmp_path, limit=9)

        benchmark_cli.require_clean_corpus(loaded)

        assert len(loaded.documents) == 9
        assert (
            "compared with source totals of 9 documents; not compared: gold spans"
            in capsys.readouterr().out
        )

    @pytest.mark.parametrize("articles", [228, 0])
    def test_a_truncated_or_empty_source_stops_the_run_before_any_model(
        self, monkeypatch, tmp_path, capsys, articles
    ):
        """A well-formed file with 228 articles, or none, is refused end to end."""
        phase_calls = []
        monkeypatch.setattr(benchmark_cli.pipelines, "resolve_device", lambda v: v)
        monkeypatch.setattr(benchmark_cli.pipelines, "describe_device", lambda v: v)
        monkeypatch.setattr(
            benchmark_cli.provenance, "source_commit", lambda root: "commit"
        )
        monkeypatch.setattr(benchmark_cli.provenance, "environment", lambda job: {})
        monkeypatch.setattr(
            benchmark_cli.runner,
            "run_phase",
            lambda *args, **kwargs: phase_calls.append(args) or {},
        )
        monkeypatch.delenv("GEOPARSER_DB_PATH", raising=False)
        _write_geovirus(tmp_path / "geovirus", articles)

        exit_code = benchmark_cli.main(
            ["--output-dir", str(tmp_path), "--device", "cpu"]
        )

        assert exit_code == 1
        assert phase_calls == []
        assert not (tmp_path / "geovirus" / "benchmark-report.md").exists()
        assert "source_count" in capsys.readouterr().err
