"""
Tests for the benchmark command line.

Several corpora share one output directory, so each gets its own folder:
checkpoints of two corpora must never be mistaken for one another.
"""

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark import __main__ as benchmark_cli
from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import corpora, pipelines, runner
from scripts.benchmark.__main__ import _ManifestPlan, build_parser, corpus_output_dir
from scripts.benchmark.corpora import LoadedCorpus
from scripts.benchmark.corpus import Document, GoldSpan
from scripts.benchmark.report import PipelineResult


@pytest.fixture(autouse=True)
def _accept_miniature_newsli_release(monkeypatch):
    """These tests write miniature releases; the pin itself is tested in test_corpora."""
    monkeypatch.setattr(corpora, "_check_newsli_release", lambda data, path: None)
    monkeypatch.setattr(
        benchmark_cli.provenance,
        "gazetteer_artifact_sha256",
        lambda name: "test-gazetteer-sha256",
    )


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
    state = ckpt.Checkpoint(benchmark_cli._run_identity(run, "swapped"))
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
    arguments = argparse.Namespace(min_similarity=0.25, limit=None)
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
    pipeline = payload["pipelines"][0]
    assert {
        "environment": payload["environment"],
        "name": pipeline["name"],
        "device": pipeline["device"],
        "models": pipeline["models"],
        "recognition": pipeline["recognition"],
        "resolution": pipeline["resolution"],
        "elapsed_seconds": pipeline["elapsed_seconds"],
        "phase_elapsed_seconds": pipeline["phase_elapsed_seconds"],
        "phase_provenance": pipeline["phase_provenance"],
        "failures": pipeline["failures"],
        "reranker_license": pipeline["model_metadata"]["reranker"]["license"],
    } == {
        "environment": {"host": "fixture"},
        "name": "swapped",
        "device": "cpu",
        "models": {"recognizer": "model@revision"},
        "recognition": {"f1": 0.75},
        "resolution": {"auc": 0.2},
        "elapsed_seconds": 2.5,
        "phase_elapsed_seconds": {},
        "phase_provenance": {},
        "failures": {},
        "reranker_license": "CC-BY-NC-4.0",
    }


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
            "phase_elapsed_seconds": {},
            "failures": {},
        }
    ]


def test_run_manifest_pins_matrix_counts_models_and_language_evidence():
    arguments = argparse.Namespace(
        min_similarity=0.0,
        limit=None,
        chunk_size=5,
        device="cuda",
    )

    manifest = benchmark_cli._run_manifest(
        _ManifestPlan(
            names=["newsli-ar"],
            pipeline_names=[pipelines.XLM_MINILM],
            phases=[runner.RECOGNITION, runner.RESOLUTION],
            arguments=arguments,
            commit="short-sha",
            full_commit="a" * 40,
            facts={"started_at": "2026-10-13T00:00:00+00:00", "gpu": "A100"},
            device="cuda",
        )
    )

    cell = manifest["protocol"]["corpora"]["newsli-ar"]
    assert (
        cell["expected_documents"],
        cell["expected_gold_toponyms"],
        "sorted order" in cell["source_split"],
        manifest["full_commit"],
        manifest["protocol"]["pipelines"][pipelines.XLM_MINILM]["model_pins"][
            "resolver"
        ].endswith("@d678769f195c194ffdc6f3735a9360118a855b43"),
    ) == (500, 3319, True, "a" * 40, True)

    resumed = json.loads(json.dumps(manifest))
    resumed["protocol"]["corpora"]["newsli-ar"].update(
        {"status": "complete", "corpus_digest": "digest", "documents": 500}
    )
    assert benchmark_cli._manifest_identity(
        resumed
    ) == benchmark_cli._manifest_identity(manifest)


def test_run_manifest_marks_explicit_document_failures():
    manifest: dict[str, Any] = {
        "protocol": {
            "corpora": {
                "fixture": {"language": "en"},
            },
            "pipelines": {
                "otter-cross-minilm": {"model_pins": {"recognizer": "otter@pin"}},
            },
        }
    }
    payload = {
        "corpus_digest": "digest",
        "documents": 2,
        "gold_toponyms": 1,
        "environment": {"hostname": "fixture"},
        "pipelines": [
            {
                "name": "otter-cross-minilm",
                "recognition": {"f1": 0.0},
                "resolution": {"auc": 0.2},
                "phase_elapsed_seconds": {"recognition": 2.0, "resolution": 3.0},
                "phase_provenance": {},
                "failures": {
                    "recognition": {
                        "long-doc": {"kind": "unsupported_input", "reason": "limit"}
                    }
                },
            }
        ],
    }

    benchmark_cli._record_manifest_cell(
        manifest,
        "fixture",
        payload,
        [runner.RECOGNITION, runner.RESOLUTION],
    )

    fixture_pipelines: Any = manifest["protocol"]["corpora"]["fixture"]["pipelines"]
    result = fixture_pipelines["otter-cross-minilm"]
    assert manifest["protocol"]["corpora"]["fixture"]["status"] == (
        "complete_with_document_failures"
    )
    assert result["failure_count"] == 1
    assert result["failures"]["recognition"]["long-doc"]["kind"] == (
        "unsupported_input"
    )


def test_corpus_report_writer_emits_markdown_and_json(tmp_path):
    run, result = _corpus_report_run(tmp_path)
    payload = benchmark_cli._corpus_report_payload(run, 3, [result])
    benchmark_cli._write_corpus_reports(tmp_path, "# fixture report", payload)

    assert (tmp_path / "benchmark-report.md").read_text() == "# fixture report"
    assert '"commit": "commit"' in (tmp_path / "benchmark-report.json").read_text()


def _run_default_corpus_main(monkeypatch, tmp_path, capsys, *, corpus_cache=None):
    """Run the default corpus path with loading and model scoring stubbed."""
    gold = GoldSpan(0, 4, "city", 47.0, 8.0)
    second_gold = GoldSpan(5, 9, "town", 47.1, 8.1)
    loaded = LoadedCorpus(
        "geovirus", "en", [Document("doc", "city town", (gold, second_gold))], "digest"
    )
    result = PipelineResult(name="swapped", device="cpu")
    load_calls = []
    monkeypatch.setattr(benchmark_cli.missing_matrix, "MATRIX", ())

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
    monkeypatch.delenv("GEOPARSER_CORPUS_CACHE_DIR", raising=False)
    if corpus_cache is not None:
        monkeypatch.setenv("GEOPARSER_CORPUS_CACHE_DIR", str(corpus_cache))
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
    manifest = json.loads((tmp_path / "run-manifest.json").read_text())
    assert manifest["protocol"]["gazetteer_artifact_sha256"] == (
        "test-gazetteer-sha256"
    )


def test_main_keeps_downloaded_corpora_in_the_configured_scratch_cache(
    monkeypatch, tmp_path, capsys
):
    corpus_cache = tmp_path / "node-scratch" / "corpora"

    exit_code, load_calls, _corpus_dir, _output = _run_default_corpus_main(
        monkeypatch, tmp_path / "durable-results", capsys, corpus_cache=corpus_cache
    )

    assert exit_code == 0
    assert load_calls == [("geovirus", corpus_cache / "geovirus", 9)]


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
        benchmark_cli.corpora,
        "CORPORA",
        {
            "alpha": SimpleNamespace(language="en", documents=1, gold_spans=1),
            "beta": SimpleNamespace(language="en", documents=1, gold_spans=1),
        },
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

    def test_changed_adapter_totals_do_not_disable_the_frozen_matrix_check(self):
        """A coordinated adapter/data change still has to match the frozen slice."""
        documents = [
            Document(
                identifier,
                "Paris",
                (GoldSpan(0, 5, "Paris", 48.0, 2.0),),
            )
            for identifier in ("doc-1", "doc-2")
        ]
        loaded = LoadedCorpus(
            "geovirus",
            "en",
            documents,
            "digest",
            expected_documents=2,
            expected_gold=2,
        )

        with pytest.raises(benchmark_cli.CorpusRejectedError, match="frozen matrix"):
            benchmark_cli._check_known_frozen_totals(
                loaded, SimpleNamespace(documents=1, gold_toponyms=1)
            )

    def test_full_slice_limit_does_not_bypass_frozen_matrix_totals(self):
        """A limit equal to a full slice still checks the frozen gold totals."""
        loaded = LoadedCorpus(
            "geovirus",
            "en",
            [Document("doc", "Paris", (GoldSpan(0, 5, "Paris", 48.0, 2.0),))],
            "digest",
            limit=229,
        )

        with pytest.raises(benchmark_cli.CorpusRejectedError, match="frozen matrix"):
            benchmark_cli._check_frozen_corpus_totals(loaded)

    @pytest.mark.parametrize(
        ("name", "limit", "expected"),
        [
            ("geovirus", None, False),
            ("geovirus", 5, True),
            ("geovirus", 229, False),
            ("newsli-fa", 71, False),
            ("unregistered-fixture", 1, True),
        ],
    )
    def test_limit_is_partial_uses_the_registered_full_slice(
        self, name, limit, expected
    ):
        """Explicit limits are partial only when they truncate the known slice."""
        assert benchmark_cli._limit_is_partial(name, limit) is expected

    def test_unknown_registered_size_is_partial_for_an_explicit_limit(
        self, monkeypatch
    ):
        """A registered corpus without a count cannot certify a full limited run."""
        name = "unfrozen-fixture"
        monkeypatch.setitem(
            corpora.CORPORA,
            name,
            corpora.CorpusSpec(name, "en", "https://example.test/data", "fixture"),
        )

        assert benchmark_cli._limit_is_partial(name, 1) is True

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
        _stub_run_environment(monkeypatch)
        _stub_corpus_load(monkeypatch, _malformed_corpus())
        phase_calls = _record_phase_calls(monkeypatch)

        exit_code = benchmark_cli.main(
            ["--output-dir", str(tmp_path), "--device", "cpu"]
        )

        assert (exit_code, phase_calls) == (1, [])
        assert not (tmp_path / "geovirus" / "benchmark-report.md").exists()
        assert not (tmp_path / "summary.json").exists()
        assert "rejected by corpus checks, no model was run" in capsys.readouterr().err

    def test_a_duplicated_document_identifier_stops_the_run_before_any_scoring(
        self, monkeypatch, tmp_path, capsys
    ):
        """Two documents with one identifier would overwrite each other's predictions."""
        first = Document("doc", "city", (GoldSpan(0, 4, "city", 47.0, 8.0),))
        second = Document("doc", "town", (GoldSpan(0, 4, "town", 47.1, 8.1),))
        _stub_run_environment(monkeypatch)
        _stub_corpus_load(
            monkeypatch, LoadedCorpus("fixture", "en", [first, second], "digest")
        )
        scored = []
        monkeypatch.setattr(
            benchmark_cli,
            "_score_requested_pipelines",
            lambda run: scored.append(run) or [],
        )

        exit_code = benchmark_cli.main(
            ["--output-dir", str(tmp_path), "--device", "cpu"]
        )

        assert (exit_code, scored) == (1, [])
        assert "duplicate_document in doc: identifier appears 2 times" in (
            capsys.readouterr().err
        )


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


def _stub_run_environment(monkeypatch) -> None:
    """Stub the device, provenance and database around a benchmark run."""
    monkeypatch.setattr(benchmark_cli.pipelines, "resolve_device", lambda v: v)
    monkeypatch.setattr(benchmark_cli.pipelines, "describe_device", lambda v: v)
    monkeypatch.setattr(
        benchmark_cli.provenance, "source_commit", lambda root: "commit"
    )
    monkeypatch.setattr(benchmark_cli.provenance, "environment", lambda job: {})
    monkeypatch.delenv("GEOPARSER_DB_PATH", raising=False)


def _stub_corpus_load(monkeypatch, loaded: LoadedCorpus) -> None:
    """Make the CLI load the given corpus whatever it is asked for."""
    monkeypatch.setattr(
        benchmark_cli.corpora, "load", lambda name, folder, *, limit: loaded
    )


def _record_phase_calls(monkeypatch) -> list:
    """Replace the phase runner with one that records each call and builds no model."""
    phase_calls: list = []
    monkeypatch.setattr(
        benchmark_cli.runner,
        "run_phase",
        lambda *args, **kwargs: phase_calls.append(args) or {},
    )
    return phase_calls


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
            "documents compared with 229; gold spans compared with 2167"
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
            "documents compared with 9; gold spans not compared"
            in capsys.readouterr().out
        )

    def test_a_capped_run_rejects_gold_lost_while_parsing_the_source(self, tmp_path):
        """A cap cannot hide a GeoVirus span dropped from the full source."""
        _write_geovirus(tmp_path, GEOVIRUS_ARTICLES)
        path = tmp_path / "GeoVirus.xml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "<lat>10.0</lat><lon>20.0</lon>", "", 1
            ),
            encoding="utf-8",
        )
        loaded = corpora.load("geovirus", tmp_path, limit=9)

        with pytest.raises(benchmark_cli.CorpusRejectedError) as raised:
            benchmark_cli.require_clean_corpus(loaded)

        assert "source publishes 2167" in str(raised.value)

    @pytest.mark.parametrize("articles", [228, 0])
    def test_a_truncated_or_empty_source_stops_the_run_before_any_model(
        self, monkeypatch, tmp_path, capsys, articles
    ):
        """A well-formed file with 228 articles, or none, is refused end to end."""
        _stub_run_environment(monkeypatch)
        phase_calls = _record_phase_calls(monkeypatch)
        _write_geovirus(tmp_path / "geovirus", articles)

        exit_code = benchmark_cli.main(
            ["--output-dir", str(tmp_path), "--device", "cpu"]
        )

        assert exit_code == 1
        assert phase_calls == []
        assert not (tmp_path / "geovirus" / "benchmark-report.md").exists()
        assert "source_count" in capsys.readouterr().err


def _newsli_articles(articles: int, gold_spans: int) -> dict:
    """Articles with unique names whose gold spans total `gold_spans`.

    The first `gold_spans - articles` hold two toponyms and the rest hold one.
    """
    two_span = gold_spans - articles
    documents = {}
    for index in range(articles):
        first, second = f"Alfa{index}", f"Beta{index}"
        if index < two_span:
            text = f"{first} {second}"
            spans = [
                (0, len(first), first, 45.0, 25.0),
                (len(first) + 1, len(text), second, 45.0, 25.0),
            ]
        else:
            text, spans = first, [(0, len(first), first, 45.0, 25.0)]
        documents[f"ro-{index:04d}"] = (text, spans)
    return documents


def _serve_newsli_release(monkeypatch, tmp_path, documents: dict) -> None:
    """Make the NewsLi loader read a release written from the given articles."""
    from tests.unit.test_benchmark.test_newsli import write_release

    release = write_release(tmp_path, language="ro", documents=documents)
    monkeypatch.setattr(
        corpora.corpus, "download_corpus", lambda cache_path, *, url: release
    )


class TestNewsliRecordedTotals:
    """An uncapped NewsLi load is compared with the counts its report recorded."""

    def test_an_uncapped_run_with_the_recorded_counts_passes_the_gate(
        self, monkeypatch, tmp_path, capsys
    ):
        """226 articles holding 385 gold toponyms are the recorded Romanian totals."""
        _serve_newsli_release(monkeypatch, tmp_path, _newsli_articles(226, 385))
        loaded = corpora.load("newsli-ro", tmp_path / "newsli-ro")

        benchmark_cli.require_clean_corpus(loaded)

        assert (len(loaded.documents), corpus_gold(loaded)) == (226, 385)
        assert (
            "documents compared with 226; gold spans compared with 385"
            in capsys.readouterr().out
        )

    def test_the_raw_text_file_count_is_not_the_expected_total(
        self, monkeypatch, tmp_path
    ):
        """241 text files hold 226 articles with gold; 241 is never the expected count."""
        documents = _newsli_articles(226, 385)
        documents.update({f"ro-x{index:04d}": ("Sin oro.", []) for index in range(15)})
        _serve_newsli_release(monkeypatch, tmp_path, documents)
        loaded = corpora.load("newsli-ro", tmp_path / "newsli-ro")

        benchmark_cli.require_clean_corpus(loaded)

        assert (loaded.expected_documents, len(loaded.documents)) == (226, 226)


HIPE_HEADER = "TOKEN\tNE-COARSE-LIT\tA\tB\tC\tD\tE\tNEL-LIT\tF\tMISC\n"


def _hipe_tsv(spans_per_document: list) -> str:
    """A HIPE file whose document i holds the given number of one-word places."""
    lines = [HIPE_HEADER]
    for index, spans in enumerate(spans_per_document):
        lines.append(f"# hipe2022:document_id = doc-{index:03d}\n")
        lines.extend("Paris\tB-loc\tO\tO\tO\tO\tO\tQ90\t_\t_\n" for _ in range(spans))
    return "".join(lines)


def _stage_hipe(monkeypatch, folder, spans_per_document: list) -> None:
    """Write a HIPE split into the corpus folder and resolve its one place offline."""
    folder.mkdir(parents=True, exist_ok=True)
    filename = corpora.CORPORA["hipe2020-de"].filename
    (folder / filename).write_text(_hipe_tsv(spans_per_document), encoding="utf-8")
    monkeypatch.setattr(
        corpora, "load_coordinates", lambda qids, cache, **_: {"Q90": (48.85, 2.35)}
    )


# 48 documents holding 558 places, as the checked-in report recorded for hipe2020-de.
RECORDED_HIPE_DE = [12] * 30 + [11] * 18


class TestHipeRecordedTotals:
    """An uncapped HIPE split is compared with the totals its report recorded."""

    def test_a_truncated_uncapped_hipe_split_is_refused_before_any_model(
        self, monkeypatch, tmp_path, capsys
    ):
        """47 of the 48 documents recorded for hipe2020-de are refused, not scored."""
        _stub_run_environment(monkeypatch)
        phase_calls = _record_phase_calls(monkeypatch)
        _stage_hipe(monkeypatch, tmp_path / "hipe2020-de", RECORDED_HIPE_DE[:-1])

        exit_code = benchmark_cli.main(
            [
                "--output-dir",
                str(tmp_path),
                "--corpus",
                "hipe2020-de",
                "--device",
                "cpu",
            ]
        )

        assert (exit_code, phase_calls) == (1, [])
        assert "source_count" in capsys.readouterr().err

    def test_the_recorded_counts_of_an_uncapped_hipe_split_pass_the_gate(
        self, monkeypatch, tmp_path, capsys
    ):
        """48 documents holding 558 places are the recorded totals, so they pass."""
        _stage_hipe(monkeypatch, tmp_path / "hipe2020-de", RECORDED_HIPE_DE)
        loaded = corpora.load("hipe2020-de", tmp_path / "hipe2020-de")

        benchmark_cli.require_clean_corpus(loaded)

        assert (len(loaded.documents), corpus_gold(loaded)) == (48, 558)
        assert (
            "documents compared with 48; gold spans compared with 558"
            in capsys.readouterr().out
        )

    def test_a_capped_hipe_run_rejects_gold_lost_while_parsing_the_source(
        self, monkeypatch, tmp_path
    ):
        """A cap cannot hide a linked HIPE span missing from the coordinate map."""
        folder = tmp_path / "hipe2020-de"
        _stage_hipe(monkeypatch, folder, RECORDED_HIPE_DE)
        path = folder / corpora.CORPORA["hipe2020-de"].filename
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\tQ90\t_\t_\n", "\tQ999\t_\t_\n", 1
            ),
            encoding="utf-8",
        )
        loaded = corpora.load("hipe2020-de", folder, limit=5)

        with pytest.raises(benchmark_cli.CorpusRejectedError) as raised:
            benchmark_cli.require_clean_corpus(loaded)

        assert "source publishes 558" in str(raised.value)
