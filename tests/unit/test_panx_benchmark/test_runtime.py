import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.panx_benchmark import __main__ as cli
from scripts.panx_benchmark import models, runner
from scripts.panx_benchmark.constants import MODELS
from scripts.panx_benchmark.data import Example, LoadedDataset


def _dataset(limit_per_language=1):
    return LoadedDataset(
        examples_by_language={},
        source_counts={"en": 3, "fr": 4},
        load_seconds=0.25,
        limit_per_language=limit_per_language,
    )


def test_snapshot_records_cache_state_and_exact_revision(monkeypatch, tmp_path):
    spec = MODELS[1]
    snapshot = (
        tmp_path
        / "hub"
        / f"models--{spec.model_id.replace('/', '--')}"
        / "snapshots"
        / spec.revision
    )
    snapshot.mkdir(parents=True)
    calls = []

    def download(**arguments):
        calls.append(arguments)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )

    path, _, cache_hit = models._snapshot(spec, tmp_path)

    assert (path, cache_hit, calls[0]["revision"]) == (
        str(snapshot),
        True,
        spec.revision,
    )


def test_snapshot_rejects_a_model_without_a_revision(tmp_path):
    spec = replace(MODELS[1], revision=None)

    with pytest.raises(ValueError, match="missing a pinned model revision"):
        models._snapshot(spec, tmp_path)


def test_spacy_loader_checks_the_pin_and_removes_unused_components(
    monkeypatch,
):
    class Pipeline:
        def __init__(self):
            self.pipe_names = [
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ]
            self.removed = []

        def remove_pipe(self, component):
            self.removed.append(component)

    pipeline = Pipeline()
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda model_id: pipeline)
    )
    monkeypatch.setattr(models.importlib.metadata, "version", lambda _: "3.8.0")

    loaded = models._load_spacy(MODELS[0])

    assert cast(models.SpacyRecognizer, loaded.predictor).pipeline is pipeline
    assert pipeline.removed == ["tagger", "parser", "attribute_ruler", "lemmatizer"]


def test_spacy_loader_rejects_an_unpinned_distribution(monkeypatch):
    monkeypatch.setattr(models.importlib.metadata, "version", lambda _: "3.7.0")

    with pytest.raises(RuntimeError, match=r"Expected en_core_web_sm 3.8.0"):
        models._load_spacy(MODELS[0])


def test_model_loader_dispatches_each_supported_key_and_rejects_unknown(
    monkeypatch, tmp_path
):
    expected = [object(), object(), object()]
    monkeypatch.setattr(models, "_load_spacy", lambda _: expected[0])
    monkeypatch.setattr(models, "_load_gliner", lambda *_: expected[1])
    monkeypatch.setattr(models, "_load_xlmr", lambda *_: expected[2])

    loaded = [models.load_model(spec, tmp_path) for spec in MODELS]
    unknown = replace(MODELS[0], key="unknown")

    assert loaded == expected
    with pytest.raises(ValueError, match="Unknown PAN-X recognizer"):
        models.load_model(unknown, tmp_path)


def test_cpu_quota_parser_handles_cgroup_values(monkeypatch):
    current = {"value": "250000 100000"}
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path, **_kwargs: current["value"],
    )

    quota = runner._cgroup_cpu_count()
    current["value"] = "max 100000"
    unlimited = runner._cgroup_cpu_count()
    current["value"] = "broken"
    invalid = runner._cgroup_cpu_count()

    assert (quota, unlimited, invalid) == (2.5, None, None)


def test_memory_parser_reads_total_memory_and_handles_missing_value(monkeypatch):
    current = {"value": "MemTotal: 1024 kB\nMemFree: 512 kB\n"}
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path, **_kwargs: current["value"],
    )

    total = runner._memory_bytes()
    current["value"] = "MemFree: 512 kB\n"
    absent = runner._memory_bytes()

    assert (total, absent) == (1024 * 1024, None)


def test_cpu_configuration_seeds_runtimes_and_uses_cgroup_quota(
    monkeypatch,
):
    recorded = []
    fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=recorded.append))
    fake_torch = SimpleNamespace(
        manual_seed=recorded.append,
        set_num_threads=recorded.append,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_cgroup_cpu_count", lambda: 2.9)

    threads = runner.configure_cpu(seed=7)

    assert (threads, recorded) == (2, [7, 7, 2])


def test_hardware_facts_records_cpu_and_accelerator_state(monkeypatch):
    cuda = {"available": False}
    fake_torch = SimpleNamespace(
        __version__="2.8.0",
        cuda=SimpleNamespace(
            is_available=lambda: cuda["available"],
            get_device_name=lambda index: f"GPU-{index}",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_cgroup_cpu_count", lambda: 2.0)
    monkeypatch.setattr(runner, "_memory_bytes", lambda: 1024)

    cpu = runner.hardware_facts(2)
    cuda["available"] = True
    gpu = runner.hardware_facts(2)

    assert (cpu["device_used"], cpu["gpu"], gpu["gpu"]) == ("cpu", None, "GPU-0")
    assert (
        cpu["torch_threads"],
        cpu["cgroup_cpu_quota"],
        cpu["memory_total_bytes"],
    ) == (
        2,
        2.0,
        1024,
    )


def test_commit_id_marks_dirty_state_and_handles_git_failure(monkeypatch):
    import subprocess

    outputs = iter(["abc123\n", "M file.py\n"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=next(outputs)),
    )
    dirty = runner._commit_id()

    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(subprocess, "run", fail)

    assert dirty == "abc123-dirty"
    assert runner._commit_id() == "unknown"


def test_run_benchmark_keeps_explicit_commit_in_identity_and_report(
    monkeypatch, tmp_path
):
    _stub_benchmark_dependencies(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_commit_id",
        lambda: pytest.fail("an explicit repository commit should bypass probing"),
    )
    checkpoint_calls = []

    def checkpoints(directory, identity):
        checkpoint_calls.append((directory, identity))
        return SimpleNamespace(snapshot_id="snapshot", directory=directory)

    monkeypatch.setattr(runner, "ModelLanguageCheckpoints", checkpoints)

    result = runner.run_benchmark(
        _dataset(),
        cache_dir=tmp_path / "cache",
        thread_count=1,
        options=runner.BenchmarkRunOptions(models_to_run=()),
        checkpoint_dir=tmp_path / "checkpoints",
        repository_commit="explicit-commit",
    )

    assert result["repository_commit"] == "explicit-commit"
    assert checkpoint_calls[0][0] == tmp_path / "checkpoints"
    assert checkpoint_calls[0][1]["repository_commit"] == "explicit-commit"


@pytest.mark.parametrize("commit", ["unknown", "abc123-dirty"])
def test_cli_rejects_unidentifiable_commit_before_side_effects(
    monkeypatch, tmp_path, commit
):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        ["panx", "--cache-dir", str(cache_dir)],
    )
    monkeypatch.setattr(cli, "_commit_id", lambda: commit)
    monkeypatch.setattr(
        cli,
        "configure_cpu",
        lambda: pytest.fail("CPU configuration must follow commit validation"),
    )
    monkeypatch.setattr(
        cli,
        "load_test_examples",
        lambda *_args, **_kwargs: pytest.fail("data loading must follow validation"),
    )

    with pytest.raises(ValueError, match="requires a clean, identifiable"):
        cli.main()

    assert not cache_dir.exists()


def _stub_benchmark_dependencies(monkeypatch):
    loaded = object()
    monkeypatch.setattr(runner, "load_model", lambda *_: loaded)
    monkeypatch.setattr(
        runner,
        "evaluate_model",
        lambda *_args, **_kwargs: {"full_matrix_estimated_inference_seconds": 4.0},
    )
    monkeypatch.setattr(runner, "_commit_id", lambda: "abc123")
    monkeypatch.setattr(runner, "hardware_facts", lambda threads: {"threads": threads})
    monkeypatch.setattr(
        runner,
        "split_manifest",
        lambda: {
            "eligible_languages": ["en", "fr"],
            "missing_target_languages": ["ha", "xh", "zu"],
        },
    )


def test_bounded_benchmark_record_disclaims_quality_comparison(monkeypatch, tmp_path):
    _stub_benchmark_dependencies(monkeypatch)
    bounded = runner.run_benchmark(
        _dataset(),
        cache_dir=tmp_path,
        thread_count=2,
        options=runner.BenchmarkRunOptions(models_to_run=MODELS[:1]),
    )

    assert (bounded["evaluation_kind"], bounded["full_quality_comparison"]) == (
        "bounded_feasibility_sample",
        False,
    )
    assert (
        "Small feasibility samples are not a quality result"
        in bounded["full_matrix_estimate_note"]
    )


def test_run_benchmark_requires_spacy_for_transfer_mode(tmp_path):
    options = runner.BenchmarkRunOptions(
        models_to_run=MODELS[1:], spacy_cross_lingual_transfer=True
    )

    with pytest.raises(ValueError, match="requires the spacy_en model"):
        runner.run_benchmark(
            _dataset(), cache_dir=tmp_path, thread_count=2, options=options
        )


def test_full_benchmark_record_marks_test_split_comparison(monkeypatch, tmp_path):
    _stub_benchmark_dependencies(monkeypatch)
    complete = runner.run_benchmark(
        _dataset(None),
        cache_dir=tmp_path,
        thread_count=2,
        options=runner.BenchmarkRunOptions(models_to_run=()),
    )

    assert (complete["evaluation_kind"], complete["full_quality_comparison"]) == (
        "full_test_split",
        True,
    )
    assert complete["full_matrix_estimate_note"] == (
        "Measured inference over the complete pinned test intersection."
    )
    assert complete["dataset"]["missing_target_languages"] == ["ha", "xh", "zu"]


def test_checkpoint_identity_pins_code_data_models_and_cpu_configuration():
    dataset = replace(
        _dataset(None),
        examples_by_language={"en": (Example("en", "Paris", frozenset({(0, 5)})),)},
    )
    identity = runner._checkpoint_identity(
        dataset,
        MODELS,
        "abc123",
        {"torch_threads": 4, "device_used": "cpu"},
    )
    changed_examples = dict(dataset.examples_by_language)
    changed_examples["en"] = (replace(changed_examples["en"][0], text="London"),)
    changed_identity = runner._checkpoint_identity(
        replace(dataset, examples_by_language=changed_examples),
        MODELS,
        "abc123",
        {"torch_threads": 4, "device_used": "cpu"},
    )
    changed_group_identity = runner._checkpoint_identity(
        dataset,
        [MODELS[0], replace(MODELS[1], batch_size=2), MODELS[2]],
        "abc123",
        {"torch_threads": 4, "device_used": "cpu"},
    )

    assert (
        identity["repository_commit"],
        identity["dataset"]["revision"],
        identity["dataset"]["split"],
        identity["dataset"]["missing_target_languages"],
        identity["dataset"]["source_test_examples_by_language"],
        [model["revision"] for model in identity["models"]],
        identity["evaluation"]["device"],
        identity["hardware"]["torch_threads"],
        set(identity["input_manifests"]),
        identity["dataset"]["held_out_examples_by_language"]["en"]["example_count"],
        identity["dataset"]["held_out_examples_by_language"]["en"]["sha256"]
        != changed_identity["dataset"]["held_out_examples_by_language"]["en"]["sha256"],
        identity["evaluation"]["batch_sizes_by_model"],
        identity != changed_group_identity,
    ) == (
        "abc123",
        "f0a3be6dc5564c0cc4150bb660144800a1f539d4",
        "test",
        ["ha", "xh", "zu"],
        {"en": 3, "fr": 4},
        [model.revision for model in MODELS],
        "cpu",
        4,
        {"target_languages_sha256", "test_split_sha256"},
        1,
        True,
        {"spacy_en": 8, "gliner2_multi": 1, "xlmr_ner_hrl": 8},
        True,
    )


def test_cli_main_runs_with_a_bounded_sample_and_writes_reports(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    checkpoint_dir = tmp_path / "checkpoints"
    report_dir = tmp_path / "report"
    dataset = _dataset(2)
    result = {"evaluation_kind": "bounded_feasibility_sample"}
    calls = []
    benchmark_calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "panx",
            "--limit-per-language",
            "2",
            "--cache-dir",
            str(cache_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )
    monkeypatch.setattr(cli, "_default_output_dir", lambda: report_dir)
    monkeypatch.setattr(cli, "_commit_id", lambda: "abc123")
    monkeypatch.setattr(cli, "configure_cpu", lambda: 2)
    monkeypatch.setattr(cli, "load_test_examples", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda *_args, **kwargs: benchmark_calls.append(kwargs) or result,
    )
    monkeypatch.setattr(
        cli,
        "write_reports",
        lambda output, value: (
            calls.append((output, value))
            or (output / "report.json", output / "report.md")
        ),
    )

    status = cli.main()

    assert (
        status,
        calls[0][0],
        calls[0][1]["resources"]["cache_directory"],
        bool(calls[0][1]["language_list"]["language_codes"]),
        benchmark_calls[0]["checkpoint_dir"],
        benchmark_calls[0]["repository_commit"],
        benchmark_calls[0]["options"].models_to_run,
        benchmark_calls[0]["options"].spacy_cross_lingual_transfer,
    ) == (
        0,
        report_dir,
        str(cache_dir.resolve()),
        True,
        checkpoint_dir,
        "abc123",
        MODELS,
        False,
    )


def test_cli_can_select_only_spacy_and_enable_transfer_mode(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    checkpoint_dir = tmp_path / "checkpoints"
    report_dir = tmp_path / "report"
    dataset = _dataset(2)
    result = {"evaluation_kind": "bounded_feasibility_sample"}
    benchmark_calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "panx",
            "--model",
            "spacy_en",
            "--spacy-cross-lingual-transfer",
            "--limit-per-language",
            "2",
            "--cache-dir",
            str(cache_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output-dir",
            str(report_dir),
        ],
    )
    monkeypatch.setattr(cli, "_commit_id", lambda: "abc123")
    monkeypatch.setattr(cli, "configure_cpu", lambda: 2)
    monkeypatch.setattr(cli, "load_test_examples", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda *_args, **kwargs: benchmark_calls.append(kwargs) or result,
    )
    monkeypatch.setattr(
        cli,
        "write_reports",
        lambda output, _value: (output / "report.json", output / "report.md"),
    )

    status = cli.main()

    assert status == 0
    assert benchmark_calls[0]["options"].models_to_run == (MODELS[0],)
    assert benchmark_calls[0]["options"].spacy_cross_lingual_transfer is True


def test_cli_rejects_a_nonpositive_sample_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["panx", "--limit-per-language", "0", "--cache-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_cli_rejects_duplicate_model_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "panx",
            "--model",
            "spacy_en",
            "--model",
            "spacy_en",
            "--cache-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_cli_requires_spacy_for_transfer_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "panx",
            "--model",
            MODELS[1].key,
            "--spacy-cross-lingual-transfer",
            "--cache-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()
