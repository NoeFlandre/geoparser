"""Sharing gold-span resolution must preserve its full provenance."""

from argparse import Namespace
from unittest.mock import Mock, call

import pytest

from scripts.benchmark import __main__ as benchmark
from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import corpora, corpus, pipelines, report, runner


def _run(tmp_path, *, phases=None):
    documents = [
        corpus.Document("0", "first", ()),
        corpus.Document("1", "second", ()),
    ]
    loaded = corpora.LoadedCorpus("fixture", "en", documents, "same-corpus")
    arguments = Namespace(
        pipeline=[pipelines.XLM_MINILM, "otter-minilm"],
        phase=phases,
        min_similarity=0.0,
        limit=None,
    )
    return benchmark._CorpusRun(
        loaded,
        arguments,
        tmp_path,
        "cuda",
        "same-code",
        {"gazetteer_artifact_sha256": "same-gazetteer-content"},
    )


def test_only_one_minilm_resolution_phase_runs_for_both_recognizers(
    monkeypatch, tmp_path
):
    run = _run(tmp_path)
    xlm_result = report.PipelineResult(pipelines.XLM_MINILM)
    otter_result = report.PipelineResult("otter-minilm")
    score_pipeline = Mock(side_effect=[xlm_result, otter_result])
    reuse_resolution = Mock(return_value=otter_result)
    monkeypatch.setattr(benchmark, "_score_pipeline", score_pipeline)
    monkeypatch.setattr(benchmark, "_reuse_resolution", reuse_resolution)

    results = benchmark._score_requested_pipelines(run)

    assert results == [xlm_result, otter_result]
    assert score_pipeline.call_args_list == [
        call(run, pipelines.XLM_MINILM, [runner.RECOGNITION, runner.RESOLUTION]),
        call(run, "otter-minilm", [runner.RECOGNITION]),
    ]
    reuse_resolution.assert_called_once_with(
        run, "otter-minilm", pipelines.XLM_MINILM, otter_result
    )


def test_reused_resolution_predictions_are_pinned_and_labeled(tmp_path, monkeypatch):
    run = _run(tmp_path)
    run.output_dir.mkdir(exist_ok=True)
    source_identity = benchmark._run_identity(run, pipelines.XLM_MINILM)
    target_identity = benchmark._run_identity(run, "otter-minilm")
    source_state = ckpt.Checkpoint(source_identity)
    source_state.phase_elapsed_seconds[runner.RESOLUTION] = 12.5
    for document_id in ("0", "1"):
        source_state.record(
            runner.RESOLUTION,
            document_id,
            [{"start": 0, "end": 1, "latitude": 48.0, "longitude": 2.0}],
        )
    source_path = run.output_dir / f"checkpoint-{pipelines.XLM_MINILM}.json"
    ckpt.save(source_path, source_state)

    target_state = ckpt.Checkpoint(target_identity)
    target_state.elapsed_seconds = 11.0
    target_state.phase_elapsed_seconds = {
        runner.RECOGNITION: 7.5,
        runner.RESOLUTION: 3.5,
    }
    target_state.record(runner.RECOGNITION, "0", [{"start": 0, "end": 1}])
    target_state.record(
        runner.RESOLUTION,
        "0",
        [{"start": 0, "end": 1, "latitude": 0.0, "longitude": 0.0}],
    )
    target_path = run.output_dir / "checkpoint-otter-minilm.json"
    ckpt.save(target_path, target_state)

    scored = report.PipelineResult(
        name="otter-minilm",
        models={"recognizer": target_identity.model_pins["recognizer"]},
    )
    score = Mock(return_value=report.PipelineResult(name="otter-minilm"))
    monkeypatch.setattr(runner, "score", score)

    benchmark._reuse_resolution(run, "otter-minilm", pipelines.XLM_MINILM, scored)

    copied, reasons = ckpt.load(target_path, target_identity)
    assert {
        "reasons": reasons,
        "completed": copied.completed(runner.RESOLUTION),
        "phase_seconds": copied.phase_elapsed_seconds[runner.RESOLUTION],
        "elapsed_seconds": copied.elapsed_seconds,
        "provenance": copied.phase_provenance[runner.RESOLUTION],
        "scored_resolver": score.call_args.kwargs["models"]["resolver"],
    } == {
        "reasons": [],
        "completed": {"0", "1"},
        "phase_seconds": 12.5,
        "elapsed_seconds": 20.0,
        "provenance": {
            "reused": True,
            "source_pipeline": pipelines.XLM_MINILM,
            "documents": 2,
            "corpus_digest": "same-corpus",
            "gazetteer": "geonames",
            "gazetteer_sha256": "same-gazetteer-content",
            "min_similarity": 0.0,
            "scoring_view": "strict-raw-v1",
            "code_commit": "same-code",
            "model_pins": {
                "resolver": target_identity.model_pins["resolver"],
                "context_sentence_splitter": target_identity.model_pins[
                    "context_sentence_splitter"
                ],
            },
        },
        "scored_resolver": target_identity.model_pins["resolver"],
    }


def test_reused_recognition_predictions_are_pinned_and_labeled(tmp_path, monkeypatch):
    run = _run(tmp_path)
    run.output_dir.mkdir(exist_ok=True)
    source_identity = benchmark._run_identity(run, pipelines.GLINER_MINILM)
    target_identity = benchmark._run_identity(run, pipelines.GLINER_JINA)
    source_state = ckpt.Checkpoint(source_identity)
    source_state.phase_elapsed_seconds[runner.RECOGNITION] = 4.5
    for document_id in ("0", "1"):
        source_state.record(
            runner.RECOGNITION,
            document_id,
            [{"start": 0, "end": 1, "identifier": None}],
        )
    ckpt.save(
        run.output_dir / f"checkpoint-{pipelines.GLINER_MINILM}.json", source_state
    )
    scored = report.PipelineResult(name=pipelines.GLINER_JINA)
    expected_result = report.PipelineResult(name=pipelines.GLINER_JINA)
    score = Mock(return_value=expected_result)
    monkeypatch.setattr(runner, "score", score)

    result = benchmark._reuse_recognition(
        run, pipelines.GLINER_JINA, pipelines.GLINER_MINILM, scored
    )

    target_path = run.output_dir / f"checkpoint-{pipelines.GLINER_JINA}.json"
    copied, reasons = ckpt.load(target_path, target_identity)
    assert (
        result is expected_result,
        reasons,
        copied.completed(runner.RECOGNITION),
        copied.recognition,
        copied.phase_elapsed_seconds[runner.RECOGNITION],
        copied.phase_provenance[runner.RECOGNITION],
        score.call_args.kwargs["models"]["recognizer"],
    ) == (
        True,
        [],
        {"0", "1"},
        source_state.recognition,
        4.5,
        {
            "reused": True,
            "source_pipeline": pipelines.GLINER_MINILM,
            "documents": 2,
            "corpus_digest": "same-corpus",
            "scoring_view": "strict-raw-v1",
            "code_commit": "same-code",
            "model_pins": {"recognizer": source_identity.model_pins["recognizer"]},
            "execution_settings": {
                "document_chunk_size": source_identity.execution_settings[
                    "document_chunk_size"
                ],
                "recognition": source_identity.execution_settings["recognition"],
            },
        },
        target_identity.model_pins["recognizer"],
    )


def test_load_reusable_recognition_rejects_a_different_recognizer(tmp_path):
    run = _run(tmp_path)

    with pytest.raises(RuntimeError, match="the corpus, recognizer"):
        benchmark._load_reusable_recognition(
            run, pipelines.GLINER_JINA, pipelines.XLM_JINA
        )


def test_load_reusable_recognition_rejects_checkpoint_load_reasons(
    monkeypatch, tmp_path
):
    run = _run(tmp_path)
    source_identity = benchmark._run_identity(run, pipelines.GLINER_MINILM)
    load = Mock(return_value=(ckpt.Checkpoint(source_identity), ["stale checkpoint"]))
    monkeypatch.setattr(ckpt, "load", load)

    with pytest.raises(RuntimeError, match="stale checkpoint"):
        benchmark._load_reusable_recognition(
            run, pipelines.GLINER_JINA, pipelines.GLINER_MINILM
        )


def test_load_reusable_recognition_rejects_an_incomplete_source(tmp_path):
    run = _run(tmp_path)
    run.output_dir.mkdir(exist_ok=True)
    source_identity = benchmark._run_identity(run, pipelines.GLINER_MINILM)
    ckpt.save(
        run.output_dir / f"checkpoint-{pipelines.GLINER_MINILM}.json",
        ckpt.Checkpoint(source_identity),
    )

    with pytest.raises(RuntimeError, match="Cannot reuse incomplete"):
        benchmark._load_reusable_recognition(
            run, pipelines.GLINER_JINA, pipelines.GLINER_MINILM
        )


def test_load_reusable_resolution_rejects_a_different_resolver(tmp_path):
    run = _run(tmp_path)

    try:
        benchmark._load_reusable_resolution(run, "otter-minilm", pipelines.SWAPPED)
    except RuntimeError as error:
        assert (f"Cannot reuse {pipelines.SWAPPED} resolution for otter-minilm") in str(
            error
        )
    else:
        raise AssertionError("a different resolver must not be reused")


def test_load_reusable_resolution_rejects_checkpoint_load_reasons(
    monkeypatch, tmp_path
):
    run = _run(tmp_path)
    run.output_dir.mkdir(exist_ok=True)
    source_identity = benchmark._run_identity(run, pipelines.XLM_MINILM)
    load = Mock(return_value=(ckpt.Checkpoint(source_identity), ["stale checkpoint"]))
    monkeypatch.setattr(ckpt, "load", load)

    try:
        benchmark._load_reusable_resolution(run, "otter-minilm", pipelines.XLM_MINILM)
    except RuntimeError as error:
        assert "stale checkpoint" in str(error)
    else:
        raise AssertionError("checkpoint load reasons must block reuse")
    load.assert_called_once_with(
        run.output_dir / f"checkpoint-{pipelines.XLM_MINILM}.json",
        source_identity,
    )


def test_load_reusable_resolution_rejects_incomplete_source(
    tmp_path,
):
    run = _run(tmp_path)
    run.output_dir.mkdir(exist_ok=True)
    source_identity = benchmark._run_identity(run, pipelines.XLM_MINILM)
    ckpt.save(
        run.output_dir / f"checkpoint-{pipelines.XLM_MINILM}.json",
        ckpt.Checkpoint(source_identity),
    )

    try:
        benchmark._load_reusable_resolution(run, "otter-minilm", pipelines.XLM_MINILM)
    except RuntimeError as error:
        assert "Cannot reuse incomplete" in str(error)
        assert "0/2 documents" in str(error)
    else:
        raise AssertionError("an incomplete resolution checkpoint must not be reused")
