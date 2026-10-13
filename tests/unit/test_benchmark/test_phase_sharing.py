"""Sharing gold-span resolution must preserve its full provenance."""

from argparse import Namespace
from unittest.mock import Mock, call

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
        pipeline=[pipelines.XLM_MINILM, pipelines.OTTER_MINILM],
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
    otter_result = report.PipelineResult(pipelines.OTTER_MINILM)
    score_pipeline = Mock(side_effect=[xlm_result, otter_result])
    reuse_resolution = Mock(return_value=otter_result)
    monkeypatch.setattr(benchmark, "_score_pipeline", score_pipeline)
    monkeypatch.setattr(benchmark, "_reuse_resolution", reuse_resolution)

    results = benchmark._score_requested_pipelines(run)

    assert results == [xlm_result, otter_result]
    assert score_pipeline.call_args_list == [
        call(run, pipelines.XLM_MINILM, [runner.RECOGNITION, runner.RESOLUTION]),
        call(run, pipelines.OTTER_MINILM, [runner.RECOGNITION]),
    ]
    reuse_resolution.assert_called_once_with(
        run, pipelines.OTTER_MINILM, pipelines.XLM_MINILM, otter_result
    )


def test_reused_resolution_predictions_are_pinned_and_labeled(tmp_path, monkeypatch):
    run = _run(tmp_path)
    run.output_dir.mkdir(exist_ok=True)
    source_identity = benchmark._run_identity(run, pipelines.XLM_MINILM)
    target_identity = benchmark._run_identity(run, pipelines.OTTER_MINILM)
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
    target_path = run.output_dir / f"checkpoint-{pipelines.OTTER_MINILM}.json"
    ckpt.save(target_path, target_state)

    scored = report.PipelineResult(
        name=pipelines.OTTER_MINILM,
        models={"recognizer": target_identity.model_pins["recognizer"]},
    )
    score = Mock(return_value=report.PipelineResult(name=pipelines.OTTER_MINILM))
    monkeypatch.setattr(runner, "score", score)

    benchmark._reuse_resolution(
        run, pipelines.OTTER_MINILM, pipelines.XLM_MINILM, scored
    )

    copied, reasons = ckpt.load(target_path, target_identity)
    assert reasons == []
    assert copied.completed(runner.RESOLUTION) == {"0", "1"}
    assert copied.phase_elapsed_seconds[runner.RESOLUTION] == 12.5
    assert copied.elapsed_seconds == 20.0
    assert copied.phase_provenance[runner.RESOLUTION] == {
        "reused": True,
        "source_pipeline": pipelines.XLM_MINILM,
        "documents": 2,
        "corpus_digest": "same-corpus",
        "gazetteer": "geonames",
        "gazetteer_sha256": "same-gazetteer-content",
        "min_similarity": 0.0,
        "code_commit": "same-code",
        "model_pins": {
            "resolver": target_identity.model_pins["resolver"],
            "context_sentence_splitter": target_identity.model_pins[
                "context_sentence_splitter"
            ],
        },
    }
    assert (
        score.call_args.kwargs["models"]["resolver"]
        == target_identity.model_pins["resolver"]
    )
