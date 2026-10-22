"""
Command line entry point for the geoparsing benchmark.

    python -m scripts.benchmark --limit 10 --device auto
    python -m scripts.benchmark --output-dir ~/geoparser-bench --device cuda
    python -m scripts.benchmark --corpus hipe2020-de --corpus newseye-fi
    python -m scripts.benchmark --corpus all

Each corpus is scored into its own folder under the output directory, and a
summary across all of them is written at its root.

Re-running the same command resumes: work already checkpointed is skipped, so
a job that ended at its walltime continues in the next one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._cli import EXIT_OK
from scripts._io import PROJECT_ROOT, write_json_atomic, write_text_atomic
from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import (
    corpora,
    corpus,
    corpus_checks,
    missing_matrix,
    pipelines,
    provenance,
    report,
    runner,
)

REPOSITORY_ROOT = PROJECT_ROOT
PROBLEMS_SHOWN = 5


class CorpusRejectedError(Exception):
    """A corpus failed its offline checks, so no model was run on it."""


@dataclass(frozen=True)
class _CorpusRun:
    """Inputs shared while one corpus is resumed, scored, and reported."""

    loaded: corpora.LoadedCorpus
    arguments: argparse.Namespace
    output_dir: Path
    device: str
    commit: str
    facts: dict


@dataclass(frozen=True)
class _ManifestPlan:
    """The requested corpus/model/phase matrix with its run identity."""

    names: Sequence[str]
    pipeline_names: Sequence[str]
    phases: Sequence[str]
    arguments: argparse.Namespace
    commit: str
    full_commit: str
    facts: dict
    device: str


@dataclass(frozen=True)
class _CorpusExecution:
    """Shared output and run inputs while selected corpora are scored."""

    output_dir: Path
    arguments: argparse.Namespace
    device: str
    commit: str
    facts: dict
    phases: Sequence[str]
    manifest: dict


@dataclass(frozen=True)
class _PhaseReusePlan:
    """The selected phases and drivers for one pipeline's reuse checks."""

    phases: Sequence[str]
    selected: Sequence[str]
    recognition_drivers: dict[str, str]
    resolver_drivers: dict[str, str]
    shareable: set[str]


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.benchmark", description=__doc__
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        choices=list(pipelines.PIPELINES),
        help="Pipeline to run; repeat for several. Defaults to all six current combinations.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        choices=[*corpora.CORPORA, "all"],
        help="Corpus to score; repeat for several, or 'all'. Defaults to geovirus.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N documents"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results"),
        help="Where checkpoints and reports are written",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to place the models; auto prefers CUDA when present",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=runner.DEFAULT_CHUNK_SIZE,
        help="Documents processed between checkpoint saves",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help=(
            "Abstention threshold shared by both resolvers. The default of 0.0 "
            "lets neither abstain, so the comparison is of ranking rather than "
            "of a threshold calibrated for one model."
        ),
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=[runner.RECOGNITION, runner.RESOLUTION],
        help="Phase to run; repeat for several. Defaults to both.",
    )
    parser.add_argument(
        "--scoring-view",
        choices=[corpora.STRICT_RAW_VIEW, corpora.EXACT_FULL_RECORD_DEDUP_VIEW],
        default=corpora.STRICT_RAW_VIEW,
        help=(
            "Scoring protocol. NewsLi exact-record dedup selects the reviewed "
            "seven-language view and remains separate from strict raw."
        ),
    )
    return parser


def corpus_output_dir(output_dir: Path, name: str) -> Path:
    """Return the folder one corpus's checkpoints and report are written to."""
    return output_dir / name


def _total_note(label: str, value: int | None) -> str:
    """Name one source total with the value it was compared with, or say it was not."""
    if value is None:
        return f"{label} not compared"
    return f"{label} compared with {value}"


def totals_note(loaded: corpora.LoadedCorpus) -> str:
    """Say which source totals a run was compared with, and which it was not."""
    documents = _total_note("documents", loaded.expected_documents)
    gold = _total_note("gold spans", loaded.expected_gold)
    note = f"{loaded.name}: {documents}; {gold}"
    if loaded.integrity_report is not None:
        note += "; full source checked against recorded totals"
    return note


def _corpus_problems(loaded: corpora.LoadedCorpus) -> list[corpus_checks.Problem]:
    """Combine subset checks with any full-source integrity checks."""
    found = corpus_checks.check_corpus(
        loaded.documents,
        expected_documents=loaded.expected_documents,
        expected_gold=loaded.expected_gold,
        dropped_gold=loaded.dropped_gold,
        allow_exact_duplicate_records=(
            loaded.name.startswith("newsli-")
            and loaded.scoring_view == corpora.STRICT_RAW_VIEW
        ),
    )
    reports = [found]
    if loaded.integrity_report is not None:
        reports.append(loaded.integrity_report)
    return list(
        dict.fromkeys(problem for report in reports for problem in report.problems)
    )


def _corpus_rejection_message(name: str, problems: list[corpus_checks.Problem]) -> str:
    """Format the bounded summary for a rejected corpus."""
    shown = "; ".join(
        f"{problem.kind} in {problem.document}: {problem.detail}"
        for problem in problems[:PROBLEMS_SHOWN]
    )
    return (
        f"{name} rejected by corpus checks, no model was run: "
        f"{len(problems)} problems. {shown}"
    )


def require_clean_corpus(loaded: corpora.LoadedCorpus) -> None:
    """
    Stop before any model runs when the corpus fails its offline checks.

    Args:
        loaded: The corpus about to be scored

    Raises:
        CorpusRejectedError: If any check found a problem in the corpus
    """
    problems = _corpus_problems(loaded)
    if not problems:
        _check_frozen_corpus_totals(loaded)
        print(totals_note(loaded))
        return
    raise CorpusRejectedError(_corpus_rejection_message(loaded.name, problems))


def _check_frozen_corpus_totals(loaded: corpora.LoadedCorpus) -> None:
    """Reject a frozen matrix slice whose loaded gold totals changed."""
    expected = _frozen_matrix_cell(loaded.name)
    if expected is None or (
        loaded.limit is not None and loaded.limit < expected.documents
    ):
        return
    _check_known_frozen_totals(loaded, expected)


def _frozen_matrix_cell(name: str) -> Any | None:
    """Return the fixed corpus slice for a benchmark corpus name."""
    return next(
        (
            cell
            for cell in missing_matrix.MATRIX
            if missing_matrix.cell_name(cell) == name
        ),
        None,
    )


def _check_known_frozen_totals(loaded: corpora.LoadedCorpus, expected: Any) -> None:
    """Compare full-load counts with the frozen slice, independent of adapter totals."""
    actual = (len(loaded.documents), corpus.gold_toponym_count(loaded.documents))
    frozen = (expected.documents, expected.gold_toponyms)
    if actual != frozen:
        message = (
            f"{loaded.name} differs from the frozen matrix: expected "
            f"{expected.documents} documents/{expected.gold_toponyms} gold spans, "
            f"found {actual[0]}/{actual[1]}; no model ran."
        )
        raise CorpusRejectedError(message)


def run_corpus(
    loaded: corpora.LoadedCorpus,
    arguments: argparse.Namespace,
    output_dir: Path,
    *,
    device: str,
    commit: str,
    facts: dict,
) -> list[dict]:
    """
    Score every requested pipeline on one corpus and write its report.

    Returns:
        One summary row per pipeline

    Raises:
        CorpusRejectedError: If the corpus fails its checks, before any model runs
    """
    require_clean_corpus(loaded)
    documents = loaded.documents
    gold_count = corpus.gold_toponym_count(documents)
    scoring_spans = runner.scoring_span_count(documents)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{loaded.name} ({loaded.language}): {len(documents)} documents, "
        f"{gold_count} gold reference rows, {scoring_spans} distinct scoring spans, "
        f"corpus {loaded.digest}, commit {commit}"
    )
    run = _CorpusRun(loaded, arguments, output_dir, device, commit, facts)
    results = _score_requested_pipelines(run)
    _write_inspection_outputs(run, results)

    text = report.render_markdown(
        results,
        corpus_name=loaded.name,
        documents=len(documents),
        gold_toponyms=gold_count,
        gazetteer=pipelines.GAZETTEER_NAME,
        min_similarity=arguments.min_similarity,
        partial=_limit_is_partial(loaded.name, arguments.limit),
        scoring_spans=scoring_spans,
    )
    _write_corpus_reports(
        output_dir,
        text,
        _corpus_report_payload(run, gold_count, results, scoring_spans=scoring_spans),
    )
    print(text)
    return _summary_rows(run, gold_count, results, scoring_spans=scoring_spans)


def _write_inspection_outputs(
    run: _CorpusRun, results: Sequence[report.PipelineResult]
) -> None:
    """Write one text-free gold/prediction file per pipeline and corpus."""
    for result in results:
        identity = _run_identity(run, result.name)
        checkpoint_path = run.output_dir / f"checkpoint-{result.name}.json"
        state, reasons = ckpt.load(checkpoint_path, identity)
        if reasons:
            msg = (
                f"Cannot write inspection output for {result.name}: "
                f"{', '.join(reasons)}"
            )
            raise RuntimeError(msg)
        runner.write_inspection(
            run.output_dir / f"inspection-{result.name}.jsonl",
            run.loaded.documents,
            state,
            pipeline=result.name,
            scoring_view=run.loaded.scoring_view,
        )


def _score_requested_pipelines(
    run: _CorpusRun,
) -> list[report.PipelineResult]:
    """Run each selected pipeline over the requested corpus phases."""
    pipelines_to_run = run.arguments.pipeline or list(pipelines.DEFAULT_PIPELINES)
    phases = _requested_phases(getattr(run.arguments, "phase", None))
    shareable = set(pipelines.DEFAULT_PIPELINES)
    drivers = _resolution_drivers(pipelines_to_run, phases, shareable)
    recognition_drivers = _recognition_drivers(pipelines_to_run, phases, shareable)
    return [
        _score_one_requested_pipeline(
            run, name, phases, shareable, drivers, recognition_drivers
        )
        for name in pipelines_to_run
    ]


def _requested_phases(phases: Sequence[str] | None) -> Sequence[str]:
    """Return selected benchmark phases or the full phase set."""
    return phases or (runner.RECOGNITION, runner.RESOLUTION)


def _resolution_drivers(
    pipeline_names: Sequence[str], phases: Sequence[str], shareable: set[str]
) -> dict[str, str]:
    """Choose one matching resolver run to produce reusable resolution data."""
    if runner.RESOLUTION not in phases:
        return {}
    drivers: dict[str, str] = {}
    for name in pipeline_names:
        if name in shareable:
            drivers.setdefault(missing_matrix.resolver_family(name), name)
    return drivers


def _recognition_drivers(
    pipeline_names: Sequence[str], phases: Sequence[str], shareable: set[str]
) -> dict[str, str]:
    """Choose one run per recognizer to produce reusable span predictions."""
    if runner.RECOGNITION not in phases:
        return {}
    drivers: dict[str, str] = {}
    for name in pipeline_names:
        if name in shareable:
            drivers.setdefault(missing_matrix.recognizer_family(name), name)
    return drivers


def _phases_for_pipeline(
    name: str,
    phases: Sequence[str],
    shareable: set[str],
    drivers: dict[str, str],
    recognition_drivers: dict[str, str] | None = None,
) -> list[str]:
    """Skip phases whose exact model components already have a driver."""
    if name not in shareable:
        return list(phases)
    selected = _remove_shared_resolver_phase(name, phases, drivers)
    return _remove_shared_recognition_phase(name, selected, recognition_drivers or {})


def _remove_shared_resolver_phase(
    name: str, phases: Sequence[str], drivers: dict[str, str]
) -> list[str]:
    driver = drivers.get(missing_matrix.resolver_family(name), name)
    return [phase for phase in phases if phase != runner.RESOLUTION or driver == name]


def _remove_shared_recognition_phase(
    name: str, phases: Sequence[str], drivers: dict[str, str]
) -> list[str]:
    driver = drivers.get(missing_matrix.recognizer_family(name), name)
    return [phase for phase in phases if phase != runner.RECOGNITION or driver == name]


def _score_one_requested_pipeline(
    run: _CorpusRun,
    name: str,
    phases: Sequence[str],
    shareable: set[str],
    drivers: dict[str, str],
    recognition_drivers: dict[str, str] | None = None,
) -> report.PipelineResult:
    """Run needed phases and attach compatible component predictions."""
    recognition_drivers = recognition_drivers or {}
    selected = _phases_for_pipeline(
        name, phases, shareable, drivers, recognition_drivers
    )
    result = _score_pipeline(run, name, selected)
    plan = _PhaseReusePlan(
        phases=phases,
        selected=selected,
        recognition_drivers=recognition_drivers,
        resolver_drivers=drivers,
        shareable=shareable,
    )
    return _reuse_skipped_phases(run, name, result, plan)


def _reuse_skipped_phases(
    run: _CorpusRun,
    name: str,
    result: report.PipelineResult,
    plan: _PhaseReusePlan,
) -> report.PipelineResult:
    if name not in plan.shareable:
        return result
    sources = (
        (
            runner.RECOGNITION,
            plan.recognition_drivers.get(missing_matrix.recognizer_family(name)),
        ),
        (
            runner.RESOLUTION,
            plan.resolver_drivers.get(missing_matrix.resolver_family(name)),
        ),
    )
    for phase, source in sources:
        result = _reuse_phase_if_needed(run, name, phase, source, result, plan)
    return result


def _reuse_phase_if_needed(
    run: _CorpusRun,
    name: str,
    phase: str,
    source: str | None,
    result: report.PipelineResult,
    plan: _PhaseReusePlan,
) -> report.PipelineResult:
    if (
        phase not in plan.phases
        or phase in plan.selected
        or not source
        or source == name
    ):
        return result
    return _reuse_phase(run, name, source, result, phase)


def _run_identity(run: _CorpusRun, pipeline_name: str) -> ckpt.RunIdentity:
    """Build the complete checkpoint identity, including immutable model pins."""
    phases = _requested_phases(getattr(run.arguments, "phase", None))
    return ckpt.RunIdentity(
        pipeline=pipeline_name,
        corpus_digest=run.loaded.digest,
        gazetteer=pipelines.GAZETTEER_NAME,
        gazetteer_sha256=run.facts.get("gazetteer_artifact_sha256"),
        min_similarity=run.arguments.min_similarity,
        limit=run.arguments.limit,
        commit=run.commit,
        model_pins=pipelines.pipeline_model_pins(pipeline_name),
        scoring_view=getattr(run.loaded, "scoring_view", corpora.STRICT_RAW_VIEW),
        execution_settings=_pipeline_execution_settings(
            pipeline_name,
            getattr(run.arguments, "chunk_size", runner.DEFAULT_CHUNK_SIZE),
            phases,
        ),
        accelerator=pipelines.describe_device(run.device),
    )


def _pipeline_execution_settings(
    pipeline_name: str, chunk_size: int, phases: Sequence[str]
) -> dict[str, Any]:
    """Record settings only for phases selected by this benchmark run."""
    return pipelines.pipeline_execution_settings(
        pipeline_name,
        chunk_size,
        include_recognition=runner.RECOGNITION in phases,
        include_resolution=runner.RESOLUTION in phases,
    )


def _reuse_resolution(
    run: _CorpusRun,
    target_pipeline: str,
    source_pipeline: str,
    target_result: report.PipelineResult,
) -> report.PipelineResult:
    """Copy a complete gold-span resolution phase after exact identity checks."""
    source_identity, target_identity, source_state = _load_reusable_resolution(
        run, target_pipeline, source_pipeline
    )
    target_state = _copy_reusable_resolution(
        run,
        target_pipeline,
        source_pipeline,
        target_identity,
        source_identity,
        source_state,
    )
    models = _resolution_score_models(target_result, target_identity)
    return runner.score(
        target_pipeline,
        run.loaded.documents,
        target_state,
        models=models,
        device=run.device,
    )


def _reuse_phase(
    run: _CorpusRun,
    target_pipeline: str,
    source_pipeline: str,
    target_result: report.PipelineResult,
    phase: str,
) -> report.PipelineResult:
    """Copy a complete phase after its full model and input identity matches."""
    if phase == runner.RECOGNITION:
        return _reuse_recognition(run, target_pipeline, source_pipeline, target_result)
    if phase == runner.RESOLUTION:
        return _reuse_resolution(run, target_pipeline, source_pipeline, target_result)
    msg = f"Unknown phase for reuse: {phase}"
    raise ValueError(msg)


def _load_reusable_recognition(
    run: _CorpusRun, target_pipeline: str, source_pipeline: str
) -> tuple[ckpt.RunIdentity, ckpt.RunIdentity, ckpt.Checkpoint]:
    """Load complete source spans only when corpus and recognizer pins match."""
    source_identity = _run_identity(run, source_pipeline)
    target_identity = _run_identity(run, target_pipeline)
    if not ckpt.recognition_reuse_compatible(source_identity, target_identity):
        msg = (
            f"Cannot reuse {source_pipeline} recognition for {target_pipeline}: "
            "the corpus, recognizer, execution settings or code pins differ."
        )
        raise RuntimeError(msg)
    source_path = run.output_dir / f"checkpoint-{source_pipeline}.json"
    source_state, reasons = ckpt.load(source_path, source_identity)
    if reasons:
        msg = f"Cannot reuse {source_pipeline} recognition: {', '.join(reasons)}"
        raise RuntimeError(msg)
    expected_documents = {document.identifier for document in run.loaded.documents}
    if source_state.completed(runner.RECOGNITION) != expected_documents:
        msg = (
            f"Cannot reuse incomplete {source_pipeline} recognition: "
            f"{len(source_state.completed(runner.RECOGNITION))}/"
            f"{len(expected_documents)} documents."
        )
        raise RuntimeError(msg)
    return source_identity, target_identity, source_state


def _recognition_reuse_provenance(
    source_pipeline: str,
    source_identity: ckpt.RunIdentity,
    document_count: int,
) -> dict:
    """Describe the recognizer and inputs behind copied span predictions."""
    return {
        "reused": True,
        "source_pipeline": source_pipeline,
        "documents": document_count,
        "corpus_digest": source_identity.corpus_digest,
        "scoring_view": source_identity.scoring_view,
        "code_commit": source_identity.commit,
        "model_pins": {
            name: pin
            for name, pin in source_identity.model_pins.items()
            if name == "recognizer"
        },
        "execution_settings": {
            "document_chunk_size": source_identity.execution_settings[
                "document_chunk_size"
            ],
            "recognition": source_identity.execution_settings["recognition"],
        },
    }


def _copy_reusable_recognition(
    run: _CorpusRun,
    target_pipeline: str,
    source_pipeline: str,
    target_identity: ckpt.RunIdentity,
    source_identity: ckpt.RunIdentity,
    source_state: ckpt.Checkpoint,
) -> ckpt.Checkpoint:
    """Copy recognition annotations, failures, timing and provenance."""
    target_path = run.output_dir / f"checkpoint-{target_pipeline}.json"
    expected_documents = {document.identifier for document in run.loaded.documents}
    target_state, reasons = ckpt.load(target_path, target_identity)
    for reason in reasons:
        print(f"  {target_pipeline}: starting fresh -- {reason}")
    if target_state.completed(runner.RECOGNITION) != expected_documents:
        previous_seconds = target_state.phase_elapsed_seconds.get(
            runner.RECOGNITION, 0.0
        )
        target_state.recognition = copy.deepcopy(source_state.recognition)
        target_state.failures[runner.RECOGNITION] = copy.deepcopy(
            source_state.failures.get(runner.RECOGNITION, {})
        )
        target_state.phase_elapsed_seconds[runner.RECOGNITION] = (
            source_state.phase_elapsed_seconds.get(runner.RECOGNITION, 0.0)
        )
        target_state.elapsed_seconds += (
            target_state.phase_elapsed_seconds[runner.RECOGNITION] - previous_seconds
        )
        target_state.phase_provenance[runner.RECOGNITION] = (
            _recognition_reuse_provenance(
                source_pipeline, source_identity, len(expected_documents)
            )
        )
        ckpt.save(target_path, target_state)
    return target_state


def _reuse_recognition(
    run: _CorpusRun,
    target_pipeline: str,
    source_pipeline: str,
    target_result: report.PipelineResult,
) -> report.PipelineResult:
    """Score one pipeline after reusing an identical recognition component."""
    source_identity, target_identity, source_state = _load_reusable_recognition(
        run, target_pipeline, source_pipeline
    )
    target_state = _copy_reusable_recognition(
        run,
        target_pipeline,
        source_pipeline,
        target_identity,
        source_identity,
        source_state,
    )
    models = dict(target_result.models)
    models["recognizer"] = target_identity.model_pins["recognizer"]
    return runner.score(
        target_pipeline,
        run.loaded.documents,
        target_state,
        models=models,
        device=run.device,
    )


def _load_reusable_resolution(
    run: _CorpusRun, target_pipeline: str, source_pipeline: str
) -> tuple[ckpt.RunIdentity, ckpt.RunIdentity, ckpt.Checkpoint]:
    """Load only a complete source checkpoint with an identical resolver identity."""
    source_identity = _run_identity(run, source_pipeline)
    target_identity = _run_identity(run, target_pipeline)
    if not ckpt.resolution_reuse_compatible(source_identity, target_identity):
        message = (
            f"Cannot reuse {source_pipeline} resolution for {target_pipeline}: "
            "the corpus, resolver, gazetteer, threshold or code pins differ."
        )
        raise RuntimeError(message)
    source_path = run.output_dir / f"checkpoint-{source_pipeline}.json"
    source_state, reasons = ckpt.load(source_path, source_identity)
    if reasons:
        message = f"Cannot reuse {source_pipeline} resolution: {', '.join(reasons)}"
        raise RuntimeError(message)
    expected_documents = {document.identifier for document in run.loaded.documents}
    if source_state.completed(runner.RESOLUTION) != expected_documents:
        message = (
            f"Cannot reuse incomplete {source_pipeline} resolution: "
            f"{len(source_state.completed(runner.RESOLUTION))}/"
            f"{len(expected_documents)} documents."
        )
        raise RuntimeError(message)
    return source_identity, target_identity, source_state


def _resolution_reuse_provenance(
    source_pipeline: str,
    source_identity: ckpt.RunIdentity,
    document_count: int,
) -> dict:
    """Describe why copied resolution predictions belong to this target run."""
    return {
        "reused": True,
        "source_pipeline": source_pipeline,
        "documents": document_count,
        "corpus_digest": source_identity.corpus_digest,
        "gazetteer": source_identity.gazetteer,
        "gazetteer_sha256": source_identity.gazetteer_sha256,
        "min_similarity": source_identity.min_similarity,
        "scoring_view": source_identity.scoring_view,
        "code_commit": source_identity.commit,
        "model_pins": {
            name: pin
            for name, pin in source_identity.model_pins.items()
            if name in {"resolver", "reranker", "context_sentence_splitter"}
        },
    }


def _copy_reusable_resolution(
    run: _CorpusRun,
    target_pipeline: str,
    source_pipeline: str,
    target_identity: ckpt.RunIdentity,
    source_identity: ckpt.RunIdentity,
    source_state: ckpt.Checkpoint,
) -> ckpt.Checkpoint:
    """Copy resolution data into a target checkpoint and write its provenance."""
    target_path = run.output_dir / f"checkpoint-{target_pipeline}.json"
    expected_documents = {document.identifier for document in run.loaded.documents}
    target_state, reasons = ckpt.load(target_path, target_identity)
    for reason in reasons:
        print(f"  {target_pipeline}: starting fresh -- {reason}")
    if target_state.completed(runner.RESOLUTION) != expected_documents:
        previous_resolution_seconds = target_state.phase_elapsed_seconds.get(
            runner.RESOLUTION, 0.0
        )
        target_state.resolution = copy.deepcopy(source_state.resolution)
        target_state.failures[runner.RESOLUTION] = copy.deepcopy(
            source_state.failures.get(runner.RESOLUTION, {})
        )
        target_state.phase_elapsed_seconds[runner.RESOLUTION] = (
            source_state.phase_elapsed_seconds.get(runner.RESOLUTION, 0.0)
        )
        target_state.elapsed_seconds += (
            target_state.phase_elapsed_seconds[runner.RESOLUTION]
            - previous_resolution_seconds
        )
        target_state.phase_provenance[runner.RESOLUTION] = {
            **_resolution_reuse_provenance(
                source_pipeline, source_identity, len(expected_documents)
            )
        }
        ckpt.save(target_path, target_state)
    return target_state


def _resolution_score_models(
    target_result: report.PipelineResult, target_identity: ckpt.RunIdentity
) -> dict[str, str]:
    """Combine recognizer pins from a partial target with its resolver pins."""
    models = dict(target_result.models)
    models.update(
        {
            name: target_identity.model_pins[name]
            for name in ("resolver", "reranker", "context_sentence_splitter")
            if name in target_identity.model_pins
        }
    )
    return models


def _score_pipeline(
    run: _CorpusRun,
    pipeline_name: str,
    phases: Sequence[str],
) -> report.PipelineResult:
    """Resume, run, and score one pipeline for a corpus."""
    loaded = run.loaded
    identity = _run_identity(run, pipeline_name)
    checkpoint_path = run.output_dir / f"checkpoint-{pipeline_name}.json"
    state, reasons = ckpt.load(checkpoint_path, identity)
    for reason in reasons:
        print(f"  {pipeline_name}: starting fresh -- {reason}")

    models: dict[str, str] = {}
    for phase in phases:
        models.update(
            _run_and_record_phase(
                run, pipeline_name, phase, identity, state, checkpoint_path
            )
        )
    models = _complete_phase_model_pins(models, phases, identity.model_pins)
    return runner.score(
        pipeline_name, loaded.documents, state, models=models, device=run.device
    )


def _run_and_record_phase(
    run: _CorpusRun,
    pipeline_name: str,
    phase: str,
    identity: ckpt.RunIdentity,
    state: ckpt.Checkpoint,
    checkpoint_path: Path,
) -> dict[str, str]:
    """Execute one phase and record its code and gazetteer provenance."""
    arguments = run.arguments
    models = runner.run_phase(
        phase,
        pipeline_name,
        run.loaded.documents,
        state,
        checkpoint_path,
        runner.PhaseSettings(
            device=run.device,
            min_similarity=arguments.min_similarity,
            chunk_size=arguments.chunk_size,
        ),
    )
    expected = {document.identifier for document in run.loaded.documents}
    if state.completed(phase) == expected:
        state.phase_provenance.setdefault(
            phase,
            _phase_provenance(run, pipeline_name, phase, identity),
        )
        ckpt.save(checkpoint_path, state)
    return models


def _phase_provenance(
    run: _CorpusRun,
    pipeline_name: str,
    phase: str,
    identity: ckpt.RunIdentity,
) -> dict:
    """Build provenance for predictions generated by this run's own phase."""
    return {
        "reused": False,
        "source_pipeline": pipeline_name,
        "documents": len(run.loaded.documents),
        "corpus_digest": run.loaded.digest,
        "gazetteer": pipelines.GAZETTEER_NAME,
        "gazetteer_sha256": identity.gazetteer_sha256,
        "min_similarity": run.arguments.min_similarity,
        "code_commit": run.commit,
        "model_pins": {
            name: pin
            for name, pin in identity.model_pins.items()
            if name
            in (
                {"recognizer"}
                if phase == runner.RECOGNITION
                else {"resolver", "reranker", "context_sentence_splitter"}
            )
        },
    }


def _complete_phase_model_pins(
    models: dict[str, str], phases: Sequence[str], model_pins: dict[str, str]
) -> dict[str, str]:
    """Keep immutable model identities in scores when checkpoints were resumed."""
    for role in (
        "recognizer",
        "resolver",
        "reranker",
        "context_sentence_splitter",
    ):
        phase = runner.RECOGNITION if role == "recognizer" else runner.RESOLUTION
        if phase in phases and role in model_pins:
            models.setdefault(role, model_pins[role])
    return models


def _corpus_report_payload(
    run: _CorpusRun,
    gold_count: int,
    results: Sequence[report.PipelineResult],
    *,
    scoring_spans: int,
) -> dict:
    """Return the metadata and pipeline measurements stored as JSON."""
    loaded = run.loaded
    limit = run.arguments.limit
    return {
        "corpus": loaded.name,
        "language": loaded.language,
        "corpus_digest": loaded.digest,
        "scoring_view": loaded.scoring_view,
        "pre_transform_gold_toponyms": loaded.pre_transform_gold_toponyms,
        "exact_duplicate_records": loaded.exact_duplicate_records,
        "duplicate_records_removed": loaded.duplicate_records_removed,
        "duplicate_groups_collapsed": loaded.duplicate_groups_collapsed,
        "source_archive_sha256": loaded.source_archive_sha256,
        "source_url": loaded.source_url,
        "source_content_sha256": loaded.source_content_sha256,
        "coordinate_cache_sha256": loaded.coordinate_cache_sha256,
        "limit": limit,
        "partial": _limit_is_partial(loaded.name, limit),
        "documents": len(loaded.documents),
        "gold_toponyms": gold_count,
        "scoring_spans": scoring_spans,
        "gazetteer": pipelines.GAZETTEER_NAME,
        "gazetteer_sha256": run.facts.get("gazetteer_artifact_sha256"),
        "min_similarity": run.arguments.min_similarity,
        "commit": run.commit,
        "environment": run.facts,
        "pipelines": [
            {
                "name": result.name,
                "device": result.device,
                "models": result.models,
                "recognition": result.recognition,
                "resolution": result.resolution,
                "elapsed_seconds": result.elapsed_seconds,
                "phase_elapsed_seconds": result.phase_elapsed_seconds,
                "phase_provenance": result.phase_provenance,
                "failures": result.failures,
                "model_metadata": (
                    missing_matrix.pipeline_model_metadata(result.name, loaded.language)
                    if result.name in pipelines.DEFAULT_PIPELINES
                    or result.name in missing_matrix.MISSING_PIPELINES
                    else {}
                ),
                "integration_provenance": (
                    pipelines.otter_integration_provenance()
                    if missing_matrix.recognizer_family(result.name) == "otter_cross"
                    else {}
                ),
            }
            for result in results
        ],
    }


def _limit_is_partial(name: str, limit: int | None) -> bool:
    """Whether an explicit run limit truncates the registered corpus slice."""
    if limit is None:
        return False
    fixed = _frozen_matrix_cell(name)
    if fixed is not None:
        return limit < fixed.documents
    spec = corpora.CORPORA.get(name)
    return spec is None or spec.documents is None or limit < spec.documents


def _write_corpus_reports(output_dir: Path, markdown: str, payload: dict) -> None:
    """Persist the human and machine-readable reports side by side."""
    write_text_atomic(output_dir / "benchmark-report.md", markdown)
    write_json_atomic(output_dir / "benchmark-report.json", payload, pretty=True)


def _summary_rows(
    run: _CorpusRun,
    gold_count: int,
    results: Sequence[report.PipelineResult],
    *,
    scoring_spans: int,
) -> list[dict]:
    """Flatten corpus metadata and pipeline scores for the root summary."""
    loaded = run.loaded
    documents = loaded.documents
    return [
        {
            "corpus": loaded.name,
            "language": loaded.language,
            "documents": len(documents),
            "gold_toponyms": gold_count,
            "scoring_spans": scoring_spans,
            "pipeline": result.name,
            "recognition": result.recognition,
            "resolution": result.resolution,
            "elapsed_seconds": result.elapsed_seconds,
            "phase_elapsed_seconds": result.phase_elapsed_seconds,
            "failures": result.failures,
        }
        for result in results
    ]


def _run_manifest(plan: _ManifestPlan) -> dict:
    """Build a reproducible, incrementally updated manifest for this run."""
    scoring_view = getattr(plan.arguments, "scoring_view", corpora.STRICT_RAW_VIEW)
    cells = _manifest_corpus_cells(plan.names, scoring_view, plan.arguments.limit)
    model_runs = _manifest_pipeline_pins(
        plan.pipeline_names, plan.arguments.chunk_size, plan.phases
    )
    return _assemble_run_manifest(plan, cells, model_runs)


def _manifest_source_split(name: str) -> str:
    """Describe the registered input split for a corpus adapter."""
    if name == "geovirus":
        return "complete GeoVirus benchmark corpus"
    if name.startswith(("hipe2020-", "newseye-", "topres19th-")):
        return "HIPE-2022 v2.1 unmasked test split"
    if name.startswith("newsli-"):
        return (
            "UniTopRank NewsLi release; registered cap selects up to 500 "
            "valid article identifiers in sorted order"
        )
    return "registered corpus adapter"


def _manifest_corpus_cell(
    name: str,
    fixed: Any,
    *,
    scoring_view: str = corpora.STRICT_RAW_VIEW,
    limit: int | None = None,
) -> dict:
    """Describe expected counts and split rules for one requested corpus."""
    spec = corpora.CORPORA[name]
    counts = _manifest_counts(name, spec, fixed, scoring_view, limit)
    return _manifest_cell_data(name, spec, fixed, scoring_view, limit, counts)


def _manifest_base_counts(spec: Any, fixed: Any) -> tuple[int | None, int | None]:
    if fixed is not None:
        return fixed.documents, fixed.gold_toponyms
    return spec.documents, spec.gold_spans


def _manifest_counts(
    name: str,
    spec: Any,
    fixed: Any,
    scoring_view: str,
    limit: int | None,
) -> tuple[int | None, int | None]:
    documents, gold_toponyms = _manifest_base_counts(spec, fixed)
    if scoring_view != corpora.EXACT_FULL_RECORD_DEDUP_VIEW:
        return documents, gold_toponyms
    gold_toponyms = _deduplicated_gold_count(name, spec, fixed)
    documents, gold_toponyms = _cap_deduplicated_totals(
        fixed, limit, documents, gold_toponyms
    )
    return documents, gold_toponyms


def _deduplicated_gold_count(name: str, spec: Any, fixed: Any) -> int:
    if (
        fixed is None
        or spec.language not in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
    ):
        message = f"{name} is outside the NewsLi exact-dedup matrix"
        raise ValueError(message)
    return (
        fixed.gold_toponyms
        - missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[spec.language]
    )


def _cap_deduplicated_totals(
    fixed: Any,
    limit: int | None,
    documents: int | None,
    gold_toponyms: int,
) -> tuple[int | None, int | None]:
    if fixed is not None and limit is not None and limit < fixed.documents:
        return limit, None
    return documents, gold_toponyms


def _manifest_exact_duplicate_count(spec: Any) -> int:
    if getattr(spec, "kind", None) != corpora.NEWSLI:
        return 0
    return missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED.get(spec.language, 0)


def _manifest_scoring_spans(
    gold_toponyms: int | None,
    fixed: Any,
    scoring_view: str,
    limit: int | None,
    exact_duplicates: int,
) -> int | None:
    if _manifest_spans_unknown(gold_toponyms, fixed, limit):
        return None
    return _manifest_span_count(gold_toponyms, scoring_view, exact_duplicates)


def _manifest_spans_unknown(
    gold_toponyms: int | None, fixed: Any, limit: int | None
) -> bool:
    return gold_toponyms is None or (
        fixed is not None and limit is not None and limit < fixed.documents
    )


def _manifest_span_count(
    gold_toponyms: int | None, scoring_view: str, exact_duplicates: int
) -> int | None:
    if gold_toponyms is None:
        return None
    if scoring_view == corpora.STRICT_RAW_VIEW:
        return gold_toponyms - exact_duplicates
    return gold_toponyms


def _manifest_cell_data(
    name: str,
    spec: Any,
    fixed: Any,
    scoring_view: str,
    limit: int | None,
    counts: tuple[int | None, int | None],
) -> dict:
    documents, gold_toponyms = counts
    exact_duplicates = _manifest_exact_duplicate_count(spec)
    return {
        "language": spec.language,
        "expected_documents": documents,
        "expected_gold_toponyms": gold_toponyms,
        "scoring_spans": _manifest_scoring_spans(
            gold_toponyms, fixed, scoring_view, limit, exact_duplicates
        ),
        "pre_transform_gold_toponyms": _manifest_pretransform_gold(
            fixed, scoring_view, limit
        ),
        "exact_duplicate_records": exact_duplicates,
        "duplicate_records_removed": _manifest_removed_duplicates(spec, scoring_view),
        "evaluation_rule": _manifest_evaluation_rule(fixed),
        "source_split": _manifest_source_split(name),
        "source_url": getattr(spec, "url", None),
        "expected_source_content_sha256": getattr(spec, "expected_source_sha256", None),
        "status": "pending",
    }


def _manifest_pretransform_gold(
    fixed: Any, scoring_view: str, limit: int | None
) -> int | None:
    if fixed is None or scoring_view != corpora.EXACT_FULL_RECORD_DEDUP_VIEW:
        return None
    if limit is not None and limit < fixed.documents:
        return None
    return fixed.gold_toponyms


def _manifest_removed_duplicates(spec: Any, scoring_view: str) -> int:
    if scoring_view != corpora.EXACT_FULL_RECORD_DEDUP_VIEW:
        return 0
    return missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[spec.language]


def _manifest_evaluation_rule(fixed: Any) -> str:
    if fixed is not None:
        return "frozen first-five-corpora matrix"
    return "registered corpus adapter"


def _manifest_corpus_cells(
    names: Sequence[str],
    scoring_view: str = corpora.STRICT_RAW_VIEW,
    limit: int | None = None,
) -> dict[str, dict]:
    """Describe selected source slices and the named scoring view."""
    fixed_cells = {
        missing_matrix.cell_name(cell): cell for cell in missing_matrix.MATRIX
    }
    return {
        name: _manifest_corpus_cell(
            name, fixed_cells.get(name), scoring_view=scoring_view, limit=limit
        )
        for name in names
    }


def _manifest_pipeline_pins(
    pipeline_names: Sequence[str], chunk_size: int, phases: Sequence[str]
) -> dict[str, dict]:
    """Record model pins, effective execution settings and integration evidence."""
    model_runs = {}
    for name in pipeline_names:
        model_runs[name] = {
            "model_pins": pipelines.pipeline_model_pins(name),
            "execution_settings": _pipeline_execution_settings(
                name, chunk_size, phases
            ),
        }
        if missing_matrix.recognizer_family(name) == "otter_cross":
            model_runs[name]["integration_provenance"] = (
                pipelines.otter_integration_provenance()
            )
    return model_runs


def _assemble_run_manifest(plan: _ManifestPlan, cells: dict, model_runs: dict) -> dict:
    """Assemble protocol, environment and experiment inputs into a manifest."""
    lock_path = REPOSITORY_ROOT / "uv.lock"
    return {
        "manifest_version": 2,
        "status": "running",
        "commit": plan.commit,
        "full_commit": plan.full_commit,
        "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "started_at": plan.facts.get("started_at"),
        "environment": plan.facts,
        "protocol": {
            "gazetteer": pipelines.GAZETTEER_NAME,
            "gazetteer_artifact_sha256": plan.facts.get("gazetteer_artifact_sha256"),
            "min_similarity": plan.arguments.min_similarity,
            "limit": plan.arguments.limit,
            "chunk_size": plan.arguments.chunk_size,
            "device": plan.device,
            "accelerator": pipelines.describe_device(plan.device),
            "phases": list(plan.phases),
            "scoring_view": getattr(
                plan.arguments, "scoring_view", corpora.STRICT_RAW_VIEW
            ),
            "pipelines": model_runs,
            "legacy_pipeline_id_map": missing_matrix.LEGACY_PIPELINE_ID_MAP,
            "corpora": cells,
        },
    }


def _record_manifest_cell(
    manifest: dict,
    name: str,
    payload: dict,
    phases: Sequence[str],
) -> None:
    """Record counts, metrics, and any explicit per-document failures."""
    cell = manifest["protocol"]["corpora"][name]
    cell.update(
        {
            "corpus_digest": payload["corpus_digest"],
            "documents": payload["documents"],
            "gold_toponyms": payload["gold_toponyms"],
            "scoring_spans": payload.get("scoring_spans"),
            "scoring_view": payload.get("scoring_view", corpora.STRICT_RAW_VIEW),
            "pre_transform_gold_toponyms": payload.get("pre_transform_gold_toponyms"),
            "source_archive_sha256": payload.get("source_archive_sha256"),
            "source_url": payload.get("source_url"),
            "source_content_sha256": payload.get("source_content_sha256"),
            "coordinate_cache_sha256": payload.get("coordinate_cache_sha256"),
            "exact_duplicate_records": payload.get("exact_duplicate_records", 0),
            "duplicate_records_removed": payload.get("duplicate_records_removed", 0),
            "duplicate_groups_collapsed": payload.get("duplicate_groups_collapsed", 0),
            "environment": payload["environment"],
            "partial": bool(payload.get("partial", False)),
            "pipelines": {},
        }
    )
    for result in payload["pipelines"]:
        cell["pipelines"][result["name"]] = _manifest_pipeline_cell(
            result,
            manifest,
            cell["language"],
            phases,
            partial=cell["partial"],
        )
    records = list(cell["pipelines"].values())
    cell["status"] = _manifest_cell_status(records, partial=cell["partial"])


def _manifest_phase_status(
    result: dict, phases: Sequence[str], *, partial: bool = False
) -> dict[str, str]:
    """Mark each requested metric phase complete, partial, or not completed."""
    metrics = {
        runner.RECOGNITION: result["recognition"],
        runner.RESOLUTION: result["resolution"],
    }
    return {
        phase: ("partial" if partial else "complete")
        if metrics[phase]
        else "not_completed"
        for phase in phases
    }


def _manifest_phases_complete(phase_status: dict[str, str]) -> bool:
    """Return whether every requested phase completed."""
    return all(status == "complete" for status in phase_status.values())


def _manifest_phases_partial(phase_status: dict[str, str], *, partial: bool) -> bool:
    """Return whether every requested phase has a permitted partial result."""
    return partial and all(
        status in {"complete", "partial"} for status in phase_status.values()
    )


def _manifest_pipeline_status(phase_status: dict[str, str], *, partial: bool) -> str:
    """Summarize one pipeline's phase completion."""
    if _manifest_phases_partial(phase_status, partial=partial):
        return "partial"
    if _manifest_phases_complete(phase_status):
        return "complete"
    return "incomplete"


def _manifest_pipeline_metadata(name: str, language: str) -> dict:
    """Return model metadata for benchmark pipelines that need it."""
    return (
        missing_matrix.pipeline_model_metadata(name, language)
        if name in missing_matrix.PIPELINE_COMPONENTS
        or name in missing_matrix.MISSING_PIPELINES
        else {}
    )


def _manifest_integration_provenance(name: str) -> dict:
    """Return integration evidence only for the Otter pipeline."""
    return (
        pipelines.otter_integration_provenance()
        if missing_matrix.recognizer_family(name) == "otter_cross"
        else {}
    )


def _manifest_pipeline_cell(
    result: dict,
    manifest: dict,
    language: str,
    phases: Sequence[str],
    *,
    partial: bool = False,
) -> dict:
    """Serialize one pipeline measurement and its pinned model evidence."""
    name = result["name"]
    phase_status = _manifest_phase_status(result, phases, partial=partial)
    failures = result["failures"]
    pipeline_manifest = manifest["protocol"]["pipelines"].get(name)
    return {
        "status": _manifest_pipeline_status(phase_status, partial=partial),
        "phase_status": phase_status,
        "model_pins": (
            pipeline_manifest["model_pins"]
            if pipeline_manifest
            else pipelines.pipeline_model_pins(name)
        ),
        "model_metadata": _manifest_pipeline_metadata(name, language),
        "integration_provenance": _manifest_integration_provenance(name),
        "recognition": result["recognition"],
        "resolution": result["resolution"],
        "phase_elapsed_seconds": result["phase_elapsed_seconds"],
        "phase_provenance": result["phase_provenance"],
        "failure_count": sum(map(len, failures.values())),
        "failures": failures,
    }


def _manifest_cell_records_scored(records: Sequence[dict]) -> bool:
    """Return whether every pipeline produced requested phase results."""
    return all(record["status"] in {"complete", "partial"} for record in records)


def _manifest_cell_has_failures(records: Sequence[dict]) -> bool:
    """Return whether any pipeline recorded failed documents."""
    return any(record["failure_count"] for record in records)


def _manifest_cell_status(records: Sequence[dict], *, partial: bool = False) -> str:
    """Summarize complete or partial pipeline rows, preserving failures."""
    if not _manifest_cell_records_scored(records):
        return "incomplete"
    if partial:
        return "partial"
    if _manifest_cell_has_failures(records):
        return "complete_with_document_failures"
    return "complete"


def _persist_run_manifest(output_dir: Path, manifest: dict) -> None:
    """Write the run manifest atomically so walltime stops keep prior cells."""
    write_json_atomic(output_dir / "run-manifest.json", manifest)


def _manifest_identity(manifest: dict) -> dict:
    """Select immutable run inputs, excluding progress and measured results."""
    protocol = manifest["protocol"]
    return {
        "manifest_version": manifest.get("manifest_version"),
        "commit": manifest["commit"],
        "full_commit": manifest["full_commit"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "protocol": {
            key: protocol.get(key)
            for key in (
                "gazetteer",
                "gazetteer_artifact_sha256",
                "min_similarity",
                "limit",
                "chunk_size",
                "device",
                "accelerator",
                "scoring_view",
                "phases",
                "pipelines",
                "legacy_pipeline_id_map",
            )
        }
        | {
            "corpora": {
                name: {
                    key: value.get(key)
                    for key in (
                        "language",
                        "expected_documents",
                        "expected_gold_toponyms",
                        "pre_transform_gold_toponyms",
                        "exact_duplicate_records",
                        "duplicate_records_removed",
                        "evaluation_rule",
                        "source_split",
                        "source_url",
                        "expected_source_content_sha256",
                    )
                }
                for name, value in protocol["corpora"].items()
            }
        },
    }


def _prepare_run_manifest(output_dir: Path, plan: _ManifestPlan) -> dict:
    """Create or resume a run manifest after checking experiment identity."""
    path = output_dir / "run-manifest.json"
    manifest = _run_manifest(plan)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if _manifest_identity(previous) != _manifest_identity(manifest):
            message = (
                "The output directory has a run manifest for a different "
                "experiment. Choose a new results directory."
            )
            raise RuntimeError(message)
        manifest = previous
        manifest["status"] = "running"
        manifest.pop("completed_at", None)
    _persist_run_manifest(output_dir, manifest)
    return manifest


def _record_rejected_corpus(
    execution: _CorpusExecution, name: str, reason: str
) -> None:
    """Persist a corpus rejection without replacing its reports or checkpoints."""
    execution.manifest["protocol"]["corpora"][name].update(
        {"status": "rejected", "reason": reason}
    )
    execution.manifest["status"] = "rejected"
    execution.manifest.pop("completed_at", None)
    _persist_run_manifest(execution.output_dir, execution.manifest)


def _run_requested_corpora(
    names: Sequence[str],
    execution: _CorpusExecution,
) -> tuple[int, list[dict]]:
    """Load and score selected corpus cells, persisting progress after each."""
    rows: list[dict] = []
    output_dir = execution.output_dir
    cache_root = Path(os.environ.get("GEOPARSER_CORPUS_CACHE_DIR", str(output_dir)))
    for name in dict.fromkeys(names):
        folder = corpus_output_dir(output_dir, name)
        scoring_view = getattr(
            execution.arguments, "scoring_view", corpora.STRICT_RAW_VIEW
        )
        load_options = {"limit": execution.arguments.limit}
        if scoring_view != corpora.STRICT_RAW_VIEW:
            load_options["scoring_view"] = scoring_view
        try:
            loaded = corpora.load(name, cache_root / name, **load_options)
        except corpora.SourceHashError as error:
            print(error, file=sys.stderr)
            _record_rejected_corpus(execution, name, str(error))
            return 1, rows
        try:
            rows += run_corpus(
                loaded,
                execution.arguments,
                folder,
                device=execution.device,
                commit=execution.commit,
                facts=execution.facts,
            )
        except CorpusRejectedError as error:
            print(error, file=sys.stderr)
            _record_rejected_corpus(execution, name, str(error))
            return 1, rows
        _record_completed_corpus(execution, folder, name, rows)
    return EXIT_OK, rows


def _record_completed_corpus(
    execution: _CorpusExecution,
    folder: Path,
    name: str,
    rows: list[dict],
) -> None:
    """Record the newly completed corpus payload and refresh run summaries."""
    payload = json.loads((folder / "benchmark-report.json").read_text(encoding="utf-8"))
    _record_manifest_cell(execution.manifest, name, payload, execution.phases)
    _persist_run_manifest(execution.output_dir, execution.manifest)
    write_text_atomic(execution.output_dir / "summary.md", report.render_summary(rows))
    write_json_atomic(
        execution.output_dir / "summary.json",
        {"commit": execution.commit, "environment": execution.facts, "rows": rows},
        pretty=True,
    )


def _finish_run_manifest(output_dir: Path, manifest: dict, rows: list[dict]) -> None:
    """Mark the run status, persist its completion time and render final scores."""
    statuses = [cell["status"] for cell in manifest["protocol"]["corpora"].values()]
    completed_statuses = {
        "complete",
        "complete_with_document_failures",
        "partial",
    }
    if all(status in completed_statuses for status in statuses):
        manifest["status"] = "partial" if "partial" in statuses else "complete"
    else:
        manifest["status"] = "incomplete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _persist_run_manifest(output_dir, manifest)
    print(report.render_summary(rows))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark on every requested corpus and write the reports."""
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEOPARSER_DB_PATH", str(output_dir / "benchmark.sqlite"))
    plan = _manifest_plan(arguments)
    manifest = _prepare_run_manifest(output_dir, plan)
    execution = _CorpusExecution(
        output_dir,
        arguments,
        plan.device,
        plan.commit,
        plan.facts,
        plan.phases,
        manifest,
    )
    exit_code, rows = _run_requested_corpora(plan.names, execution)
    if exit_code != EXIT_OK:
        return exit_code
    _finish_run_manifest(output_dir, manifest, rows)
    return EXIT_OK


def _manifest_plan(arguments: argparse.Namespace) -> _ManifestPlan:
    """Resolve CLI choices and capture code, device and environment identity."""
    names = _requested_corpus_names(arguments)
    device = pipelines.resolve_device(arguments.device)
    commit = provenance.source_commit(REPOSITORY_ROOT)
    full_commit = provenance.source_commit_full(REPOSITORY_ROOT)
    facts = provenance.environment(os.environ.get("OAR_JOB_ID"))
    phases = arguments.phase or [runner.RECOGNITION, runner.RESOLUTION]
    if runner.RESOLUTION in phases:
        facts["gazetteer_artifact_sha256"] = provenance.gazetteer_artifact_sha256(
            pipelines.GAZETTEER_NAME
        )
    print(f"device: {pipelines.describe_device(device)}")
    return _ManifestPlan(
        names=names,
        pipeline_names=arguments.pipeline or list(pipelines.DEFAULT_PIPELINES),
        phases=phases,
        arguments=arguments,
        commit=commit,
        full_commit=full_commit,
        facts=facts,
        device=device,
    )


def _requested_corpus_names(arguments: argparse.Namespace) -> list[str]:
    """Expand the all-corpora selector and preserve unique request order."""
    scoring_view = getattr(arguments, "scoring_view", corpora.STRICT_RAW_VIEW)
    requested = _corpus_request(arguments, scoring_view)
    names = _expand_corpus_request(requested, scoring_view)
    _validate_dedup_corpus_names(names, scoring_view)
    return list(dict.fromkeys(names))


def _corpus_request(arguments: argparse.Namespace, scoring_view: str) -> list[str]:
    if arguments.corpus:
        return arguments.corpus
    if scoring_view == corpora.EXACT_FULL_RECORD_DEDUP_VIEW:
        return ["all"]
    return list(corpora.DEFAULT)


def _expand_corpus_request(requested: list[str], scoring_view: str) -> list[str]:
    if "all" in requested and scoring_view == corpora.EXACT_FULL_RECORD_DEDUP_VIEW:
        return [
            f"newsli-{language}"
            for language in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
        ]
    return list(corpora.CORPORA) if "all" in requested else requested


def _validate_dedup_corpus_names(names: Sequence[str], scoring_view: str) -> None:
    if scoring_view != corpora.EXACT_FULL_RECORD_DEDUP_VIEW:
        return
    allowed = {
        f"newsli-{language}" for language in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
    }
    invalid = [name for name in names if name not in allowed]
    _require_requested_corpus_names(invalid)


def _require_requested_corpus_names(invalid: list[str]) -> None:
    if invalid:
        raise ValueError(
            "NewsLi exact-record dedup cannot score: " + ", ".join(invalid)
        )


if __name__ == "__main__":
    raise SystemExit(main())
