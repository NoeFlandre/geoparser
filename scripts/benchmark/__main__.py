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
from scripts._io import PROJECT_ROOT, write_json_atomic
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


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.benchmark", description=__doc__
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        choices=list(pipelines.PIPELINES),
        help=(
            "Pipeline to run; repeat for several. Defaults to all but the ablations."
        ),
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
        default=5,
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
    if loaded.limit is not None:
        return
    expected = next(
        (
            cell
            for cell in missing_matrix.MATRIX
            if missing_matrix.cell_name(cell) == loaded.name
        ),
        None,
    )
    if expected is not None:
        _check_known_frozen_totals(loaded, expected)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{loaded.name} ({loaded.language}): {len(documents)} documents, "
        f"{gold_count} gold toponyms, corpus {loaded.digest}, commit {commit}"
    )
    run = _CorpusRun(loaded, arguments, output_dir, device, commit, facts)
    results = _score_requested_pipelines(run)

    text = report.render_markdown(
        results,
        corpus_name=loaded.name,
        documents=len(documents),
        gold_toponyms=gold_count,
        gazetteer=pipelines.GAZETTEER_NAME,
        min_similarity=arguments.min_similarity,
    )
    _write_corpus_reports(
        output_dir,
        text,
        _corpus_report_payload(run, gold_count, results),
    )
    print(text)
    return _summary_rows(run, gold_count, results)


def _score_requested_pipelines(
    run: _CorpusRun,
) -> list[report.PipelineResult]:
    """Run each selected pipeline over the requested corpus phases."""
    pipelines_to_run = run.arguments.pipeline or list(pipelines.DEFAULT_PIPELINES)
    phases = run.arguments.phase or [runner.RECOGNITION, runner.RESOLUTION]
    shareable = {pipelines.XLM_MINILM, pipelines.OTTER_MINILM}
    drivers = _resolution_drivers(pipelines_to_run, phases, shareable)
    return [
        _score_one_requested_pipeline(run, name, phases, shareable, drivers)
        for name in pipelines_to_run
    ]


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


def _phases_for_pipeline(
    name: str, phases: Sequence[str], shareable: set[str], drivers: dict[str, str]
) -> list[str]:
    """Skip resolution when an equivalent recognizer pipeline already ran it."""
    selected = list(phases)
    family = missing_matrix.resolver_family(name)
    if name in shareable and drivers.get(family, name) != name:
        selected = [phase for phase in selected if phase != runner.RESOLUTION]
    return selected


def _score_one_requested_pipeline(
    run: _CorpusRun,
    name: str,
    phases: Sequence[str],
    shareable: set[str],
    drivers: dict[str, str],
) -> report.PipelineResult:
    """Run one pipeline and attach matching reusable resolution predictions."""
    driver = drivers.get(missing_matrix.resolver_family(name))
    result = _score_pipeline(
        run, name, _phases_for_pipeline(name, phases, shareable, drivers)
    )
    if name in shareable and runner.RESOLUTION in phases and driver and driver != name:
        return _reuse_resolution(run, name, driver, result)
    return result


def _run_identity(run: _CorpusRun, pipeline_name: str) -> ckpt.RunIdentity:
    """Build the complete checkpoint identity, including immutable model pins."""
    return ckpt.RunIdentity(
        pipeline=pipeline_name,
        corpus_digest=run.loaded.digest,
        gazetteer=pipelines.GAZETTEER_NAME,
        gazetteer_sha256=run.facts.get("gazetteer_artifact_sha256"),
        min_similarity=run.arguments.min_similarity,
        limit=run.arguments.limit,
        commit=run.commit,
        model_pins=pipelines.pipeline_model_pins(pipeline_name),
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
            for name in ("resolver", "reranker")
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
            _phase_provenance(run, pipeline_name, identity),
        )
        ckpt.save(checkpoint_path, state)
    return models


def _phase_provenance(
    run: _CorpusRun, pipeline_name: str, identity: ckpt.RunIdentity
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
        "model_pins": identity.model_pins,
    }


def _complete_phase_model_pins(
    models: dict[str, str], phases: Sequence[str], model_pins: dict[str, str]
) -> dict[str, str]:
    """Keep immutable model identities in scores when checkpoints were resumed."""
    for role in ("recognizer", "resolver", "reranker"):
        phase = runner.RECOGNITION if role == "recognizer" else runner.RESOLUTION
        if phase in phases and role in model_pins:
            models.setdefault(role, model_pins[role])
    return models


def _corpus_report_payload(
    run: _CorpusRun,
    gold_count: int,
    results: Sequence[report.PipelineResult],
) -> dict:
    """Return the metadata and pipeline measurements stored as JSON."""
    loaded = run.loaded
    return {
        "corpus": loaded.name,
        "language": loaded.language,
        "corpus_digest": loaded.digest,
        "limit": run.arguments.limit,
        "documents": len(loaded.documents),
        "gold_toponyms": gold_count,
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
                    if result.name in missing_matrix.MISSING_PIPELINES
                    else {}
                ),
                "integration_provenance": (
                    pipelines.otter_integration_provenance()
                    if result.name == pipelines.OTTER_MINILM
                    else {}
                ),
            }
            for result in results
        ],
    }


def _write_corpus_reports(output_dir: Path, markdown: str, payload: dict) -> None:
    """Persist the human and machine-readable reports side by side."""
    (output_dir / "benchmark-report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "benchmark-report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _summary_rows(
    run: _CorpusRun,
    gold_count: int,
    results: Sequence[report.PipelineResult],
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
    cells = _manifest_corpus_cells(plan.names)
    model_runs = _manifest_pipeline_pins(plan.pipeline_names)
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


def _manifest_corpus_cell(name: str, fixed: Any) -> dict:
    """Describe expected counts and split rules for one requested corpus."""
    spec = corpora.CORPORA[name]
    return {
        "language": spec.language,
        "expected_documents": fixed.documents if fixed else spec.documents,
        "expected_gold_toponyms": fixed.gold_toponyms if fixed else spec.gold_spans,
        "evaluation_rule": (
            "frozen first-five-corpora matrix"
            if fixed
            else "registered benchmark adapter"
        ),
        "source_split": _manifest_source_split(name),
        "status": "pending",
    }


def _manifest_corpus_cells(names: Sequence[str]) -> dict[str, dict]:
    """Describe each selected corpus against the frozen matrix or registry."""
    fixed_cells = {
        missing_matrix.cell_name(cell): cell for cell in missing_matrix.MATRIX
    }
    return {name: _manifest_corpus_cell(name, fixed_cells.get(name)) for name in names}


def _manifest_pipeline_pins(pipeline_names: Sequence[str]) -> dict[str, dict]:
    """Record immutable model pins and retained integration evidence."""
    model_runs = {}
    for name in pipeline_names:
        model_runs[name] = {"model_pins": pipelines.pipeline_model_pins(name)}
        if name == pipelines.OTTER_MINILM:
            model_runs[name]["integration_provenance"] = (
                pipelines.otter_integration_provenance()
            )
    return model_runs


def _assemble_run_manifest(plan: _ManifestPlan, cells: dict, model_runs: dict) -> dict:
    """Assemble protocol, environment and experiment inputs into a manifest."""
    lock_path = REPOSITORY_ROOT / "uv.lock"
    return {
        "manifest_version": 1,
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
            "accelerator": plan.facts.get("gpu"),
            "phases": list(plan.phases),
            "pipelines": model_runs,
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
            "environment": payload["environment"],
            "pipelines": {},
        }
    )
    for result in payload["pipelines"]:
        cell["pipelines"][result["name"]] = _manifest_pipeline_cell(
            result, manifest, cell["language"], phases
        )
    records = list(cell["pipelines"].values())
    cell["status"] = _manifest_cell_status(records)


def _manifest_phase_status(result: dict, phases: Sequence[str]) -> dict[str, str]:
    """Mark each requested metric phase as complete or not completed."""
    metrics = {
        runner.RECOGNITION: result["recognition"],
        runner.RESOLUTION: result["resolution"],
    }
    return {
        phase: "complete" if metrics[phase] else "not_completed" for phase in phases
    }


def _manifest_pipeline_cell(
    result: dict, manifest: dict, language: str, phases: Sequence[str]
) -> dict:
    """Serialize one pipeline measurement and its pinned model evidence."""
    name = result["name"]
    phase_status = _manifest_phase_status(result, phases)
    failures = result["failures"]
    complete = all(status == "complete" for status in phase_status.values())
    return {
        "status": "complete" if complete else "incomplete",
        "phase_status": phase_status,
        "model_pins": manifest["protocol"]["pipelines"][name]["model_pins"],
        "model_metadata": (
            missing_matrix.pipeline_model_metadata(name, language)
            if name in missing_matrix.MISSING_PIPELINES
            else {}
        ),
        "integration_provenance": (
            pipelines.otter_integration_provenance()
            if name == pipelines.OTTER_MINILM
            else {}
        ),
        "recognition": result["recognition"],
        "resolution": result["resolution"],
        "phase_elapsed_seconds": result["phase_elapsed_seconds"],
        "phase_provenance": result["phase_provenance"],
        "failure_count": sum(map(len, failures.values())),
        "failures": failures,
    }


def _manifest_cell_status(records: Sequence[dict]) -> str:
    """Summarize complete pipeline rows, preserving document failures."""
    all_complete = all(record["status"] == "complete" for record in records)
    if not all_complete:
        return "incomplete"
    if any(record["failure_count"] for record in records):
        return "complete_with_document_failures"
    return "complete"


def _persist_run_manifest(output_dir: Path, manifest: dict) -> None:
    """Write the run manifest atomically so walltime stops keep prior cells."""
    write_json_atomic(output_dir / "run-manifest.json", manifest)


def _manifest_identity(manifest: dict) -> dict:
    """Select immutable run inputs, excluding progress and measured results."""
    protocol = manifest["protocol"]
    return {
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
                "phases",
                "pipelines",
            )
        }
        | {
            "corpora": {
                name: {
                    key: value[key]
                    for key in (
                        "language",
                        "expected_documents",
                        "expected_gold_toponyms",
                        "evaluation_rule",
                        "source_split",
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
        loaded = corpora.load(name, cache_root / name, limit=execution.arguments.limit)
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
            execution.manifest["protocol"]["corpora"][name].update(
                {"status": "rejected", "reason": str(error)}
            )
            execution.manifest["status"] = "rejected"
            _persist_run_manifest(output_dir, execution.manifest)
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
    (execution.output_dir / "summary.md").write_text(
        report.render_summary(rows), encoding="utf-8"
    )
    (execution.output_dir / "summary.json").write_text(
        json.dumps(
            {"commit": execution.commit, "environment": execution.facts, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )


def _finish_run_manifest(output_dir: Path, manifest: dict, rows: list[dict]) -> None:
    """Mark the run status, persist its completion time and render final scores."""
    manifest["status"] = (
        "complete"
        if all(
            cell["status"] in {"complete", "complete_with_document_failures"}
            for cell in manifest["protocol"]["corpora"].values()
        )
        else "incomplete"
    )
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
    requested = arguments.corpus or list(corpora.DEFAULT)
    names = list(corpora.CORPORA) if "all" in requested else requested
    return list(dict.fromkeys(names))


if __name__ == "__main__":
    raise SystemExit(main())
