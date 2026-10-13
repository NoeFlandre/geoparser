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
        expected_cell = next(
            (
                cell
                for cell in missing_matrix.MATRIX
                if missing_matrix.cell_name(cell) == loaded.name
            ),
            None,
        )
        if (
            expected_cell is not None
            and loaded.expected_documents == expected_cell.documents
        ):
            actual_gold = corpus.gold_toponym_count(loaded.documents)
            if (len(loaded.documents), actual_gold) != (
                expected_cell.documents,
                expected_cell.gold_toponyms,
            ):
                message = (
                    f"{loaded.name} differs from the frozen matrix: expected "
                    f"{expected_cell.documents} documents/"
                    f"{expected_cell.gold_toponyms} gold spans, found "
                    f"{len(loaded.documents)}/{actual_gold}; no model ran."
                )
                raise CorpusRejectedError(message)
        print(totals_note(loaded))
        return
    raise CorpusRejectedError(_corpus_rejection_message(loaded.name, problems))


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
    resolution_drivers: dict[str, str] = {}
    if runner.RESOLUTION in phases:
        for pipeline_name in pipelines_to_run:
            if pipeline_name in shareable:
                family = missing_matrix.resolver_family(pipeline_name)
                resolution_drivers.setdefault(family, pipeline_name)

    results: dict[str, report.PipelineResult] = {}
    for pipeline_name in pipelines_to_run:
        phase_selection = list(phases)
        family = missing_matrix.resolver_family(pipeline_name)
        driver = resolution_drivers.get(family)
        if (
            pipeline_name in shareable
            and runner.RESOLUTION in phase_selection
            and driver is not None
            and driver != pipeline_name
        ):
            phase_selection.remove(runner.RESOLUTION)
        result = _score_pipeline(run, pipeline_name, phase_selection)
        if (
            pipeline_name in shareable
            and runner.RESOLUTION in phases
            and driver is not None
            and driver != pipeline_name
        ):
            result = _reuse_resolution(run, pipeline_name, driver, result)
        results[pipeline_name] = result
    return [results[name] for name in pipelines_to_run]


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

    target_path = run.output_dir / f"checkpoint-{target_pipeline}.json"
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
            "reused": True,
            "source_pipeline": source_pipeline,
            "documents": len(expected_documents),
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
        ckpt.save(target_path, target_state)

    models = dict(target_result.models)
    models.update(
        {
            name: target_identity.model_pins[name]
            for name in ("resolver", "reranker")
            if name in target_identity.model_pins
        }
    )
    return runner.score(
        target_pipeline,
        run.loaded.documents,
        target_state,
        models=models,
        device=run.device,
    )


def _score_pipeline(
    run: _CorpusRun,
    pipeline_name: str,
    phases: Sequence[str],
) -> report.PipelineResult:
    """Resume, run, and score one pipeline for a corpus."""
    loaded = run.loaded
    arguments = run.arguments
    identity = _run_identity(run, pipeline_name)
    checkpoint_path = run.output_dir / f"checkpoint-{pipeline_name}.json"
    state, reasons = ckpt.load(checkpoint_path, identity)
    for reason in reasons:
        print(f"  {pipeline_name}: starting fresh -- {reason}")

    models: dict[str, str] = {}
    for phase in phases:
        models.update(
            runner.run_phase(
                phase,
                pipeline_name,
                loaded.documents,
                state,
                checkpoint_path,
                runner.PhaseSettings(
                    device=run.device,
                    min_similarity=arguments.min_similarity,
                    chunk_size=arguments.chunk_size,
                ),
            )
        )
        if state.completed(phase) == {
            document.identifier for document in loaded.documents
        }:
            state.phase_provenance.setdefault(
                phase,
                {
                    "reused": False,
                    "source_pipeline": pipeline_name,
                    "documents": len(loaded.documents),
                    "corpus_digest": loaded.digest,
                    "gazetteer": pipelines.GAZETTEER_NAME,
                    "gazetteer_sha256": identity.gazetteer_sha256,
                    "min_similarity": arguments.min_similarity,
                    "code_commit": run.commit,
                    "model_pins": identity.model_pins,
                },
            )
            ckpt.save(checkpoint_path, state)
    for role in ("recognizer", "resolver", "reranker"):
        phase = runner.RECOGNITION if role == "recognizer" else runner.RESOLUTION
        if phase in phases and role in identity.model_pins:
            models.setdefault(role, identity.model_pins[role])
    return runner.score(
        pipeline_name, loaded.documents, state, models=models, device=run.device
    )


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
    names = plan.names
    pipeline_names = plan.pipeline_names
    phases = plan.phases
    arguments = plan.arguments
    commit = plan.commit
    full_commit = plan.full_commit
    facts = plan.facts
    fixed_cells = {
        missing_matrix.cell_name(cell): cell for cell in missing_matrix.MATRIX
    }
    cells = {}
    for name in names:
        spec = corpora.CORPORA[name]
        fixed = fixed_cells.get(name)
        cells[name] = {
            "language": spec.language,
            "expected_documents": fixed.documents if fixed else spec.documents,
            "expected_gold_toponyms": (
                fixed.gold_toponyms if fixed else spec.gold_spans
            ),
            "evaluation_rule": (
                "frozen first-five-corpora matrix"
                if fixed
                else "registered benchmark adapter"
            ),
            "source_split": (
                "complete GeoVirus benchmark corpus"
                if name == "geovirus"
                else "HIPE-2022 v2.1 unmasked test split"
                if name.startswith(("hipe2020-", "newseye-", "topres19th-"))
                else "UniTopRank NewsLi release; registered cap selects up to 500 "
                "valid article identifiers in sorted order"
                if name.startswith("newsli-")
                else "registered corpus adapter"
            ),
            "status": "pending",
        }
    model_runs = {}
    for name in pipeline_names:
        pins = pipelines.pipeline_model_pins(name)
        model_runs[name] = {
            "model_pins": pins,
        }
        if name == pipelines.OTTER_MINILM:
            model_runs[name]["integration_provenance"] = (
                pipelines.otter_integration_provenance()
            )
    lock_path = REPOSITORY_ROOT / "uv.lock"
    return {
        "manifest_version": 1,
        "status": "running",
        "commit": commit,
        "full_commit": full_commit,
        "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "started_at": facts.get("started_at"),
        "environment": facts,
        "protocol": {
            "gazetteer": pipelines.GAZETTEER_NAME,
            "gazetteer_artifact_sha256": facts.get("gazetteer_artifact_sha256"),
            "min_similarity": arguments.min_similarity,
            "limit": arguments.limit,
            "chunk_size": arguments.chunk_size,
            "device": plan.device,
            "accelerator": facts.get("gpu"),
            "phases": list(phases),
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
    all_complete = True
    has_failures = False
    for result in payload["pipelines"]:
        phase_results = {
            runner.RECOGNITION: result["recognition"],
            runner.RESOLUTION: result["resolution"],
        }
        phase_status = {
            phase: ("complete" if phase_results[phase] else "not_completed")
            for phase in phases
        }
        failures = result["failures"]
        has_failures |= bool(failures)
        complete = all(status == "complete" for status in phase_status.values())
        all_complete &= complete
        cell["pipelines"][result["name"]] = {
            "status": "complete" if complete else "incomplete",
            "phase_status": phase_status,
            "model_pins": manifest["protocol"]["pipelines"][result["name"]][
                "model_pins"
            ],
            "model_metadata": (
                missing_matrix.pipeline_model_metadata(result["name"], cell["language"])
                if result["name"] in missing_matrix.MISSING_PIPELINES
                else {}
            ),
            "integration_provenance": (
                pipelines.otter_integration_provenance()
                if result["name"] == pipelines.OTTER_MINILM
                else {}
            ),
            "recognition": result["recognition"],
            "resolution": result["resolution"],
            "phase_elapsed_seconds": result["phase_elapsed_seconds"],
            "phase_provenance": result["phase_provenance"],
            "failure_count": sum(map(len, failures.values())),
            "failures": failures,
        }
    cell["status"] = (
        "complete_with_document_failures"
        if all_complete and has_failures
        else "complete"
        if all_complete
        else "incomplete"
    )


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark on every requested corpus and write the reports."""
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEOPARSER_DB_PATH", str(output_dir / "benchmark.sqlite"))

    requested = arguments.corpus or list(corpora.DEFAULT)
    names = list(corpora.CORPORA) if "all" in requested else requested

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

    pipeline_names = arguments.pipeline or list(pipelines.DEFAULT_PIPELINES)
    manifest_path = output_dir / "run-manifest.json"
    manifest = _run_manifest(
        _ManifestPlan(
            names=list(dict.fromkeys(names)),
            pipeline_names=pipeline_names,
            phases=phases,
            arguments=arguments,
            commit=commit,
            full_commit=full_commit,
            facts=facts,
            device=device,
        )
    )
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
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

    rows: list[dict] = []
    corpus_cache_root = Path(
        os.environ.get("GEOPARSER_CORPUS_CACHE_DIR", str(output_dir))
    )
    for name in dict.fromkeys(names):
        folder = corpus_output_dir(output_dir, name)
        loaded = corpora.load(name, corpus_cache_root / name, limit=arguments.limit)
        try:
            rows += run_corpus(
                loaded, arguments, folder, device=device, commit=commit, facts=facts
            )
        except CorpusRejectedError as error:
            print(error, file=sys.stderr)
            manifest["protocol"]["corpora"][name].update(
                {"status": "rejected", "reason": str(error)}
            )
            manifest["status"] = "rejected"
            _persist_run_manifest(output_dir, manifest)
            return 1
        payload = json.loads(
            (folder / "benchmark-report.json").read_text(encoding="utf-8")
        )
        _record_manifest_cell(manifest, name, payload, phases)
        _persist_run_manifest(output_dir, manifest)
        (output_dir / "summary.md").write_text(
            report.render_summary(rows), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(
                {"commit": commit, "environment": facts, "rows": rows}, indent=2
            ),
            encoding="utf-8",
        )
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
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
