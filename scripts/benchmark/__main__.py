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
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import corpora, corpus, pipelines, provenance, report, runner

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
    """
    documents = loaded.documents
    gold_count = corpus.gold_toponym_count(documents)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{loaded.name} ({loaded.language}): {len(documents)} documents, "
        f"{gold_count} gold toponyms, corpus {loaded.digest}, commit {commit}"
    )

    phases = arguments.phase or [runner.RECOGNITION, runner.RESOLUTION]
    results = []
    for pipeline in arguments.pipeline or list(pipelines.DEFAULT_PIPELINES):
        identity = ckpt.RunIdentity(
            pipeline=pipeline,
            corpus_digest=loaded.digest,
            gazetteer=pipelines.GAZETTEER_NAME,
            min_similarity=arguments.min_similarity,
            limit=arguments.limit,
            commit=commit,
        )
        checkpoint_path = output_dir / f"checkpoint-{pipeline}.json"
        state, reasons = ckpt.load(checkpoint_path, identity)
        for reason in reasons:
            print(f"  {pipeline}: starting fresh -- {reason}")

        models: dict[str, str] = {}
        for phase in phases:
            models.update(
                runner.run_phase(
                    phase,
                    pipeline,
                    documents,
                    state,
                    checkpoint_path,
                    device=device,
                    min_similarity=arguments.min_similarity,
                    chunk_size=arguments.chunk_size,
                )
            )
        results.append(
            runner.score(pipeline, documents, state, models=models, device=device)
        )

    text = report.render_markdown(
        results,
        corpus_name=loaded.name,
        documents=len(documents),
        gold_toponyms=gold_count,
        gazetteer=pipelines.GAZETTEER_NAME,
        min_similarity=arguments.min_similarity,
    )
    (output_dir / "benchmark-report.md").write_text(text, encoding="utf-8")
    (output_dir / "benchmark-report.json").write_text(
        json.dumps(
            {
                "corpus": loaded.name,
                "language": loaded.language,
                "corpus_digest": loaded.digest,
                "documents": len(documents),
                "gold_toponyms": gold_count,
                "gazetteer": pipelines.GAZETTEER_NAME,
                "min_similarity": arguments.min_similarity,
                "commit": commit,
                "environment": facts,
                "pipelines": [
                    {
                        "name": result.name,
                        "device": result.device,
                        "models": result.models,
                        "recognition": result.recognition,
                        "resolution": result.resolution,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
                    for result in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(text)
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
        }
        for result in results
    ]


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
    facts = provenance.environment(os.environ.get("OAR_JOB_ID"))
    print(f"device: {pipelines.describe_device(device)}")

    rows: list[dict] = []
    for name in dict.fromkeys(names):
        folder = corpus_output_dir(output_dir, name)
        loaded = corpora.load(name, folder, limit=arguments.limit)
        rows += run_corpus(
            loaded, arguments, folder, device=device, commit=commit, facts=facts
        )
        (output_dir / "summary.md").write_text(
            report.render_summary(rows), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(
                {"commit": commit, "environment": facts, "rows": rows}, indent=2
            ),
            encoding="utf-8",
        )
    print(report.render_summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
