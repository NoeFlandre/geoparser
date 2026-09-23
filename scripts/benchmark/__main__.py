"""
Command line entry point for the GeoVirus benchmark.

    python -m scripts.benchmark --limit 10 --device auto
    python -m scripts.benchmark --output-dir ~/geoparser-bench --device cuda

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
from scripts.benchmark import corpus, pipelines, provenance, report, runner

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
        help="Pipeline to run; repeat for several. Defaults to both.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N articles"
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its report."""
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEOPARSER_DB_PATH", str(output_dir / "benchmark.sqlite"))

    corpus_path = corpus.download_corpus(output_dir / f"{corpus.CORPUS_NAME}.xml")
    digest = corpus.corpus_digest(corpus_path)
    documents = corpus.parse_corpus(corpus_path, limit=arguments.limit)
    gold_count = corpus.gold_toponym_count(documents)

    device = pipelines.resolve_device(arguments.device)
    commit = provenance.source_commit(REPOSITORY_ROOT)
    facts = provenance.environment(os.environ.get("OAR_JOB_ID"))

    print(
        f"{len(documents)} documents, {gold_count} gold toponyms, "
        f"corpus {digest}, commit {commit}"
    )
    print(f"device: {pipelines.describe_device(device)}")

    phases = arguments.phase or [runner.RECOGNITION, runner.RESOLUTION]
    results = []
    for pipeline in arguments.pipeline or list(pipelines.PIPELINES):
        identity = ckpt.RunIdentity(
            pipeline=pipeline,
            corpus_digest=digest,
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
        corpus_name=corpus.CORPUS_NAME,
        documents=len(documents),
        gold_toponyms=gold_count,
        gazetteer=pipelines.GAZETTEER_NAME,
        min_similarity=arguments.min_similarity,
    )
    (output_dir / "benchmark-report.md").write_text(text, encoding="utf-8")
    (output_dir / "benchmark-report.json").write_text(
        json.dumps(
            {
                "corpus": corpus.CORPUS_NAME,
                "corpus_digest": digest,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
