"""
Publish benchmark evidence to a Hugging Face dataset.

    python -m scripts.benchmark.publish --repo-id NoeFlandre/geoparser-benchmark-results

Everything under ``benchmark-evidence/`` is uploaded to ``runs/`` in the
dataset, and the dataset card is regenerated from the reports found there, so
the Hub copy never shows a number the committed evidence does not hold.

The results derive from HIPE-2022, licensed CC-BY-NC-SA 4.0, and share-alike
carries that licence over to the dataset. The corpus text is not uploaded:
reports and checkpoints hold scores, offsets and coordinates, not articles.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

DEFAULT_REPO_ID = "NoeFlandre/geoparser-benchmark-results"
EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "benchmark-evidence"
LICENSE = "cc-by-nc-sa-4.0"
REPORT_NAME = "benchmark-report.json"
BASELINE_PREFIX = "baseline"
# One flat table for the Hub's viewer: left to itself it tries to read every
# JSON file in the repo as one dataset, and reports, checkpoints and
# summaries do not share a schema.
RESULTS_FILE = "results.csv"
MODEL_ROLES = ("recognizer", "resolver", "reranker")
COLUMNS = (
    "run",
    "corpus",
    "language",
    "documents",
    "gold_toponyms",
    "pipeline",
    *MODEL_ROLES,
    "f1",
    "accuracy_at_161km",
    "auc",
    "elapsed_seconds",
    "commit",
)

PIPELINE_DESCRIPTIONS = {
    "upstream": (
        "The original geoparser pipeline: spaCy named-entity recognition "
        "(English model) finds toponyms, and a sentence-transformer "
        "fine-tuned for geocoding ranks GeoNames candidates by similarity "
        "to the toponym in context."
    ),
    "swapped": (
        "Both stages replaced: GLiNER2 (`fastino/gliner2.5-multi-v1`), a "
        "multilingual zero-shot extractor, recognizes toponyms; GeoNames "
        "candidates are ranked by `jinaai/jina-embeddings-v5-text-small` and "
        "re-scored by `jinaai/jina-reranker-v3.5`. Historical only: its "
        "GeoVirus report is kept under `runs/*/baseline-jina/`."
    ),
    "hybrid": (
        "GLiNER2's multilingual zero-shot recognition combined with the "
        "upstream geocoding sentence-transformer resolver."
    ),
}


def collect_rows(evidence_dir: Path) -> list[dict[str, t.Any]]:
    """
    Return one row per run, corpus and pipeline found under the evidence folder.

    Historical reports copied into a run for comparison live in folders named
    ``baseline*`` and are skipped, since the run they came from has its own row.
    """
    rows = []
    for path in sorted(evidence_dir.rglob(REPORT_NAME)):
        relative = path.relative_to(evidence_dir)
        if any(part.startswith(BASELINE_PREFIX) for part in relative.parts):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for pipeline in data["pipelines"]:
            recognition = pipeline.get("recognition") or {}
            resolution = pipeline.get("resolution") or {}
            rows.append(
                {
                    "run": relative.parts[0],
                    "corpus": data["corpus"],
                    "language": data.get("language", "en"),
                    "documents": data["documents"],
                    "gold_toponyms": data["gold_toponyms"],
                    "pipeline": pipeline["name"],
                    **{
                        role: (pipeline.get("models") or {}).get(role, "")
                        for role in MODEL_ROLES
                    },
                    "f1": recognition.get("f1"),
                    "accuracy_at_161km": resolution.get("accuracy_at_161km"),
                    "auc": resolution.get("auc"),
                    "elapsed_seconds": pipeline["elapsed_seconds"],
                    "commit": data["commit"],
                }
            )
    return rows


def _number(value: float | None) -> str:
    """Format a score, or a dash for a phase that did not run."""
    return "-" if value is None else f"{value:.3f}"


def render_csv(rows: Sequence[dict[str, t.Any]]) -> str:
    """Return the rows as the CSV the dataset viewer shows."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in COLUMNS})
    return buffer.getvalue()


def _pipeline_section(rows: Sequence[dict[str, t.Any]]) -> list[str]:
    """Describe each pipeline, with the models it was run with."""
    lines = ["## Pipelines", ""]
    for name, description in PIPELINE_DESCRIPTIONS.items():
        combos = sorted(
            {
                tuple(row.get(role, "") for role in MODEL_ROLES)
                for row in rows
                if row["pipeline"] == name
            }
        )
        lines.append(f"- **{name}**: {description}")
        for combo in combos:
            models = ", ".join(
                f"{role} `{model}`"
                for role, model in zip(MODEL_ROLES, combo, strict=True)
                if model
            )
            lines.append(f"  - Models: {models}")
    lines += [
        "",
        "Every pipeline resolves against the same GeoNames gazetteer.",
        "",
    ]
    return lines


def render_card(rows: Sequence[dict[str, t.Any]]) -> str:
    """Return the dataset card, with a results table built from ``rows``."""
    lines = [
        "---",
        f"license: {LICENSE}",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        f"    path: {RESULTS_FILE}",
        "language: [en, de, fr, fi, sv]",
        "pretty_name: Geoparser benchmark results",
        "tags: [geoparsing, toponym-resolution, benchmark]",
        "---",
        "",
        "# Geoparser benchmark results",
        "",
        "Scores from [geoparser](https://github.com/NoeFlandre/geoparser)'s",
        "benchmark harness (`python -m scripts.benchmark`) on GeoVirus and the",
        "HIPE-2022 test splits. Recognition is exact-match F1 on each pipeline's",
        "own spans; resolution is scored on the gold spans, so the resolvers are",
        "compared on the same toponyms. AUC is lower-is-better and normalised by",
        "the harness itself.",
        "",
        f"`{RESULTS_FILE}` is the table below, one row per run, corpus and",
        "pipeline; `runs/` holds each run's reports, checkpoints and",
        "Grid'5000 job logs.",
        "",
        *_pipeline_section(rows),
        "## Results",
        "",
        "| Run | Corpus | Lang | Docs | Gold | Pipeline | Rec F1 | Acc@161km "
        "| AUC | Seconds | Commit |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| {row['run']} | {row['corpus']} | {row['language']} "
        f"| {row['documents']} | {row['gold_toponyms']} | {row['pipeline']} "
        f"| {_number(row['f1'])} | {_number(row['accuracy_at_161km'])} "
        f"| {_number(row['auc'])} | {row['elapsed_seconds']:.1f} | {row['commit']} |"
        for row in rows
    ]
    lines += [
        "",
        "## Licence",
        "",
        "HIPE-2022 is CC-BY-NC-SA 4.0 (Ehrmann et al.), so these derived",
        "results are shared under the same terms. GeoVirus is from Gritta et",
        "al., *A Pragmatic Guide to Geoparsing Evaluation*.",
        "",
    ]
    return "\n".join(lines)


def publish(evidence_dir: Path, repo_id: str, *, api: t.Any) -> None:
    """
    Upload the evidence folder and a regenerated card to a public dataset.

    Raises:
        ValueError: When the folder holds no reports
    """
    rows = collect_rows(evidence_dir)
    if not rows:
        raise ValueError(f"no {REPORT_NAME} under {evidence_dir}")
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(evidence_dir),
        path_in_repo="runs",
        commit_message="Upload benchmark evidence",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=render_csv(rows).encode("utf-8"),
        path_in_repo=RESULTS_FILE,
        commit_message="Regenerate results table",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=render_card(rows).encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Regenerate dataset card",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Publish the committed evidence to the Hub."""
    parser = argparse.ArgumentParser(prog="python -m scripts.benchmark.publish")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    arguments = parser.parse_args(argv)

    from huggingface_hub import HfApi

    publish(arguments.evidence_dir, arguments.repo_id, api=HfApi())
    print(f"https://huggingface.co/datasets/{arguments.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
