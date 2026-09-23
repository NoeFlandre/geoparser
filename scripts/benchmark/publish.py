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

from scripts.benchmark.chart import render_bar_chart
from scripts.benchmark.corpora import CORPORA

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
# Metric to the path its chart is uploaded to, in the order they are shown.
CHART_FILES = {
    "accuracy_at_161km": "charts/resolution-acc161.svg",
    "auc": "charts/resolution-auc.svg",
    "median_error_km": "charts/resolution-median-error.svg",
    "f1": "charts/recognition-f1.svg",
}
CHART_TITLES = {
    "accuracy_at_161km": "Resolution: Acc@161km (higher is better)",
    "auc": "Resolution: AUC of the error curve (lower is better)",
    "median_error_km": "Resolution: median error, log scale (lower is better)",
    "f1": "Recognition: F1 (higher is better)",
}
CHART_SCALES = {"median_error_km": "log"}
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
    "median_error_km",
    "mean_error_km",
    "elapsed_seconds",
    "commit",
    "started_at",
)

# name: (recognition, resolution, what it is), one line each on the card.
PIPELINE_DESCRIPTIONS = {
    "upstream": (
        "spaCy `en_core_web_sm`",
        "MiniLM `dguzh/geo-all-MiniLM-L6-v2`",
        "The original geoparser pipeline.",
    ),
    "swapped": (
        "GLiNER2 `fastino/gliner2.5-multi-v1`",
        "Jina embeddings v5 + Jina reranker v3.5",
        "Historical, GeoVirus only (`runs/*/baseline-jina/`).",
    ),
    "hybrid": (
        "GLiNER2 `fastino/gliner2.5-multi-v1`",
        "MiniLM `dguzh/geo-all-MiniLM-L6-v2`",
        "Multilingual recognition, upstream resolution.",
    ),
    "prior": (
        "GLiNER2 `fastino/gliner2.5-multi-v1`",
        "MiniLM + inflection fallback + population prior",
        "hybrid, with exact-match misses retried with the ending trimmed "
        "and a small log-population prior; no extra model.",
    ),
}

# Corpus-name prefix: (display name, one-line description).
BENCHMARKS = {
    "geovirus": (
        "GeoVirus",
        "WikiNews articles on epidemics (Gritta et al., 2018).",
    ),
    "hipe2020": (
        "HIPE-2020",
        "Historical Swiss, Luxembourgish and American newspapers, OCR.",
    ),
    "newseye": (
        "NewsEye",
        "Historical European newspapers, OCR (HIPE-2022).",
    ),
    "topres19th": (
        "TopRes19th",
        "19th-century British newspapers, OCR (HIPE-2022).",
    ),
    "newsli": (
        "NewsLi",
        "Wikinews linked to GeoNames (UniTopRank, Hu et al., 2026); "
        "first 500 articles per language.",
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
                    "median_error_km": resolution.get("median_error_km"),
                    "mean_error_km": resolution.get("mean_error_km"),
                    "elapsed_seconds": pipeline["elapsed_seconds"],
                    "commit": data["commit"],
                    "started_at": data.get("environment", {}).get("started_at", ""),
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


def _benchmark_section(rows: Sequence[dict[str, t.Any]]) -> list[str]:
    """One line per corpus family that ran, with its languages and size."""
    families: dict[str, dict[str, t.Any]] = {}
    for (corpus, _), row in latest_results(rows).items():
        prefix = corpus.split("-")[0]
        family = families.setdefault(prefix, {"languages": {}, "sizes": {}})
        family["languages"][row["language"]] = None
        family["sizes"][corpus] = (row["documents"], row["gold_toponyms"])
    lines = [
        "## Benchmarks",
        "",
        "| Benchmark | Languages | Docs | Toponyms | Source |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for prefix, family in families.items():
        name, description = BENCHMARKS.get(prefix, (prefix, ""))
        documents = sum(size[0] for size in family["sizes"].values())
        toponyms = sum(size[1] for size in family["sizes"].values())
        languages = ", ".join(family["languages"])
        lines.append(
            f"| {name} | {languages} | {documents} | {toponyms} | {description} |"
        )
    return [*lines, ""]


def _pipeline_section() -> list[str]:
    """One row per pipeline: its two stages and what it is."""
    lines = [
        "## Pipelines",
        "",
        "| Pipeline | Recognition | Resolution | Notes |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| **{name}** | {recognition} | {resolution} | {note} |"
        for name, (recognition, resolution, note) in PIPELINE_DESCRIPTIONS.items()
    ]
    return [*lines, "", "All resolve against GeoNames.", ""]


def latest_results(
    rows: Sequence[dict[str, t.Any]],
) -> dict[tuple[str, str], dict[str, t.Any]]:
    """
    Keep, for every corpus and pipeline, the row of the most recent run.

    Runs are ordered by when they started rather than by folder name, so a
    partial run cannot shadow the complete rerun that followed it.

    Returns:
        (corpus, pipeline) to its latest row, corpus names lower-cased
    """
    latest: dict[tuple[str, str], dict[str, t.Any]] = {}
    for row in sorted(rows, key=lambda row: row.get("started_at", "")):
        latest[(row["corpus"].lower(), row["pipeline"])] = row
    return latest


def mark_best(
    values: Sequence[float | None], *, higher_is_better: bool = True
) -> list[str]:
    """
    Format scores, bolding the best and underlining the second best.

    Scores are compared as displayed, to three decimals, so values that look
    equal share their emphasis.

    Returns:
        One Markdown cell per value, a dash for a missing one
    """
    shown = [None if value is None else f"{value:.3f}" for value in values]
    ranked = sorted(
        {cell for cell in shown if cell is not None},
        key=float,
        reverse=higher_is_better,
    )
    emphasis = dict(zip(ranked, ("**{}**", "<u>{}</u>"), strict=False))
    return [
        "-" if cell is None else emphasis.get(cell, "{}").format(cell) for cell in shown
    ]


def _corpus_order(corpora: set[str]) -> list[str]:
    """Registry order first, then any corpus the registry no longer lists."""
    known = [name for name in CORPORA if name in corpora]
    return known + sorted(corpora - set(known))


def _leaderboard(
    latest: dict[tuple[str, str], dict[str, t.Any]],
    metric: str,
    title: str,
    *,
    higher_is_better: bool = True,
) -> list[str]:
    """One compact table: a row per corpus, a column per pipeline."""
    pipelines = [
        name
        for name in PIPELINE_DESCRIPTIONS
        if any(pipeline == name for _, pipeline in latest)
    ]
    corpora = _corpus_order({corpus for corpus, _ in latest})
    lines = [
        f"### {title}",
        "",
        "| Corpus | Lang | " + " | ".join(pipelines) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in pipelines) + " |",
    ]
    for corpus in corpora:
        cells = [latest.get((corpus, pipeline)) for pipeline in pipelines]
        language = next(row["language"] for row in cells if row)
        marked = mark_best(
            [row[metric] if row else None for row in cells],
            higher_is_better=higher_is_better,
        )
        lines.append(f"| {corpus} | {language} | " + " | ".join(marked) + " |")
    return [*lines, ""]


def render_charts(rows: Sequence[dict[str, t.Any]]) -> dict[str, str]:
    """Return each chart's upload path and SVG, from the latest results."""
    latest = latest_results(rows)
    corpora = _corpus_order({corpus for corpus, _ in latest})
    return {
        path: render_bar_chart(
            latest,
            metric,
            CHART_TITLES[metric],
            corpora=corpora,
            scale=CHART_SCALES.get(metric, "unit"),
        )
        for metric, path in CHART_FILES.items()
    }


def _chart_links(repo_id: str | None) -> list[str]:
    """Image links to the uploaded charts, or nothing when the repo is unknown."""
    if repo_id is None:
        return []
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    return [
        f"![{CHART_TITLES[metric]}]({base}/{path})\n"
        for metric, path in CHART_FILES.items()
    ]


def render_card(rows: Sequence[dict[str, t.Any]], *, repo_id: str | None = None) -> str:
    """Return the dataset card, with compact results tables built from ``rows``."""
    latest = latest_results(rows)
    lines = [
        "---",
        f"license: {LICENSE}",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        f"    path: {RESULTS_FILE}",
        "language: [en, de, fr, fi, sv, ar, es, fa, ja, pl, ro, sr, ta, tr, uk]",
        "pretty_name: Geoparser benchmark results",
        "tags: [geoparsing, toponym-resolution, benchmark]",
        "---",
        "",
        "# Geoparser benchmark results",
        "",
        "Geoparsing pipelines from [geoparser]"
        "(https://github.com/NoeFlandre/geoparser) scored on English and",
        "multilingual corpora, run on Grid'5000 (one Tesla T4).",
        "",
        *_benchmark_section(rows),
        *_pipeline_section(),
        "## Results",
        "",
        "Latest run per corpus and pipeline. **Resolution** is the share of",
        "gold toponyms placed within 161 km, scored on the gold spans so the",
        "resolvers see the same toponyms. **Recognition** is exact-match F1.",
        "AUC summarises the whole log-scaled error curve, lower being better.",
        "**Bold** is best on a corpus, <u>underlined</u> second.",
        "",
        *_chart_links(repo_id),
        *_leaderboard(latest, "accuracy_at_161km", "Resolution: Acc@161km"),
        *_leaderboard(
            latest, "auc", "Resolution: AUC (lower is better)", higher_is_better=False
        ),
        *_leaderboard(latest, "f1", "Recognition: F1"),
        "## Files",
        "",
        f"`{RESULTS_FILE}` (the viewer) has every run with AUC, errors and",
        "runtime; `runs/` holds reports, checkpoints and job logs.",
        "",
        "## Licence",
        "",
        "CC-BY-NC-SA 4.0, following HIPE-2022 (Ehrmann et al.).",
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
    for path, svg in render_charts(rows).items():
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=svg.encode("utf-8"),
            path_in_repo=path,
            commit_message="Regenerate results chart",
        )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=render_card(rows, repo_id=repo_id).encode("utf-8"),
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
