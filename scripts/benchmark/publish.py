"""
Publish benchmark evidence to a Hugging Face dataset.

    python -m scripts.benchmark.publish --repo-id NoeFlandre/geoparser-benchmark-results

Everything under ``benchmark-evidence/`` is uploaded to ``runs/`` in the
dataset, and the dataset card is regenerated from the reports found there, so
the Hub copy never shows a number the committed evidence does not hold.

The corpora carry different terms: HIPE-2022 is CC BY-NC-SA 4.0, while the
UniTopRank release containing GeoVirus and NewsLi is Apache-2.0. The corpus
text is not uploaded; reports and checkpoints hold scores, offsets and
coordinates, not articles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import typing as t
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts._cli import EXIT_OK
from scripts._io import PROJECT_ROOT
from scripts.benchmark import missing_matrix, pipelines
from scripts.benchmark.chart import render_bar_chart
from scripts.benchmark.corpora import (
    CORPORA,
    EXACT_FULL_RECORD_DEDUP_VIEW,
    STRICT_RAW_VIEW,
)
from scripts.benchmark.pipelines import HYBRID, PRIOR_SETTINGS
from scripts.benchmark.report import _number

DEFAULT_REPO_ID = "NoeFlandre/geoparser-benchmark-results"
EVIDENCE_DIR = PROJECT_ROOT / "benchmark-evidence"
LICENSE = "other"
LICENSE_NAME = "Mixed corpus terms; see the dataset card"
HIPE_DATA_URL = "https://github.com/hipe-eval/HIPE-2022-data"
UNITOPRANK_DATA_URL = "https://figshare.com/articles/software/UniTopRank/30445541"
REPORT_NAME = "benchmark-report.json"
BASELINE_PREFIX = "baseline"
CURRENT_PREFIX = "current/v2"
ARCHIVE_PREFIX = "archive/snapshots"
SCORING_VIEWS = (STRICT_RAW_VIEW, EXACT_FULL_RECORD_DEDUP_VIEW)
PRIVATE_EVIDENCE_KEYS = frozenset(
    {"text", "article", "raw_output", "raw_text", "source_text", "message"}
)
INSPECTION_RECORD_FIELDS = frozenset(
    {
        "schema",
        "pipeline",
        "scoring_view",
        "document_id",
        "reference",
        "resolution_input",
        "prediction",
        "phase_status",
        "failures",
    }
)
INSPECTION_REFERENCE_FIELDS = frozenset(
    {"start", "end", "name", "latitude", "longitude"}
)
INSPECTION_PREDICTION_FIELDS = frozenset(
    # The enclosing inspection record carries document_id once.
    {"start", "end", "identifier", "latitude", "longitude"}
)
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
    "scoring_view",
    "run",
    "corpus",
    "language",
    "documents",
    "gold_toponyms",
    "scoring_spans",
    "run_limit",
    "run_partial",
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
    "gliner-minilm": (
        "`fastino/gliner2.5-multi-v1`",
        "`dguzh/geo-all-MiniLM-L6-v2`",
        "GLiNER recognition with MiniLM resolution.",
    ),
    "gliner-jina": (
        "`fastino/gliner2.5-multi-v1`",
        "Jina embeddings v5 + reranker v3.5",
        "GLiNER recognition with the Jina resolver stack.",
    ),
    "xlm-minilm": (
        "`Davlan/xlm-roberta-base-ner-hrl`",
        "`dguzh/geo-all-MiniLM-L6-v2`",
        "XLM-R recognition with MiniLM resolution.",
    ),
    "xlm-jina": (
        "`Davlan/xlm-roberta-base-ner-hrl`",
        "Jina embeddings v5 + reranker v3.5",
        "XLM-R recognition with the Jina resolver stack.",
    ),
    "otter-minilm": (
        "`whoisjones/otter-cross-mmbert`",
        "`dguzh/geo-all-MiniLM-L6-v2`",
        "Otter recognition with MiniLM resolution.",
    ),
    "otter-jina": (
        "`whoisjones/otter-cross-mmbert`",
        "Jina embeddings v5 + reranker v3.5",
        "Otter recognition with the Jina resolver stack.",
    ),
}

# The pipelines the main tables and charts compare; the ablations get their
# own section.
MAIN_PIPELINES = missing_matrix.CURRENT_PIPELINES

# Hybrid, then each ablation variant; the factorial cells are the first four.
ABLATION_PIPELINES: tuple[str, ...] = ()
SWEEP_PIPELINES: tuple[str, ...] = ()
ABLATION_CHART = "charts/ablation-acc161.svg"

# Corpus-name prefix: (display name, one-line description).
BENCHMARKS = {
    "geovirus": (
        "GeoVirus",
        "WikiNews epidemics (Gritta et al.; UniTopRank release, Apache-2.0).",
    ),
    "hipe2020": (
        "HIPE-2020",
        "Historical newspapers and commentary (HIPE-2022, CC BY-NC-SA 4.0).",
    ),
    "newseye": (
        "NewsEye",
        "Historical European newspapers (HIPE-2022, CC BY-NC-SA 4.0).",
    ),
    "topres19th": (
        "TopRes19th",
        "19th-century British newspapers (HIPE-2022, CC BY-NC-SA 4.0).",
    ),
    "newsli": (
        "NewsLi",
        "Wikinews linked to GeoNames (UniTopRank, Apache-2.0); "
        "first 500 articles per language.",
    ),
}


def collect_rows(evidence_dir: Path) -> list[dict[str, t.Any]]:
    """
    Return one row per run, corpus and pipeline found under the evidence folder.

    Historical reports copied into a run for comparison live in folders named
    ``baseline*`` and are skipped, since the run they came from has its own row.
    """
    rows: list[dict[str, t.Any]] = []
    for path in sorted(evidence_dir.rglob(REPORT_NAME)):
        relative = path.relative_to(evidence_dir)
        if _is_baseline_path(relative):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_report_rows(relative, data))
    return rows


def _is_baseline_path(relative: Path) -> bool:
    """Whether a report is a copied historical baseline, not a new run."""
    return any(part.startswith(BASELINE_PREFIX) for part in relative.parts)


def _report_rows(relative: Path, data: dict[str, t.Any]) -> list[dict[str, t.Any]]:
    """Flatten the pipelines from one report into viewer rows."""
    return [_pipeline_row(relative, data, pipeline) for pipeline in data["pipelines"]]


def _pipeline_row(
    relative: Path, data: dict[str, t.Any], pipeline: dict[str, t.Any]
) -> dict[str, t.Any]:
    """Combine one pipeline's scores with corpus and run provenance."""
    recognition = pipeline.get("recognition") or {}
    resolution = pipeline.get("resolution") or {}
    models = pipeline.get("models") or {}
    return {
        "scoring_view": data.get("scoring_view", "legacy-unscoped"),
        "run": relative.parts[0],
        "corpus": data["corpus"],
        "language": data.get("language", "en"),
        "documents": data["documents"],
        "gold_toponyms": data["gold_toponyms"],
        "scoring_spans": data["scoring_spans"],
        "run_limit": data.get("limit"),
        # Reports written before the partial flag existed treated every
        # explicit limit as a smoke run. Keep that conservative behavior for
        # old evidence while allowing new full-slice caps to remain eligible.
        "run_partial": data.get("partial", data.get("limit") is not None),
        "pipeline": pipeline["name"],
        **{role: models.get(role, "") for role in MODEL_ROLES},
        "f1": recognition.get("f1"),
        "accuracy_at_161km": resolution.get("accuracy_at_161km"),
        "auc": resolution.get("auc"),
        "median_error_km": resolution.get("median_error_km"),
        "mean_error_km": resolution.get("mean_error_km"),
        "elapsed_seconds": pipeline["elapsed_seconds"],
        "commit": data["commit"],
        "started_at": data.get("environment", {}).get("started_at", ""),
    }


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
    *,
    metric: str | None = None,
    pipelines: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict[str, t.Any]]:
    """
    Keep, for every corpus and pipeline, the row of the most recent full run.

    Partial smoke runs remain in the raw CSV with their limit, but cannot
    replace full-corpus scores in the card. A limit at or above the registered
    slice size is complete and remains eligible. Runs are ordered by start
    time; metric-specific reruns cannot blank out other phase scores.

    Args:
        rows: Collected result rows
        metric: Consider only rows where this value is present
        pipelines: Consider only these pipelines

    Returns:
        (corpus, pipeline) to its latest row, corpus names lower-cased
    """
    latest: dict[tuple[str, str], dict[str, t.Any]] = {}
    for row in sorted(rows, key=lambda row: row.get("started_at", "")):
        if row.get("run_partial", False):
            continue
        if _missing_metric(row, metric):
            continue
        if _excluded_pipeline(row, pipelines):
            continue
        latest[(row["corpus"].lower(), row["pipeline"])] = row
    return latest


def _missing_metric(row: dict[str, t.Any], metric: str | None) -> bool:
    """Whether a requested metric is absent from this result row."""
    return metric is not None and row.get(metric) is None


def _excluded_pipeline(row: dict[str, t.Any], pipelines: Sequence[str] | None) -> bool:
    """Whether a pipeline filter excludes this row."""
    return pipelines is not None and row["pipeline"] not in pipelines


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
    shown = [_display_score(value) for value in values]
    emphasis = _score_emphasis(shown, higher_is_better=higher_is_better)
    return [_emphasized_score(cell, emphasis) for cell in shown]


def _display_score(value: float | None) -> str | None:
    """Format one score to the precision used in the result tables."""
    return None if value is None else _number(value)


def _score_emphasis(
    shown: Sequence[str | None], *, higher_is_better: bool
) -> dict[str, str]:
    """Choose displayed best and second-best score formatting."""
    ranked = sorted(
        {cell for cell in shown if cell is not None},
        key=float,
        reverse=higher_is_better,
    )
    return dict(zip(ranked, ("**{}**", "<u>{}</u>"), strict=False))


def _emphasized_score(cell: str | None, emphasis: dict[str, str]) -> str:
    """Render a missing, best, second-best, or ordinary table cell."""
    if cell is None:
        return "-"
    return emphasis.get(cell, "{}").format(cell)


def _corpus_order(corpora: set[str]) -> list[str]:
    """Registry order first, then any corpus the registry no longer lists."""
    known = [name for name in CORPORA if name in corpora]
    return known + sorted(corpora - set(known))


def _leaderboard(
    rows: Sequence[dict[str, t.Any]],
    metric: str,
    title: str,
    *,
    higher_is_better: bool = True,
) -> list[str]:
    """One compact table: a row per corpus, a column per main pipeline."""
    latest = latest_results(rows, metric=metric, pipelines=MAIN_PIPELINES)
    pipelines = _leaderboard_pipelines(latest)
    corpora = _corpus_order({corpus for corpus, _ in latest})
    lines = [
        f"### {title}",
        "",
        *_leaderboard_header(pipelines),
    ]
    lines.extend(
        _leaderboard_row(
            corpus, pipelines, latest, metric, higher_is_better=higher_is_better
        )
        for corpus in corpora
    )
    return [*lines, ""]


def _leaderboard_pipelines(
    latest: dict[tuple[str, str], dict[str, t.Any]],
) -> list[str]:
    """Order only pipeline columns with a result row."""
    available = {pipeline for _, pipeline in latest}
    return [name for name in PIPELINE_DESCRIPTIONS if name in available]


def _leaderboard_header(pipelines: list[str]) -> tuple[str, str]:
    """Render the Markdown column heading and separator rows."""
    return (
        "| Corpus | Lang | " + " | ".join(pipelines) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in pipelines) + " |",
    )


def _leaderboard_row(
    corpus: str,
    pipelines: list[str],
    latest: dict[tuple[str, str], dict[str, t.Any]],
    metric: str,
    *,
    higher_is_better: bool,
) -> str:
    """Render one corpus's scores and language label."""
    cells = _leaderboard_cells(corpus, pipelines, latest)
    language = next(row["language"] for row in cells if row)
    values = _leaderboard_values(cells, metric)
    marked = mark_best(values, higher_is_better=higher_is_better)
    return f"| {corpus} | {language} | " + " | ".join(marked) + " |"


def _leaderboard_cells(
    corpus: str,
    pipelines: list[str],
    latest: dict[tuple[str, str], dict[str, t.Any]],
) -> list[dict[str, t.Any] | None]:
    """Look up the requested pipeline rows for one corpus."""
    return [latest.get((corpus, pipeline)) for pipeline in pipelines]


def _leaderboard_values(
    cells: Sequence[dict[str, t.Any] | None], metric: str
) -> list[float | None]:
    """Select one metric from each available pipeline row."""
    return [row[metric] if row else None for row in cells]


def _settings(pipeline: str) -> tuple[str, str]:
    """A variant's (inflection trimming, population weight), as shown."""
    weight, fallback = PRIOR_SETTINGS.get(pipeline, (0.0, False))
    return ("on" if fallback else "off"), f"{weight:g}"


@dataclass(frozen=True)
class _AblationData:
    """Latest score tables and common corpora for one ablation card section."""

    accuracy: dict[tuple[str, str], dict[str, t.Any]]
    auc: dict[tuple[str, str], dict[str, t.Any]]
    present: tuple[str, ...]
    corpora: tuple[str, ...]


def _ablation_data(rows: Sequence[dict[str, t.Any]]) -> _AblationData | None:
    """Select the variant rows and corpora shared by every available variant."""
    variants = [*ABLATION_PIPELINES, *SWEEP_PIPELINES]
    accuracy = latest_results(rows, metric="accuracy_at_161km", pipelines=variants)
    auc = latest_results(rows, metric="auc", pipelines=variants)
    present = _present_variants(variants, accuracy)
    if len(present) < 2 or HYBRID not in present:
        return None
    corpora = _shared_corpora(accuracy, present)
    return _AblationData(accuracy, auc, present, corpora)


def _present_variants(
    variants: Sequence[str],
    accuracy: dict[tuple[str, str], dict[str, t.Any]],
) -> tuple[str, ...]:
    """Keep configured variants with at least one latest accuracy row."""
    return tuple(
        name for name in variants if any(pipeline == name for _, pipeline in accuracy)
    )


def _shared_corpora(
    accuracy: dict[tuple[str, str], dict[str, t.Any]], present: tuple[str, ...]
) -> tuple[str, ...]:
    """Return corpora with a score for every present ablation variant."""
    common = {corpus for corpus, _ in accuracy}
    for name in present:
        common.intersection_update(
            corpus for corpus, pipeline in accuracy if pipeline == name
        )
    return tuple(sorted(common))


def _ablation_value(
    table: dict[tuple[str, str], dict[str, t.Any]],
    corpus: str,
    name: str,
    metric: str,
) -> float:
    """Return one displayed score, rounded to the card's precision."""
    return round(float(table[(corpus, name)][metric]), 3)


def _ablation_mean(
    table: dict[tuple[str, str], dict[str, t.Any]],
    corpora: tuple[str, ...],
    name: str,
    metric: str,
) -> float:
    """Average a variant's displayed score over its common corpora."""
    return sum(
        _ablation_value(table, corpus, name, metric) for corpus in corpora
    ) / len(corpora)


def _ablation_win_counts(data: _AblationData, name: str) -> tuple[int, int]:
    """Count corpora where this variant beats or trails the hybrid."""
    deltas = [
        _ablation_value(data.accuracy, corpus, name, "accuracy_at_161km")
        - _ablation_value(data.accuracy, corpus, HYBRID, "accuracy_at_161km")
        for corpus in data.corpora
    ]
    return sum(delta > 0 for delta in deltas), sum(delta < 0 for delta in deltas)


def _ablation_row(data: _AblationData, name: str) -> str:
    """Render one variant's metrics and wins against the hybrid."""
    trimming, weight = _settings(name)
    mean_acc = _ablation_mean(data.accuracy, data.corpora, name, "accuracy_at_161km")
    mean_auc = _ablation_mean(data.auc, data.corpora, name, "auc")
    better, worse = _ablation_win_counts(data, name)
    return (
        f"| {name} | {trimming} | {weight} | {mean_acc:.3f} | {mean_auc:.3f} "
        f"| {better} / {worse} |"
    )


def _ablation_section(rows: Sequence[dict[str, t.Any]]) -> list[str]:
    """Hybrid and each variant, over the corpora every one of them ran on."""
    data = _ablation_data(rows)
    if data is None:
        return []
    return [
        "## Ablation",
        "",
        "hybrid with one change at a time, recognition held fixed. Means are",
        f"over the {len(data.corpora)} corpora every variant ran on; the last column",
        "counts corpora where Acc@161km beats / trails hybrid.",
        "",
        "| Variant | Trimming | Prior weight | Mean Acc@161km | Mean AUC "
        "| Better / worse |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *(_ablation_row(data, name) for name in data.present),
        "",
    ]


def _main_charts(
    rows: Sequence[dict[str, t.Any]], corpora: list[str]
) -> dict[str, str]:
    """Render the standard metric charts in their declared order."""
    return {
        path: render_bar_chart(
            latest_results(rows, metric=metric, pipelines=MAIN_PIPELINES),
            metric,
            CHART_TITLES[metric],
            corpora=corpora,
            scale=CHART_SCALES.get(metric, "unit"),
        )
        for metric, path in CHART_FILES.items()
    }


def _ablation_corpora(
    corpora: list[str],
    ablation: dict[tuple[str, str], dict[str, t.Any]],
) -> list[str]:
    """Keep corpus chart labels represented in the ablation results."""
    available = {corpus for corpus, _ in ablation}
    return [corpus for corpus in corpora if corpus in available]


def _ablation_chart(rows: Sequence[dict[str, t.Any]], corpora: list[str]) -> str | None:
    """Render the optional ablation chart when multiple variants are present."""
    ablation = latest_results(
        rows, metric="accuracy_at_161km", pipelines=ABLATION_PIPELINES
    )
    present = {pipeline for _, pipeline in ablation}
    if len(present) <= 1:
        return None
    return render_bar_chart(
        ablation,
        "accuracy_at_161km",
        "Ablation: Acc@161km, one change at a time (higher is better)",
        corpora=_ablation_corpora(corpora, ablation),
    )


def render_charts(rows: Sequence[dict[str, t.Any]]) -> dict[str, str]:
    """Return each chart's upload path and SVG, from the latest results."""
    corpora = _corpus_order({row["corpus"].lower() for row in rows})
    charts = _main_charts(rows, corpora)
    ablation = _ablation_chart(rows, corpora)
    if ablation is not None:
        charts[ABLATION_CHART] = ablation
    return charts


def _chart_links(repo_id: str | None) -> list[str]:
    """Image links to the uploaded charts, or nothing when the repo is unknown."""
    if repo_id is None:
        return []
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    return [
        f"![{CHART_TITLES[metric]}]({base}/{path})\n"
        for metric, path in CHART_FILES.items()
    ]


def _ablation_chart_link(
    rows: Sequence[dict[str, t.Any]], repo_id: str | None
) -> list[str]:
    """The ablation chart's image link, when there is an ablation to show."""
    if repo_id is None or not _ablation_section(rows):
        return []
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    return [f"![Ablation: Acc@161km]({base}/{ABLATION_CHART})", ""]


def render_card(rows: Sequence[dict[str, t.Any]], *, repo_id: str | None = None) -> str:
    """Return a compact card that keeps the two scoring protocols separate."""
    del rows
    configs = [
        ("strict_raw", STRICT_RAW_VIEW),
        ("newsli_exact_record_dedup", EXACT_FULL_RECORD_DEDUP_VIEW),
    ]
    config_lines = ["configs:"]
    for name, view in configs:
        config_lines.extend(
            [
                f"- config_name: {name}",
                "  data_files:",
                "  - split: train",
                f"    path: {CURRENT_PREFIX}/{view}/{RESULTS_FILE}",
            ]
        )
    lines = [
        "---",
        *config_lines,
        "language: [en, de, fr, fi, sv, ar, es, fa, ja, pl, ro, sr, ta, tr, uk]",
        "pretty_name: Geoparser benchmark results",
        f"license: {LICENSE}",
        f"license_name: {LICENSE_NAME}",
        "tags: [geoparsing, toponym-resolution, benchmark]",
        "---",
        "",
        "# Geoparser benchmark results",
        "",
        "Six recognizer/resolver combinations from "
        "[geoparser](https://github.com/NoeFlandre/geoparser), evaluated as two "
        "independent stages. Resolution is scored on gold toponym spans; no "
        "end-to-end score is reported.",
        "",
        "## Protocols",
        "",
        "The `strict_raw` config covers 20 corpus slices by 6 combinations. "
        "The separate `newsli_exact_record_dedup` config covers the seven "
        "affected NewsLi languages by 6 combinations. Duplicate removal is "
        "limited to exact full-record matches; its scores are never combined "
        "with strict-raw results.",
        "",
        "Both phases score distinct (document ID, start, end) spans. Strict raw "
        "keeps exact duplicate reference rows inspectable, but each span identity "
        "counts once in the metric denominator. The dedup view removes audited "
        "duplicate rows before scoring, so its metric values can match strict raw.",
        "",
        "## Results",
        "",
    ]
    if repo_id:
        for _, view in configs:
            root = f"{CURRENT_PREFIX}/{view}"
            base = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{root}"
            lines.extend(
                [
                    f"### {view}",
                    "",
                    f"[Metrics CSV]({base}/{RESULTS_FILE}) · "
                    f"[protocol and evidence index]({base}/protocol.json)",
                    "",
                    f"![Recognition F1]({base}/charts/recognition-f1.svg)",
                    "",
                    f"![Resolution Acc@161km]({base}/charts/resolution-acc161.svg)",
                    "",
                    f"[AUC chart]({base}/charts/resolution-auc.svg) · "
                    f"[median error chart]({base}/charts/resolution-median-error.svg)",
                    "",
                ]
            )
    lines.extend(
        [
            "## Inspect and reproduce",
            "",
            f"Per-document gold references, predictions, phase status, and "
            f"failures are under `{CURRENT_PREFIX}/<protocol>/runs/`. "
            f"Model and configuration pins are in each `run-manifest.json`; "
            f"corpus download links and terms are in "
            f"[`corpus-sources.md`]({CURRENT_PREFIX}/corpus-sources.md). "
            f"Article text is not redistributed.",
            "",
            "The `archive/` directory retains earlier public files by source "
            "commit; see its README and index for recovery paths.",
            "",
        ]
    )
    return "\n".join(lines)


def render_source_instructions() -> str:
    """Give retrieval links and terms for each selected corpus slice."""
    lines = [
        "# Corpus sources and retrieval",
        "",
        "Corpus article text is not included in this dataset. Download sources "
        "from their maintainers and apply the source terms before reuse.",
        "",
        "| Slices | Source | Retrieval | Terms |",
        "| --- | --- | --- | --- |",
        f"| GeoVirus (English) | [UniTopRank release]({UNITOPRANK_DATA_URL}) | "
        "Download the release and select its GeoVirus XML corpus. | Apache-2.0 |",
        f"| HIPE-2020, NewsEye, TopRes19th | [HIPE-2022 v2.1]({HIPE_DATA_URL}) | "
        "Use the registered language-specific unmasked test TSV URL listed in "
        "the run manifest. | CC BY-NC-SA 4.0 |",
        f"| NewsLi | [UniTopRank release]({UNITOPRANK_DATA_URL}) | "
        "Download `data.zip`; the adapter selects valid article identifiers "
        "in sorted order, up to 500 per language. | Apache-2.0 |",
        "",
        "Each run manifest records the exact source URL, content SHA-256, "
        "registered split, corpus digest, and (for HIPE-derived slices) the "
        "coordinate-cache digest. The NewsLi dedup view also records its "
        "pre-transform count and number of exact duplicate records removed.",
        "",
    ]
    return "\n".join(lines)


def render_protocol_manifest(scoring_view: str) -> str:
    """Serialize protocol scope, component pins, and legacy display mappings."""
    if scoring_view == STRICT_RAW_VIEW:
        corpora = [missing_matrix.cell_name(cell) for cell in missing_matrix.MATRIX]
    elif scoring_view == EXACT_FULL_RECORD_DEDUP_VIEW:
        corpora = [
            f"newsli-{language}"
            for language in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
        ]
    else:
        msg = f"Unknown scoring view: {scoring_view}"
        raise ValueError(msg)
    value = {
        "schema": "geoparser-benchmark-publication-v2",
        "scoring_view": scoring_view,
        "corpora": corpora,
        "pipelines": {
            name: {
                "components": missing_matrix.PIPELINE_COMPONENTS[name],
                "model_pins": pipelines.pipeline_model_pins(name),
            }
            for name in MAIN_PIPELINES
        },
        "legacy_pipeline_id_map": missing_matrix.LEGACY_PIPELINE_ID_MAP,
        "scoring": {
            "unit": "distinct (document_id, start, end) span identity",
            "duplicate_identity_policy": (
                "repeated gold or predicted rows with the same span identity count once"
            ),
            "recognition": "set-based exact-match F1 on recognizer span identities",
            "resolution": "geodesic error and Acc@161km over distinct gold span identities",
            "end_to_end_score": None,
        },
        "transform": (
            {
                "record_key": ["start", "end", "name", "latitude", "longitude"],
                "retained_occurrence": "first exact record in original order",
            }
            if scoring_view == EXACT_FULL_RECORD_DEDUP_VIEW
            else None
        ),
        "archive": "Earlier public artifacts remain under archive/snapshots/<parent-sha>/.",
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _sha256_text(value: t.Any) -> bool:
    """Return whether ``value`` looks like a lowercase 64-digit SHA-256."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _contains_private_evidence(value: t.Any) -> bool:
    """Detect fields that can contain source text or model echo output."""
    if isinstance(value, dict):
        return any(
            key in PRIVATE_EVIDENCE_KEYS or _contains_private_evidence(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_private_evidence(item) for item in value)
    return False


def _public_json_bytes(data: dict[str, t.Any]) -> bytes:
    """Serialize evidence after removing source text and unbounded model output."""

    def clean(value: t.Any) -> t.Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in PRIVATE_EVIDENCE_KEYS
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return (json.dumps(clean(data), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _expected_corpus_totals(scoring_view: str) -> dict[str, tuple[int, int]]:
    """Return the exact size of every full publication slice."""
    if scoring_view == STRICT_RAW_VIEW:
        return {
            missing_matrix.cell_name(cell): (cell.documents, cell.gold_toponyms)
            for cell in missing_matrix.MATRIX
        }
    if scoring_view == EXACT_FULL_RECORD_DEDUP_VIEW:
        return {
            f"newsli-{language}": (
                cell.documents,
                cell.gold_toponyms
                - missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[language],
            )
            for cell in missing_matrix.MATRIX
            if cell.corpus == "newsli"
            and cell.language in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
            for language in (cell.language,)
        }
    msg = f"Unknown scoring view: {scoring_view}"
    raise ValueError(msg)


def _validate_inspection(
    report_path: Path,
    report_data: dict[str, t.Any],
    scoring_view: str,
    pipeline_name: str,
) -> dict[str, list[dict[str, t.Any]]]:
    """Require complete, text-free per-document evidence for one pipeline."""
    path = report_path.parent / f"inspection-{pipeline_name}.jsonl"
    if not path.is_file():
        msg = f"missing {path.name}"
        raise ValueError(msg)
    references: dict[str, list[dict[str, t.Any]]] = {}
    toponyms = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            toponyms += _validate_inspection_record(
                json.loads(line),
                path=path,
                line_number=line_number,
                scoring_view=scoring_view,
                pipeline_name=pipeline_name,
                references=references,
            )
    expected_documents = report_data["documents"]
    expected_toponyms = report_data["gold_toponyms"]
    if len(references) != expected_documents or toponyms != expected_toponyms:
        msg = (
            f"{path.name} covers {len(references)} documents/{toponyms} references; "
            f"expected {expected_documents}/{expected_toponyms}"
        )
        raise ValueError(msg)
    return references


def _validate_inspection_record(
    record: t.Any,
    *,
    path: Path,
    line_number: int,
    scoring_view: str,
    pipeline_name: str,
    references: dict[str, list[dict[str, t.Any]]],
) -> int:
    """Validate one inspectable example and return its reference count."""
    if not isinstance(record, dict):
        msg = f"{path.name}:{line_number} is not a JSON object"
        raise TypeError(msg)
    prefix = f"{path.name}:{line_number}"
    if record.get("schema") != "geoparser-prediction-inspection-v1":
        msg = f"{prefix} has an unknown schema"
        raise ValueError(msg)
    if record.get("pipeline") != pipeline_name:
        msg = f"{prefix} has the wrong pipeline"
        raise ValueError(msg)
    if record.get("scoring_view") != scoring_view:
        msg = f"{prefix} has the wrong protocol"
        raise ValueError(msg)
    if _contains_private_evidence(record):
        msg = f"{prefix} includes private text"
        raise ValueError(msg)
    _require_inspection_fields(record, INSPECTION_RECORD_FIELDS, prefix)
    document_id = record.get("document_id")
    if not isinstance(document_id, str) or document_id in references:
        msg = f"{prefix} has a duplicate/invalid id"
        raise ValueError(msg)
    reference = record.get("reference")
    if not isinstance(reference, list):
        msg = f"{prefix} lacks reference/prediction"
        raise TypeError(msg)
    prediction = _validate_inspection_collections(record, prefix)
    if record.get("resolution_input") != "gold_reference":
        msg = f"{prefix} does not identify gold-span resolution input"
        raise ValueError(msg)
    _validate_reference_spans(reference, prefix)
    for phase in ("recognition", "resolution"):
        _validate_inspection_phase(record, prediction, phase, prefix)
    references[document_id] = reference
    return len(reference)


def _validate_inspection_collections(
    record: dict[str, t.Any], prefix: str
) -> dict[str, t.Any]:
    """Validate prediction, status, and failure containers in an inspection."""
    prediction = record.get("prediction")
    if not isinstance(prediction, dict):
        msg = f"{prefix} lacks reference/prediction"
        raise TypeError(msg)
    _require_inspection_fields(prediction, {"recognition", "resolution"}, prefix)
    statuses = record["phase_status"]
    if not isinstance(statuses, dict):
        msg = f"{prefix} has malformed phase status"
        raise TypeError(msg)
    _require_inspection_fields(statuses, {"recognition", "resolution"}, prefix)
    failures = record["failures"]
    if not isinstance(failures, dict):
        msg = f"{prefix} has malformed failure details"
        raise TypeError(msg)
    _require_inspection_fields(
        failures, {"recognition", "resolution"}, prefix, exact=False
    )
    for failure in failures.values():
        if not isinstance(failure, dict):
            msg = f"{prefix} has malformed failure details"
            raise TypeError(msg)
        _require_inspection_fields(failure, {"kind", "reason"}, prefix, exact=False)
        if any(not isinstance(item, str) for item in failure.values()):
            msg = f"{prefix} has non-scalar failure details"
            raise ValueError(msg)
    if any(not isinstance(status, str) for status in statuses.values()):
        msg = f"{prefix} has non-scalar phase status"
        raise ValueError(msg)
    return prediction


def _validate_reference_spans(reference: list[t.Any], prefix: str) -> None:
    """Require offsets, mention, and coordinates for every reference span."""
    for span in reference:
        if not isinstance(span, dict):
            msg = f"{prefix} has an incomplete gold reference"
            raise TypeError(msg)
        _require_inspection_fields(span, INSPECTION_REFERENCE_FIELDS, prefix)
        if type(span["start"]) is not int or type(span["end"]) is not int:
            msg = f"{prefix} has non-scalar gold offsets"
            raise ValueError(msg)
        if not isinstance(span["name"], str):
            msg = f"{prefix} has a non-scalar gold mention"
            raise TypeError(msg)
        if not _finite_number(span["latitude"]) or not _finite_number(
            span["longitude"]
        ):
            msg = f"{prefix} has non-scalar gold coordinates"
            raise ValueError(msg)


def _finite_number(value: t.Any) -> bool:
    """Return whether a value is a finite JSON number, excluding booleans."""
    return type(value) in (int, float) and math.isfinite(value)


def _require_inspection_fields(
    value: dict[str, t.Any],
    allowed: t.AbstractSet[str],
    prefix: str,
    *,
    exact: bool = True,
) -> None:
    """Reject inspection fields outside the public, text-free schema."""
    fields = value.keys()
    extra = fields - allowed
    missing = allowed - fields if exact else set()
    if extra or missing:
        details = []
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        if missing:
            details.append(f"missing {sorted(missing)}")
        msg = f"{prefix} has unsupported inspection fields: {', '.join(details)}"
        raise ValueError(msg)


def _validate_inspection_phase(
    record: dict[str, t.Any],
    prediction: dict[str, t.Any],
    phase: str,
    prefix: str,
) -> None:
    """Require a prediction and explicit complete/failure status per phase."""
    predictions = prediction.get(phase)
    if not isinstance(predictions, list):
        msg = f"{prefix} lacks {phase} output"
        raise TypeError(msg)
    if any(not isinstance(item, dict) for item in predictions):
        msg = f"{prefix} has a malformed {phase} prediction"
        raise TypeError(msg)
    for item in predictions:
        _validate_inspection_prediction(item, phase, prefix)
    statuses = record.get("phase_status") or {}
    if not isinstance(statuses, dict):
        msg = f"{prefix} has malformed phase status"
        raise TypeError(msg)
    status = statuses.get(phase)
    if status not in {"complete", "failed"}:
        msg = f"{prefix} has {phase}={status}"
        raise ValueError(msg)
    failures = record.get("failures") or {}
    if not isinstance(failures, dict):
        msg = f"{prefix} has malformed failure details"
        raise TypeError(msg)
    if status == "failed" and phase not in failures:
        msg = f"{prefix} has an unrecorded {phase} failure"
        raise ValueError(msg)


def _validate_inspection_prediction(
    item: dict[str, t.Any], phase: str, prefix: str
) -> None:
    """Require public scalar fields for one predicted annotation."""
    _require_inspection_fields(item, INSPECTION_PREDICTION_FIELDS, prefix)
    if type(item["start"]) is not int or type(item["end"]) is not int:
        msg = f"{prefix} has non-scalar {phase} offsets"
        raise ValueError(msg)
    if item["identifier"] is not None and not isinstance(item["identifier"], str):
        msg = f"{prefix} has a non-scalar {phase} identifier"
        raise ValueError(msg)
    if any(
        item[field] is not None and not _finite_number(item[field])
        for field in ("latitude", "longitude")
    ):
        msg = f"{prefix} has non-scalar {phase} coordinates"
        raise ValueError(msg)


def _validate_report_scope(data: dict[str, t.Any]) -> tuple[str, str]:
    """Require a complete frozen corpus slice within a declared scoring view."""
    scoring_view = data.get("scoring_view")
    if scoring_view not in SCORING_VIEWS:
        msg = "missing or unsupported scoring_view"
        raise ValueError(msg)
    totals = _expected_corpus_totals(scoring_view)
    corpus_name = data.get("corpus")
    if corpus_name not in totals:
        msg = f"{corpus_name!r} is outside {scoring_view}"
        raise ValueError(msg)
    documents, gold_toponyms = totals[corpus_name]
    if data.get("partial") is not False or data.get("documents") != documents:
        msg = f"{corpus_name} is partial or has the wrong document count"
        raise ValueError(msg)
    if data.get("gold_toponyms") != gold_toponyms:
        msg = f"{corpus_name} has the wrong gold toponym count"
        raise ValueError(msg)
    if type(data.get("scoring_spans")) is not int or data["scoring_spans"] < 0:
        msg = f"{corpus_name} has an invalid scoring span count"
        raise ValueError(msg)
    if data.get("limit") is not None and data["limit"] < documents:
        msg = f"{corpus_name} was capped below its full size"
        raise ValueError(msg)
    return scoring_view, corpus_name


def _validate_source_pins(
    data: dict[str, t.Any], corpus_name: str, spec: t.Any
) -> None:
    """Require exact source URLs and content hashes for the loaded corpus."""
    if data.get("source_url") != spec.url:
        msg = f"{corpus_name} has no matching source URL pin"
        raise ValueError(msg)
    if not _sha256_text(data.get("source_content_sha256")):
        msg = f"{corpus_name} has no SHA-256 source content pin"
        raise ValueError(msg)
    if spec.kind == "newsli" and data.get("source_archive_sha256") != data.get(
        "source_content_sha256"
    ):
        msg = f"{corpus_name} has no matching NewsLi archive SHA-256"
        raise ValueError(msg)
    if spec.kind == "hipe" and not _sha256_text(data.get("coordinate_cache_sha256")):
        msg = f"{corpus_name} has no coordinate-cache SHA-256"
        raise ValueError(msg)
    if not isinstance(data.get("corpus_digest"), str) or not data["corpus_digest"]:
        msg = f"{corpus_name} has no parsed corpus digest"
        raise ValueError(msg)


def _validate_manifest_header(
    manifest: dict[str, t.Any], protocol: dict[str, t.Any], view: str, corpus_name: str
) -> None:
    """Require immutable run pins and both independent scoring phases."""
    if manifest.get("status") not in {"complete", "partial"}:
        msg = f"{corpus_name} manifest is not complete"
        raise ValueError(msg)
    if protocol.get("scoring_view") != view:
        msg = f"{corpus_name} manifest scoring_view does not match"
        raise ValueError(msg)
    full_commit = manifest.get("full_commit")
    if not isinstance(full_commit, str) or len(full_commit) != 40:
        msg = f"{corpus_name} manifest has no full source commit pin"
        raise ValueError(msg)
    if not _sha256_text(manifest.get("dependency_lock_sha256")):
        msg = f"{corpus_name} manifest has no dependency-lock SHA-256"
        raise ValueError(msg)
    if set(protocol.get("phases", [])) != {"recognition", "resolution"}:
        msg = f"{corpus_name} manifest did not request both phases"
        raise ValueError(msg)
    _validate_manifest_runtime(protocol, corpus_name)


def _validate_manifest_runtime(protocol: dict[str, t.Any], corpus_name: str) -> None:
    """Require fixed gazetteer, resolver, chunk, and device configuration pins."""
    if protocol.get("gazetteer") != pipelines.GAZETTEER_NAME:
        msg = f"{corpus_name} manifest has an unexpected gazetteer"
        raise ValueError(msg)
    if not _sha256_text(protocol.get("gazetteer_artifact_sha256")):
        msg = f"{corpus_name} manifest has no gazetteer SHA-256"
        raise ValueError(msg)
    if not isinstance(protocol.get("min_similarity"), (int, float)):
        msg = f"{corpus_name} manifest has no resolver similarity setting"
        raise TypeError(msg)
    if not isinstance(protocol.get("chunk_size"), int) or protocol["chunk_size"] < 1:
        msg = f"{corpus_name} manifest has no valid checkpoint chunk size"
        raise ValueError(msg)
    if not isinstance(protocol.get("accelerator"), str) or not isinstance(
        protocol.get("device"), str
    ):
        msg = f"{corpus_name} manifest has no device configuration"
        raise TypeError(msg)


def _validate_manifest_source(
    manifest_cell: dict[str, t.Any],
    data: dict[str, t.Any],
    spec: t.Any,
    corpus_name: str,
) -> None:
    """Match source pins between the run manifest and the corpus report."""
    if manifest_cell.get("source_url") != spec.url:
        msg = f"{corpus_name} manifest source URL does not match"
        raise ValueError(msg)
    if manifest_cell.get("source_content_sha256") != data["source_content_sha256"]:
        msg = f"{corpus_name} source SHA-256 differs in its manifest"
        raise ValueError(msg)
    if manifest_cell.get("corpus_digest") != data["corpus_digest"]:
        msg = f"{corpus_name} corpus digest differs in its manifest"
        raise ValueError(msg)
    for pin in ("coordinate_cache_sha256", "source_archive_sha256"):
        if manifest_cell.get(pin) != data.get(pin):
            msg = f"{corpus_name} {pin} differs in its manifest"
            raise ValueError(msg)


def _load_manifest_cell(
    report_path: Path,
    data: dict[str, t.Any],
    scoring_view: str,
    corpus_name: str,
    spec: t.Any,
) -> tuple[dict[str, t.Any], dict[str, t.Any], dict[str, t.Any]]:
    """Load a run manifest and validate its source and six-pipeline sections."""
    manifest_path = report_path.parent.parent / "run-manifest.json"
    if not manifest_path.is_file():
        msg = f"missing run-manifest.json for {corpus_name}"
        raise ValueError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    _validate_manifest_header(manifest, protocol, scoring_view, corpus_name)
    if data.get("gazetteer") != protocol.get("gazetteer") or data.get(
        "gazetteer_sha256"
    ) != protocol.get("gazetteer_artifact_sha256"):
        msg = f"{corpus_name} report and manifest gazetteer pins differ"
        raise ValueError(msg)
    if data.get("min_similarity") != protocol.get("min_similarity"):
        msg = f"{corpus_name} report and manifest resolver settings differ"
        raise ValueError(msg)
    if data.get("commit") and not manifest["full_commit"].startswith(data["commit"]):
        msg = f"{corpus_name} report and manifest commits do not match"
        raise ValueError(msg)
    model_runs = protocol.get("pipelines", {})
    if set(model_runs) != set(MAIN_PIPELINES):
        msg = f"{corpus_name} manifest must pin exactly six pipelines"
        raise ValueError(msg)
    manifest_cell = protocol.get("corpora", {}).get(corpus_name, {})
    _validate_manifest_source(manifest_cell, data, spec, corpus_name)
    if manifest_cell.get("partial") is not False:
        msg = f"{corpus_name} manifest marks the corpus as partial"
        raise ValueError(msg)
    if (
        manifest_cell.get("documents") != data.get("documents")
        or manifest_cell.get("gold_toponyms") != data.get("gold_toponyms")
        or manifest_cell.get("scoring_spans") != data.get("scoring_spans")
    ):
        msg = f"{corpus_name} report and manifest corpus counts differ"
        raise ValueError(msg)
    if manifest_cell.get("status") not in {
        "complete",
        "complete_with_document_failures",
    }:
        msg = f"{corpus_name} manifest corpus status is incomplete"
        raise ValueError(msg)
    cell_pipelines = manifest_cell.get("pipelines", {})
    if set(cell_pipelines) != set(MAIN_PIPELINES):
        msg = f"{corpus_name} manifest has incomplete pipeline status"
        raise ValueError(msg)
    return model_runs, cell_pipelines, manifest_cell


def _validate_pipeline_evidence(
    name: str,
    report_path: Path,
    report_data: dict[str, t.Any],
    scoring_view: str,
    evidence: tuple[dict[str, t.Any], dict[str, t.Any], dict[str, t.Any]],
) -> dict[str, list[dict[str, t.Any]]]:
    """Require all model-role pins, phase metrics, and inspection records."""
    report_pipeline, manifest_pipeline, cell_pipeline = evidence
    expected_pins = pipelines.pipeline_model_pins(name)
    if manifest_pipeline.get("model_pins", {}) != expected_pins:
        msg = f"{name} manifest is missing or mispins a model role"
        raise ValueError(msg)
    if report_pipeline.get("models") != expected_pins:
        msg = f"{name} report does not carry every expected model role"
        raise ValueError(msg)
    settings = manifest_pipeline.get("execution_settings")
    if not isinstance(settings, dict) or not all(
        settings.get(role) for role in ("recognition", "resolution")
    ):
        msg = f"{name} manifest lacks execution settings"
        raise ValueError(msg)
    if not isinstance(settings.get("document_chunk_size"), int):
        msg = f"{name} manifest lacks its document chunk setting"
        raise TypeError(msg)
    recognition = report_pipeline.get("recognition") or {}
    resolution = report_pipeline.get("resolution") or {}
    if not all(
        recognition.get(metric) is not None for metric in ("precision", "recall", "f1")
    ):
        msg = f"{name} is missing recognition metrics"
        raise ValueError(msg)
    if not all(
        resolution.get(metric) is not None
        for metric in (
            "accuracy_at_161km",
            "auc",
            "median_error_km",
            "mean_error_km",
        )
    ):
        msg = f"{name} is missing recognition or gold-span resolution"
        raise ValueError(msg)
    if cell_pipeline.get("model_pins") != expected_pins:
        msg = f"{name} result manifest is missing expected model roles"
        raise ValueError(msg)
    if cell_pipeline.get("status") != "complete":
        msg = f"{name} has incomplete phase status"
        raise ValueError(msg)
    if cell_pipeline.get("phase_status") != {
        "recognition": "complete",
        "resolution": "complete",
    }:
        msg = f"{name} is missing a completed scoring phase"
        raise ValueError(msg)
    return _validate_inspection(report_path, report_data, scoring_view, name)


def _validate_current_report(path: Path, data: dict[str, t.Any]) -> str:
    """Validate pins, source provenance, metrics, and inspections for a full cell."""
    scoring_view, corpus_name = _validate_report_scope(data)
    spec = CORPORA[corpus_name]
    _validate_source_pins(data, corpus_name, spec)
    results = {item.get("name"): item for item in data.get("pipelines", [])}
    if set(results) != set(MAIN_PIPELINES):
        msg = f"{corpus_name} must contain exactly the six current pipelines"
        raise ValueError(msg)
    model_runs, cell_pipelines, _manifest_cell = _load_manifest_cell(
        path, data, scoring_view, corpus_name, spec
    )
    references_by_pipeline: dict[str, dict[str, list[dict[str, t.Any]]]] = {}
    for name in MAIN_PIPELINES:
        references_by_pipeline[name] = _validate_pipeline_evidence(
            name,
            path,
            data,
            scoring_view,
            (results[name], model_runs[name], cell_pipelines[name]),
        )
    first_references = next(iter(references_by_pipeline.values()))
    if any(refs != first_references for refs in references_by_pipeline.values()):
        msg = f"{corpus_name} pipelines inspect different gold references"
        raise ValueError(msg)
    scoring_spans = sum(
        len({(span["start"], span["end"]) for span in reference})
        for reference in first_references.values()
    )
    if data["scoring_spans"] != scoring_spans:
        msg = f"{corpus_name} scoring span count does not match its inspections"
        raise ValueError(msg)
    _validate_dedup_count(data, scoring_view, spec.language, corpus_name)
    return scoring_view


def _validate_dedup_count(
    data: dict[str, t.Any], scoring_view: str, language: str, corpus_name: str
) -> None:
    """Check the reviewed full-slice exact-record removal count when applicable."""
    if scoring_view != EXACT_FULL_RECORD_DEDUP_VIEW:
        return
    expected_removed = missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[language]
    if data.get("duplicate_records_removed") != expected_removed:
        msg = f"{corpus_name} does not match the reviewed dedup count"
        raise ValueError(msg)


def _bundle_files(
    evidence_dir: Path, repo_id: str
) -> tuple[dict[str, bytes], list[dict[str, t.Any]]]:
    """Build the strict-raw and dedup artifacts after validating every cell."""
    candidates, errors = _current_candidates(evidence_dir)
    files: dict[str, bytes] = {}
    all_rows = []
    for view in SCORING_VIEWS:
        rows = _stage_view(files, evidence_dir, candidates, errors, view)
        all_rows.extend(rows)
    files[f"{CURRENT_PREFIX}/corpus-sources.md"] = render_source_instructions().encode(
        "utf-8"
    )
    files["README.md"] = render_card(all_rows, repo_id=repo_id).encode("utf-8")
    return files, all_rows


def _current_candidates(
    evidence_dir: Path,
) -> tuple[dict[tuple[str, str], tuple[str, Path, dict[str, t.Any]]], list[str]]:
    """Find the latest complete, inspectable report for each current cell."""
    candidates: dict[tuple[str, str], tuple[str, Path, dict[str, t.Any]]] = {}
    errors = []
    for path in sorted(evidence_dir.rglob(REPORT_NAME)):
        relative = path.relative_to(evidence_dir)
        if _is_baseline_path(relative):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("scoring_view") not in SCORING_VIEWS:
            continue
        try:
            scoring_view = _validate_current_report(path, data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        key = (scoring_view, data["corpus"])
        started = data.get("environment", {}).get("started_at", "")
        previous = candidates.get(key)
        if previous is None or started >= previous[0]:
            candidates[key] = (started, path, data)
    return candidates, errors


def _complete_view_candidates(
    candidates: dict[tuple[str, str], tuple[str, Path, dict[str, t.Any]]],
    errors: list[str],
    view: str,
) -> list[tuple[str, Path, dict[str, t.Any]]]:
    """Return the complete view in registry order or explain its missing cells."""
    expected = _expected_corpus_totals(view)
    missing = sorted(
        set(expected)
        - {name for candidate_view, name in candidates if candidate_view == view}
    )
    if missing:
        detail = "; ".join(errors[:5])
        msg = (
            f"cannot publish {view}: missing {len(missing)} complete corpus cells "
            f"({', '.join(missing)}). {detail}"
        )
        raise ValueError(msg)
    return [
        (name, candidates[(view, name)][1], candidates[(view, name)][2])
        for name in expected
    ]


def _stage_corpus_files(
    files: dict[str, bytes],
    evidence_dir: Path,
    view_root: str,
    corpus_name: str,
    report_path: Path,
    data: dict[str, t.Any],
) -> list[dict[str, t.Any]]:
    """Stage sanitized report, run manifest, and all six inspection files."""
    run_path = report_path.parent.parent
    run_name = run_path.name
    manifest_path = run_path / "run-manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files[f"{view_root}/runs/{run_name}/run-manifest.json"] = _public_json_bytes(
        manifest_data
    )
    report_base = f"{view_root}/runs/{run_name}/{corpus_name}"
    files[f"{report_base}/{REPORT_NAME}"] = _public_json_bytes(data)
    markdown = report_path.with_suffix(".md")
    if markdown.is_file():
        files[f"{report_base}/benchmark-report.md"] = markdown.read_bytes()
    for pipeline in MAIN_PIPELINES:
        inspection = report_path.parent / f"inspection-{pipeline}.jsonl"
        records = [
            json.loads(line)
            for line in inspection.read_text(encoding="utf-8").splitlines()
        ]
        content = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in records
        )
        files[f"{report_base}/inspection/{pipeline}.jsonl"] = (content + "\n").encode(
            "utf-8"
        )
    return _report_rows(report_path.relative_to(evidence_dir), data)


def _stage_view(
    files: dict[str, bytes],
    evidence_dir: Path,
    candidates: dict[tuple[str, str], tuple[str, Path, dict[str, t.Any]]],
    errors: list[str],
    view: str,
) -> list[dict[str, t.Any]]:
    """Stage one protocol's files, score table, plots, and protocol manifest."""
    selected = _complete_view_candidates(candidates, errors, view)
    root = f"{CURRENT_PREFIX}/{view}"
    rows = []
    for corpus_name, report_path, data in selected:
        rows.extend(
            _stage_corpus_files(
                files, evidence_dir, root, corpus_name, report_path, data
            )
        )
    files[f"{root}/{RESULTS_FILE}"] = render_csv(rows).encode("utf-8")
    files[f"{root}/protocol.json"] = render_protocol_manifest(view).encode("utf-8")
    for chart_path, svg in render_charts(rows).items():
        files[f"{root}/{chart_path}"] = svg.encode("utf-8")
    return rows


def _metadata_value(value: t.Any, name: str) -> t.Any:
    """Read a Hub metadata field from either its object or mapping form."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _git_blob_digest(content: bytes, algorithm: str) -> str:
    """Return the Git object digest for file content."""
    header = f"blob {len(content)}\0".encode()
    digest = hashlib.new(algorithm)
    digest.update(header)
    digest.update(content)
    return digest.hexdigest()


def _remote_file_matches(sibling: t.Any, content: bytes) -> bool:
    """Compare desired bytes with the Hub's LFS or Git blob content hash."""
    lfs = _metadata_value(sibling, "lfs")
    lfs_sha256 = _metadata_value(lfs, "sha256") if lfs is not None else None
    if isinstance(lfs_sha256, str):
        return hashlib.sha256(content).hexdigest() == lfs_sha256

    blob_id = _metadata_value(sibling, "blob_id")
    if not isinstance(blob_id, str):
        return False
    algorithm = {40: "sha1", 64: "sha256"}.get(len(blob_id))
    return algorithm is not None and _git_blob_digest(content, algorithm) == blob_id


def _repository_matches_bundle(
    info: t.Any,
    repository_paths: t.AbstractSet[str],
    bundle_files: dict[str, bytes],
    *,
    required_additions: dict[str, bytes] | None = None,
    required_paths: t.AbstractSet[str] | None = None,
) -> bool:
    """Check the live tree and selected new files against exact desired bytes."""
    live_paths = {
        path
        for path in repository_paths
        if path != ".gitattributes" and not path.startswith("archive/")
    }
    if live_paths != set(bundle_files):
        return False
    if required_paths and not required_paths.issubset(repository_paths):
        return False

    siblings = getattr(info, "siblings", None)
    if siblings is None:
        return False
    sibling_by_path = {
        _metadata_value(sibling, "rfilename"): sibling for sibling in siblings
    }
    expected = dict(bundle_files)
    if required_additions:
        expected.update(required_additions)
    return all(
        path in sibling_by_path and _remote_file_matches(sibling_by_path[path], content)
        for path, content in expected.items()
    )


def _matching_current_revision(
    api: t.Any,
    repo_id: str,
    bundle_files: dict[str, bytes],
    *,
    required_additions: dict[str, bytes] | None = None,
    required_paths: t.AbstractSet[str] | None = None,
) -> str | None:
    """Return the current SHA when an uncertain publish is fully present."""
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or len(sha) != 40:
        return None
    paths = set(api.list_repo_files(repo_id, repo_type="dataset", revision=sha))
    if not _repository_matches_bundle(
        info,
        paths,
        bundle_files,
        required_additions=required_additions,
        required_paths=required_paths,
    ):
        return None
    return sha


@dataclass(frozen=True)
class _PublishRecovery:
    """Content needed to reconcile a response lost after a Hub commit."""

    desired_files: dict[str, bytes]
    archive_additions: dict[str, bytes]
    archived_paths: dict[str, str]


def _try_reconcile_publish(
    api: t.Any,
    repo_id: str,
    desired_files: dict[str, bytes],
    archive_additions: dict[str, bytes],
    archived_paths: dict[str, str],
) -> str | None:
    """Best-effort check whether an uncertain Hub write fully landed."""
    try:
        return _matching_current_revision(
            api,
            repo_id,
            desired_files,
            required_additions=archive_additions,
            required_paths=set(archived_paths.values()),
        )
    except OSError:
        return None


def _archive_paths(existing: Sequence[str], parent_sha: str) -> dict[str, str]:
    """Map current and legacy files into a recoverable parent snapshot."""
    root = f"{ARCHIVE_PREFIX}/{parent_sha}"
    archived = {}
    for source in existing:
        if source == ".gitattributes" or source.startswith("archive/"):
            continue
        if source == "README.md":
            destination = f"{root}/legacy/README.md"
        elif source.startswith("current/"):
            destination = f"{root}/current/{source.removeprefix('current/')}"
        else:
            destination = f"{root}/legacy/{source}"
        archived[source] = destination
    return archived


def _add_archive_metadata(
    files: dict[str, bytes],
    desired_files: dict[str, bytes],
    parent_sha: str,
    archived_paths: dict[str, str],
) -> dict[str, bytes]:
    """Add the snapshot index and return its new content for reconciliation."""
    root = f"{ARCHIVE_PREFIX}/{parent_sha}"
    archive_record = {
        "schema": "geoparser-benchmark-archive-v1",
        "source_parent_sha": parent_sha,
        "snapshot_prefix": root,
        "files": archived_paths,
        "note": "Previous public paths are recoverable under this snapshot prefix.",
    }
    archive_bytes = (
        json.dumps(archive_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    files[f"{root}/ARCHIVE.json"] = archive_bytes
    files[f"archive/index/{parent_sha}.json"] = archive_bytes
    files["archive/README.md"] = (
        b"# Archived benchmark evidence\n\n"
        b"Every publication records the exact previous Hub commit and maps its "
        b"old paths into `snapshots/<parent-sha>/`. The matching JSON index is "
        b"under `index/`; archived files remain available at their mapped paths.\n"
    )
    return {
        path: content for path, content in files.items() if path not in desired_files
    }


def _publish_operations(
    files: dict[str, bytes], archived_paths: dict[str, str], parent_sha: str
) -> list[t.Any]:
    """Build one Hub transaction that snapshots old files and writes the bundle."""
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
    )

    operations = [
        CommitOperationCopy(
            src_path_in_repo=source,
            path_in_repo=destination,
            src_revision=parent_sha,
        )
        for source, destination in archived_paths.items()
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=source)
        for source in archived_paths
        if source != "README.md" and source not in files
    )
    operations.extend(
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
        for path, content in files.items()
    )
    return operations


def _verify_publish_readback(
    api: t.Any,
    repo_id: str,
    commit_sha: str,
    files: dict[str, bytes],
    archived_paths: dict[str, str],
) -> None:
    """Check required files and byte hashes after the atomic Hub commit."""
    paths_after = set(
        api.list_repo_files(repo_id, repo_type="dataset", revision=commit_sha)
    )
    required_paths = set(files) | set(archived_paths.values())
    missing_paths = sorted(required_paths - paths_after)
    if missing_paths:
        msg = f"Hub readback is missing {missing_paths[:10]}"
        raise RuntimeError(msg)
    inspection_sample = next(path for path in sorted(files) if "/inspection/" in path)
    readback_paths = [
        "README.md",
        f"{CURRENT_PREFIX}/{STRICT_RAW_VIEW}/{RESULTS_FILE}",
        f"{CURRENT_PREFIX}/{STRICT_RAW_VIEW}/protocol.json",
        f"{CURRENT_PREFIX}/{EXACT_FULL_RECORD_DEDUP_VIEW}/{RESULTS_FILE}",
        f"{CURRENT_PREFIX}/{EXACT_FULL_RECORD_DEDUP_VIEW}/protocol.json",
        f"{CURRENT_PREFIX}/corpus-sources.md",
        inspection_sample,
    ]
    for path in readback_paths:
        downloaded = Path(
            api.hf_hub_download(
                repo_id,
                filename=path,
                repo_type="dataset",
                revision=commit_sha,
            )
        ).read_bytes()
        if hashlib.sha256(downloaded).digest() != hashlib.sha256(files[path]).digest():
            msg = f"Hub readback differs from committed artifact {path}"
            raise RuntimeError(msg)


def _create_publish_commit(
    api: t.Any,
    repo_id: str,
    parent_sha: str,
    operations: list[t.Any],
    recovery: _PublishRecovery,
) -> tuple[str, str]:
    """Create the guarded commit or reconcile a response lost after it landed."""
    try:
        result = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Publish validated benchmark matrix v2",
            parent_commit=parent_sha,
        )
    except Exception:
        reconciled_sha = _try_reconcile_publish(
            api,
            repo_id,
            recovery.desired_files,
            recovery.archive_additions,
            recovery.archived_paths,
        )
        if reconciled_sha is not None:
            return reconciled_sha, "reconciled"
        raise

    commit_sha = getattr(result, "oid", None)
    if not isinstance(commit_sha, str) or len(commit_sha) != 40:
        error = ValueError("Hub commit did not return a full commit SHA")
        reconciled_sha = _try_reconcile_publish(
            api,
            repo_id,
            recovery.desired_files,
            recovery.archive_additions,
            recovery.archived_paths,
        )
        if reconciled_sha is not None:
            return reconciled_sha, "reconciled"
        raise error
    return commit_sha, "published"


def publish(evidence_dir: Path, repo_id: str, *, api: t.Any) -> dict[str, str]:
    """Publish a fully validated bundle once, with retries reconciled by hash."""
    files, _rows = _bundle_files(evidence_dir, repo_id)
    desired_files = dict(files)
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    parent_sha = info.sha
    if not isinstance(parent_sha, str) or len(parent_sha) != 40:
        msg = "dataset repository did not return a full parent commit SHA"
        raise ValueError(msg)
    existing = api.list_repo_files(repo_id, repo_type="dataset", revision=parent_sha)
    if _repository_matches_bundle(info, set(existing), desired_files):
        return {
            "parent_sha": parent_sha,
            "commit_sha": parent_sha,
            "status": "unchanged",
        }

    archived_paths = _archive_paths(existing, parent_sha)
    archive_additions = _add_archive_metadata(
        files, desired_files, parent_sha, archived_paths
    )
    operations = _publish_operations(files, archived_paths, parent_sha)
    recovery = _PublishRecovery(desired_files, archive_additions, archived_paths)
    commit_sha, status = _create_publish_commit(
        api,
        repo_id,
        parent_sha,
        operations,
        recovery,
    )
    if status == "published":
        try:
            _verify_publish_readback(api, repo_id, commit_sha, files, archived_paths)
        except Exception:
            reconciled_sha = _try_reconcile_publish(
                api,
                repo_id,
                recovery.desired_files,
                recovery.archive_additions,
                recovery.archived_paths,
            )
            if reconciled_sha is None:
                raise
            commit_sha = reconciled_sha
            status = "reconciled"
    return {"parent_sha": parent_sha, "commit_sha": commit_sha, "status": status}


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the Hub publisher."""
    parser = argparse.ArgumentParser(prog="python -m scripts.benchmark.publish")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Publish the committed evidence to the Hub."""
    arguments = build_parser().parse_args(argv)

    from huggingface_hub import HfApi

    publish(arguments.evidence_dir, arguments.repo_id, api=HfApi())
    print(f"https://huggingface.co/datasets/{arguments.repo_id}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
