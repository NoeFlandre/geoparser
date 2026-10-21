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

if t.TYPE_CHECKING:
    from huggingface_hub import CommitOperation

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


def _corpus_order(corpora: set[str]) -> list[str]:
    """Registry order first, then any corpus the registry no longer lists."""
    known = [name for name in CORPORA if name in corpora]
    return known + sorted(corpora - set(known))


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
    corpora = _protocol_corpora(scoring_view)
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
        "transform": _protocol_transform(scoring_view),
        "archive": "Earlier public artifacts remain under archive/snapshots/<parent-sha>/.",
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _protocol_corpora(scoring_view: str) -> list[str]:
    builders = {
        STRICT_RAW_VIEW: lambda: [
            missing_matrix.cell_name(cell) for cell in missing_matrix.MATRIX
        ],
        EXACT_FULL_RECORD_DEDUP_VIEW: lambda: [
            f"newsli-{language}"
            for language in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
        ],
    }
    try:
        return builders[scoring_view]()
    except KeyError as error:
        message = f"Unknown scoring view: {scoring_view}"
        raise ValueError(message) from error


def _protocol_transform(scoring_view: str) -> dict[str, list[str] | str] | None:
    if scoring_view != EXACT_FULL_RECORD_DEDUP_VIEW:
        return None
    return {
        "record_key": ["start", "end", "name", "latitude", "longitude"],
        "retained_occurrence": "first exact record in original order",
    }


def _sha256_text(value: t.Any) -> bool:
    """Return whether ``value`` looks like a lowercase 64-digit SHA-256."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_condition(
    condition: t.Any,
    message: str,
    error_type: type[Exception] = ValueError,
) -> None:
    """Raise the requested validation error when a condition is false."""
    if not condition:
        raise error_type(message)


def _contains_private_evidence(value: t.Any) -> bool:
    """Detect fields that can contain source text or model echo output."""
    if isinstance(value, dict):
        return _mapping_contains_private_evidence(value)
    return _sequence_contains_private_evidence(value)


def _mapping_contains_private_evidence(value: dict[str, t.Any]) -> bool:
    return any(
        key in PRIVATE_EVIDENCE_KEYS or _contains_private_evidence(item)
        for key, item in value.items()
    )


def _sequence_contains_private_evidence(value: t.Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(_contains_private_evidence(item) for item in value)


def _public_json_bytes(data: dict[str, t.Any]) -> bytes:
    """Serialize evidence after removing source text and unbounded model output."""
    clean = _clean_public_value(data)
    return (json.dumps(clean, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _clean_public_value(value: t.Any) -> t.Any:
    if isinstance(value, dict):
        return _clean_public_mapping(value)
    if isinstance(value, list):
        return [_clean_public_value(item) for item in value]
    return value


def _clean_public_mapping(value: dict[str, t.Any]) -> dict[str, t.Any]:
    return {
        key: _clean_public_value(item)
        for key, item in value.items()
        if key not in PRIVATE_EVIDENCE_KEYS
    }


def _expected_corpus_totals(scoring_view: str) -> dict[str, tuple[int, int]]:
    """Return the exact size of every full publication slice."""
    builders = {
        STRICT_RAW_VIEW: _strict_raw_corpus_totals,
        EXACT_FULL_RECORD_DEDUP_VIEW: _deduplicated_corpus_totals,
    }
    try:
        builder = builders[scoring_view]
    except KeyError as error:
        message = f"Unknown scoring view: {scoring_view}"
        raise ValueError(message) from error
    return builder()


def _strict_raw_corpus_totals() -> dict[str, tuple[int, int]]:
    return {
        missing_matrix.cell_name(cell): (cell.documents, cell.gold_toponyms)
        for cell in missing_matrix.MATRIX
    }


def _deduplicated_corpus_totals() -> dict[str, tuple[int, int]]:
    return {
        f"newsli-{cell.language}": (
            cell.documents,
            cell.gold_toponyms
            - missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED[cell.language],
        )
        for cell in missing_matrix.MATRIX
        if cell.corpus == "newsli"
        and cell.language in missing_matrix.NEWSLI_EXACT_DEDUP_LANGUAGES
    }


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
    _require_condition(
        isinstance(record, dict),
        f"{path.name}:{line_number} is not a JSON object",
        TypeError,
    )
    record = t.cast(dict[str, t.Any], record)
    prefix = f"{path.name}:{line_number}"
    _require_condition(
        record.get("schema") == "geoparser-prediction-inspection-v1",
        f"{prefix} has an unknown schema",
    )
    _require_condition(
        record.get("pipeline") == pipeline_name, f"{prefix} has the wrong pipeline"
    )
    _require_condition(
        record.get("scoring_view") == scoring_view, f"{prefix} has the wrong protocol"
    )
    _require_condition(
        not _contains_private_evidence(record), f"{prefix} includes private text"
    )
    _require_inspection_fields(record, INSPECTION_RECORD_FIELDS, prefix)
    document_id = record.get("document_id")
    _require_condition(
        isinstance(document_id, str) and document_id not in references,
        f"{prefix} has a duplicate/invalid id",
    )
    document_id = t.cast(str, document_id)
    reference = record.get("reference")
    _require_condition(
        isinstance(reference, list), f"{prefix} lacks reference/prediction", TypeError
    )
    reference = t.cast(list[t.Any], reference)
    prediction = _validate_inspection_collections(record, prefix)
    _require_condition(
        record.get("resolution_input") == "gold_reference",
        f"{prefix} does not identify gold-span resolution input",
    )
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
    _require_condition(
        isinstance(prediction, dict), f"{prefix} lacks reference/prediction", TypeError
    )
    prediction = t.cast(dict[str, t.Any], prediction)
    _require_inspection_fields(prediction, {"recognition", "resolution"}, prefix)
    statuses = record["phase_status"]
    _require_condition(
        isinstance(statuses, dict), f"{prefix} has malformed phase status", TypeError
    )
    statuses = t.cast(dict[str, t.Any], statuses)
    _require_inspection_fields(statuses, {"recognition", "resolution"}, prefix)
    failures = record["failures"]
    _require_condition(
        isinstance(failures, dict), f"{prefix} has malformed failure details", TypeError
    )
    failures = t.cast(dict[str, t.Any], failures)
    _require_inspection_fields(
        failures, {"recognition", "resolution"}, prefix, exact=False
    )
    for failure in failures.values():
        _require_condition(
            isinstance(failure, dict),
            f"{prefix} has malformed failure details",
            TypeError,
        )
        failure = t.cast(dict[str, t.Any], failure)
        _require_inspection_fields(failure, {"kind", "reason"}, prefix, exact=False)
        _require_condition(
            all(isinstance(item, str) for item in failure.values()),
            f"{prefix} has non-scalar failure details",
        )
    _require_condition(
        all(isinstance(status, str) for status in statuses.values()),
        f"{prefix} has non-scalar phase status",
    )
    return prediction


def _validate_reference_spans(reference: list[t.Any], prefix: str) -> None:
    """Require offsets, mention, and coordinates for every reference span."""
    for span in reference:
        _validate_reference_span(span, prefix)


def _validate_reference_span(span: t.Any, prefix: str) -> None:
    _require_condition(
        isinstance(span, dict), f"{prefix} has an incomplete gold reference", TypeError
    )
    span = t.cast(dict[str, t.Any], span)
    _require_inspection_fields(span, INSPECTION_REFERENCE_FIELDS, prefix)
    _require_condition(
        type(span["start"]) is int and type(span["end"]) is int,
        f"{prefix} has non-scalar gold offsets",
    )
    _require_condition(
        isinstance(span["name"], str),
        f"{prefix} has a non-scalar gold mention",
        TypeError,
    )
    _require_condition(
        _finite_number(span["latitude"]) and _finite_number(span["longitude"]),
        f"{prefix} has non-scalar gold coordinates",
    )


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
    details = _inspection_field_details(extra, missing)
    _require_condition(
        not details,
        f"{prefix} has unsupported inspection fields: {', '.join(details)}",
    )


def _inspection_field_details(
    extra: t.AbstractSet[str], missing: t.AbstractSet[str]
) -> list[str]:
    details = []
    if extra:
        details.append(f"unexpected {sorted(extra)}")
    if missing:
        details.append(f"missing {sorted(missing)}")
    return details


def _validate_inspection_phase(
    record: dict[str, t.Any],
    prediction: dict[str, t.Any],
    phase: str,
    prefix: str,
) -> None:
    """Require a prediction and explicit complete/failure status per phase."""
    _validated_phase_predictions(prediction, phase, prefix)
    statuses = record.get("phase_status") or {}
    failures = record.get("failures") or {}
    status = _validated_phase_status(statuses, phase, prefix)
    failures = _validated_phase_failures(failures, prefix)
    _require_condition(
        status != "failed" or phase in failures,
        f"{prefix} has an unrecorded {phase} failure",
    )


def _validated_phase_predictions(
    prediction: dict[str, t.Any], phase: str, prefix: str
) -> list[dict[str, t.Any]]:
    values = prediction.get(phase)
    _require_condition(
        isinstance(values, list), f"{prefix} lacks {phase} output", TypeError
    )
    predictions = t.cast(list[t.Any], values)
    for item in predictions:
        _require_condition(
            isinstance(item, dict),
            f"{prefix} has a malformed {phase} prediction",
            TypeError,
        )
        _validate_inspection_prediction(t.cast(dict[str, t.Any], item), phase, prefix)
    return t.cast(list[dict[str, t.Any]], predictions)


def _validated_phase_status(statuses: t.Any, phase: str, prefix: str) -> str:
    _require_condition(
        isinstance(statuses, dict), f"{prefix} has malformed phase status", TypeError
    )
    status = t.cast(dict[str, t.Any], statuses).get(phase)
    _require_condition(
        status in {"complete", "failed"}, f"{prefix} has {phase}={status}"
    )
    return t.cast(str, status)


def _validated_phase_failures(failures: t.Any, prefix: str) -> dict[str, t.Any]:
    _require_condition(
        isinstance(failures, dict), f"{prefix} has malformed failure details", TypeError
    )
    return t.cast(dict[str, t.Any], failures)


def _validate_inspection_prediction(
    item: dict[str, t.Any], phase: str, prefix: str
) -> None:
    """Require public scalar fields for one predicted annotation."""
    _require_inspection_fields(item, INSPECTION_PREDICTION_FIELDS, prefix)
    _require_condition(
        type(item["start"]) is int and type(item["end"]) is int,
        f"{prefix} has non-scalar {phase} offsets",
    )
    _require_condition(
        item["identifier"] is None or isinstance(item["identifier"], str),
        f"{prefix} has a non-scalar {phase} identifier",
    )
    _require_condition(
        all(
            item[field] is None or _finite_number(item[field])
            for field in ("latitude", "longitude")
        ),
        f"{prefix} has non-scalar {phase} coordinates",
    )


def _validate_report_scope(data: dict[str, t.Any]) -> tuple[str, str]:
    """Require a complete frozen corpus slice within a declared scoring view."""
    scoring_view = data.get("scoring_view")
    _require_condition(
        scoring_view in SCORING_VIEWS, "missing or unsupported scoring_view"
    )
    scoring_view = t.cast(str, scoring_view)
    totals = _expected_corpus_totals(scoring_view)
    corpus_name = data.get("corpus")
    _require_condition(
        corpus_name in totals, f"{corpus_name!r} is outside {scoring_view}"
    )
    corpus_name = t.cast(str, corpus_name)
    documents, gold_toponyms = totals[corpus_name]
    _require_condition(
        data.get("partial") is False and data.get("documents") == documents,
        f"{corpus_name} is partial or has the wrong document count",
    )
    _require_condition(
        data.get("gold_toponyms") == gold_toponyms,
        f"{corpus_name} has the wrong gold toponym count",
    )
    scoring_spans = data.get("scoring_spans")
    _require_condition(
        type(scoring_spans) is int and scoring_spans >= 0,
        f"{corpus_name} has an invalid scoring span count",
    )
    limit = data.get("limit")
    _require_condition(
        limit is None or limit >= documents,
        f"{corpus_name} was capped below its full size",
    )
    return scoring_view, corpus_name


def _validate_source_pins(
    data: dict[str, t.Any], corpus_name: str, spec: t.Any
) -> None:
    """Require exact source URLs and content hashes for the loaded corpus."""
    _require_condition(
        data.get("source_url") == spec.url,
        f"{corpus_name} has no matching source URL pin",
    )
    _require_condition(
        _sha256_text(data.get("source_content_sha256")),
        f"{corpus_name} has no SHA-256 source content pin",
    )
    _validate_optional_source_hashes(data, corpus_name, spec)
    digest = data.get("corpus_digest")
    _require_condition(
        isinstance(digest, str) and digest,
        f"{corpus_name} has no parsed corpus digest",
    )


def _validate_optional_source_hashes(
    data: dict[str, t.Any], corpus_name: str, spec: t.Any
) -> None:
    if spec.kind == "newsli":
        _require_condition(
            data.get("source_archive_sha256") == data.get("source_content_sha256"),
            f"{corpus_name} has no matching NewsLi archive SHA-256",
        )
    if spec.kind == "hipe":
        _require_condition(
            _sha256_text(data.get("coordinate_cache_sha256")),
            f"{corpus_name} has no coordinate-cache SHA-256",
        )


def _validate_manifest_header(
    manifest: dict[str, t.Any], protocol: dict[str, t.Any], view: str, corpus_name: str
) -> None:
    """Require immutable run pins and both independent scoring phases."""
    _require_condition(
        manifest.get("status") in {"complete", "partial"},
        f"{corpus_name} manifest is not complete",
    )
    _require_condition(
        protocol.get("scoring_view") == view,
        f"{corpus_name} manifest scoring_view does not match",
    )
    full_commit = manifest.get("full_commit")
    _require_condition(
        isinstance(full_commit, str) and len(full_commit) == 40,
        f"{corpus_name} manifest has no full source commit pin",
    )
    _require_condition(
        _sha256_text(manifest.get("dependency_lock_sha256")),
        f"{corpus_name} manifest has no dependency-lock SHA-256",
    )
    _require_condition(
        set(protocol.get("phases", [])) == {"recognition", "resolution"},
        f"{corpus_name} manifest did not request both phases",
    )
    _validate_manifest_runtime(protocol, corpus_name)


def _validate_manifest_runtime(protocol: dict[str, t.Any], corpus_name: str) -> None:
    """Require fixed gazetteer, resolver, chunk, and device configuration pins."""
    _require_condition(
        protocol.get("gazetteer") == pipelines.GAZETTEER_NAME,
        f"{corpus_name} manifest has an unexpected gazetteer",
    )
    _require_condition(
        _sha256_text(protocol.get("gazetteer_artifact_sha256")),
        f"{corpus_name} manifest has no gazetteer SHA-256",
    )
    _require_condition(
        isinstance(protocol.get("min_similarity"), (int, float)),
        f"{corpus_name} manifest has no resolver similarity setting",
        TypeError,
    )
    _require_condition(
        isinstance(protocol.get("chunk_size"), int) and protocol["chunk_size"] >= 1,
        f"{corpus_name} manifest has no valid checkpoint chunk size",
    )
    _require_condition(
        isinstance(protocol.get("accelerator"), str)
        and isinstance(protocol.get("device"), str),
        f"{corpus_name} manifest has no device configuration",
        TypeError,
    )


def _validate_manifest_source(
    manifest_cell: dict[str, t.Any],
    data: dict[str, t.Any],
    spec: t.Any,
    corpus_name: str,
) -> None:
    """Match source pins between the run manifest and the corpus report."""
    pins = {
        "source_url": spec.url,
        "source_content_sha256": data["source_content_sha256"],
        "corpus_digest": data["corpus_digest"],
        "coordinate_cache_sha256": data.get("coordinate_cache_sha256"),
        "source_archive_sha256": data.get("source_archive_sha256"),
    }
    for pin, expected in pins.items():
        _require_condition(
            manifest_cell.get(pin) == expected,
            f"{corpus_name} {pin} differs in its manifest",
        )


def _load_manifest_cell(
    report_path: Path,
    data: dict[str, t.Any],
    scoring_view: str,
    corpus_name: str,
    spec: t.Any,
) -> tuple[dict[str, t.Any], dict[str, t.Any], dict[str, t.Any]]:
    """Load a run manifest and validate its source and six-pipeline sections."""
    manifest_path = report_path.parent.parent / "run-manifest.json"
    _require_condition(
        manifest_path.is_file(), f"missing run-manifest.json for {corpus_name}"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    _validate_manifest_header(manifest, protocol, scoring_view, corpus_name)
    _validate_report_manifest_runtime(data, protocol, manifest, corpus_name)
    model_runs = protocol.get("pipelines", {})
    _require_condition(
        set(model_runs) == set(MAIN_PIPELINES),
        f"{corpus_name} manifest must pin exactly six pipelines",
    )
    manifest_cell = protocol.get("corpora", {}).get(corpus_name, {})
    _validate_manifest_source(manifest_cell, data, spec, corpus_name)
    _require_condition(
        manifest_cell.get("partial") is False,
        f"{corpus_name} manifest marks the corpus as partial",
    )
    _require_condition(
        manifest_cell.get("documents") == data.get("documents")
        and manifest_cell.get("gold_toponyms") == data.get("gold_toponyms")
        and manifest_cell.get("scoring_spans") == data.get("scoring_spans"),
        f"{corpus_name} report and manifest corpus counts differ",
    )
    _require_condition(
        manifest_cell.get("status") in {"complete", "complete_with_document_failures"},
        f"{corpus_name} manifest corpus status is incomplete",
    )
    cell_pipelines = manifest_cell.get("pipelines", {})
    _require_condition(
        set(cell_pipelines) == set(MAIN_PIPELINES),
        f"{corpus_name} manifest has incomplete pipeline status",
    )
    return model_runs, cell_pipelines, manifest_cell


def _validate_report_manifest_runtime(
    data: dict[str, t.Any],
    protocol: dict[str, t.Any],
    manifest: dict[str, t.Any],
    corpus_name: str,
) -> None:
    gazetteer_matches = data.get("gazetteer") == protocol.get("gazetteer") and data.get(
        "gazetteer_sha256"
    ) == protocol.get("gazetteer_artifact_sha256")
    _require_condition(
        gazetteer_matches,
        f"{corpus_name} report and manifest gazetteer pins differ",
    )
    _require_condition(
        data.get("min_similarity") == protocol.get("min_similarity"),
        f"{corpus_name} report and manifest resolver settings differ",
    )
    commit = data.get("commit")
    _require_condition(
        not commit or manifest["full_commit"].startswith(commit),
        f"{corpus_name} report and manifest commits do not match",
    )


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
    _require_condition(
        manifest_pipeline.get("model_pins", {}) == expected_pins,
        f"{name} manifest is missing or mispins a model role",
    )
    _require_condition(
        report_pipeline.get("models") == expected_pins,
        f"{name} report does not carry every expected model role",
    )
    settings = manifest_pipeline.get("execution_settings")
    _require_condition(
        isinstance(settings, dict), f"{name} manifest lacks execution settings"
    )
    settings = t.cast(dict[str, t.Any], settings)
    _validate_pipeline_execution_settings(settings, name)
    recognition = report_pipeline.get("recognition") or {}
    resolution = report_pipeline.get("resolution") or {}
    _validate_pipeline_metrics(recognition, resolution, name)
    _validate_cell_pipeline_status(cell_pipeline, expected_pins, name)
    return _validate_inspection(report_path, report_data, scoring_view, name)


def _validate_pipeline_execution_settings(
    settings: dict[str, t.Any], name: str
) -> None:
    _require_condition(
        all(settings.get(role) for role in ("recognition", "resolution")),
        f"{name} manifest lacks execution settings",
    )
    _require_condition(
        isinstance(settings.get("document_chunk_size"), int),
        f"{name} manifest lacks its document chunk setting",
        TypeError,
    )


def _validate_pipeline_metrics(
    recognition: dict[str, t.Any], resolution: dict[str, t.Any], name: str
) -> None:
    recognition_metrics = ("precision", "recall", "f1")
    resolution_metrics = (
        "accuracy_at_161km",
        "auc",
        "median_error_km",
        "mean_error_km",
    )
    _require_condition(
        all(recognition.get(metric) is not None for metric in recognition_metrics),
        f"{name} is missing recognition metrics",
    )
    _require_condition(
        all(resolution.get(metric) is not None for metric in resolution_metrics),
        f"{name} is missing recognition or gold-span resolution",
    )


def _validate_cell_pipeline_status(
    cell_pipeline: dict[str, t.Any], expected_pins: dict[str, str], name: str
) -> None:
    _require_condition(
        cell_pipeline.get("model_pins") == expected_pins,
        f"{name} result manifest is missing expected model roles",
    )
    _require_condition(
        cell_pipeline.get("status") == "complete",
        f"{name} has incomplete phase status",
    )
    _require_condition(
        cell_pipeline.get("phase_status")
        == {"recognition": "complete", "resolution": "complete"},
        f"{name} is missing a completed scoring phase",
    )


def _validate_current_report(path: Path, data: dict[str, t.Any]) -> str:
    """Validate pins, source provenance, metrics, and inspections for a full cell."""
    scoring_view, corpus_name = _validate_report_scope(data)
    spec = CORPORA[corpus_name]
    _validate_source_pins(data, corpus_name, spec)
    results = {item.get("name"): item for item in data.get("pipelines", [])}
    _require_condition(
        set(results) == set(MAIN_PIPELINES),
        f"{corpus_name} must contain exactly the six current pipelines",
    )
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
    first_references = _matching_pipeline_references(
        references_by_pipeline, corpus_name
    )
    scoring_spans = _scoring_span_total(first_references)
    _require_condition(
        data["scoring_spans"] == scoring_spans,
        f"{corpus_name} scoring span count does not match its inspections",
    )
    _validate_dedup_count(data, scoring_view, spec.language, corpus_name)
    return scoring_view


def _matching_pipeline_references(
    references_by_pipeline: dict[str, dict[str, list[dict[str, t.Any]]]],
    corpus_name: str,
) -> dict[str, list[dict[str, t.Any]]]:
    first_references = next(iter(references_by_pipeline.values()))
    _require_condition(
        all(refs == first_references for refs in references_by_pipeline.values()),
        f"{corpus_name} pipelines inspect different gold references",
    )
    return first_references


def _scoring_span_total(
    references_by_document: dict[str, list[dict[str, t.Any]]],
) -> int:
    return sum(
        len({(span["start"], span["end"]) for span in reference})
        for reference in references_by_document.values()
    )


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
        _consider_current_report(path, evidence_dir, candidates, errors)
    return candidates, errors


def _consider_current_report(
    path: Path,
    evidence_dir: Path,
    candidates: dict[tuple[str, str], tuple[str, Path, dict[str, t.Any]]],
    errors: list[str],
) -> None:
    if _is_baseline_path(path.relative_to(evidence_dir)):
        return
    data = _read_candidate_report(path)
    if data is None or data.get("scoring_view") not in SCORING_VIEWS:
        return
    try:
        scoring_view = _validate_current_report(path, data)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{path}: {error}")
        return
    _keep_latest_candidate(path, data, scoring_view, candidates)


def _read_candidate_report(path: Path) -> dict[str, t.Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _keep_latest_candidate(
    path: Path,
    data: dict[str, t.Any],
    scoring_view: str,
    candidates: dict[tuple[str, str], tuple[str, Path, dict[str, t.Any]]],
) -> None:
    key = (scoring_view, data["corpus"])
    started = data.get("environment", {}).get("started_at", "")
    previous = candidates.get(key)
    if previous is None or started >= previous[0]:
        candidates[key] = (started, path, data)


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
    if _live_repository_paths(repository_paths) != set(bundle_files):
        return False
    if not _required_paths_present(repository_paths, required_paths):
        return False
    siblings = getattr(info, "siblings", None)
    if siblings is None:
        return False
    return _bundle_siblings_match(siblings, bundle_files, required_additions)


def _live_repository_paths(repository_paths: t.AbstractSet[str]) -> set[str]:
    return {
        path
        for path in repository_paths
        if path != ".gitattributes" and not path.startswith("archive/")
    }


def _required_paths_present(
    repository_paths: t.AbstractSet[str], required_paths: t.AbstractSet[str] | None
) -> bool:
    return required_paths is None or set(required_paths).issubset(repository_paths)


def _bundle_siblings_match(
    siblings: Sequence[t.Any],
    bundle_files: dict[str, bytes],
    required_additions: dict[str, bytes] | None,
) -> bool:
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
    return {
        source: _archive_destination(source, root)
        for source in existing
        if _should_archive(source)
    }


def _should_archive(source: str) -> bool:
    return source != ".gitattributes" and not source.startswith("archive/")


def _archive_destination(source: str, root: str) -> str:
    if source == "README.md":
        return f"{root}/legacy/README.md"
    if source.startswith("current/"):
        return f"{root}/current/{source.removeprefix('current/')}"
    return f"{root}/legacy/{source}"


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
) -> list[CommitOperation]:
    """Build one Hub transaction that snapshots old files and writes the bundle."""
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
    )

    return [
        *_archive_copy_operations(archived_paths, parent_sha, CommitOperationCopy),
        *_archive_delete_operations(archived_paths, files, CommitOperationDelete),
        *_bundle_add_operations(files, CommitOperationAdd),
    ]


def _archive_copy_operations(
    archived_paths: dict[str, str], parent_sha: str, operation_type: t.Any
) -> list[CommitOperation]:
    return [
        operation_type(
            src_path_in_repo=source,
            path_in_repo=destination,
            src_revision=parent_sha,
        )
        for source, destination in archived_paths.items()
    ]


def _archive_delete_operations(
    archived_paths: dict[str, str],
    files: dict[str, bytes],
    operation_type: t.Any,
) -> list[CommitOperation]:
    return [
        operation_type(path_in_repo=source)
        for source in archived_paths
        if source != "README.md" and source not in files
    ]


def _bundle_add_operations(
    files: dict[str, bytes], operation_type: t.Any
) -> list[CommitOperation]:
    return [
        operation_type(path_in_repo=path, path_or_fileobj=content)
        for path, content in files.items()
    ]


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
    _require_condition(
        not missing_paths,
        f"Hub readback is missing {missing_paths[:10]}",
        RuntimeError,
    )
    inspection_sample = next(path for path in sorted(files) if "/inspection/" in path)
    readback_paths = _readback_sample_paths(inspection_sample)
    for path in readback_paths:
        downloaded = Path(
            api.hf_hub_download(
                repo_id,
                filename=path,
                repo_type="dataset",
                revision=commit_sha,
            )
        ).read_bytes()
        _require_condition(
            hashlib.sha256(downloaded).digest() == hashlib.sha256(files[path]).digest(),
            f"Hub readback differs from committed artifact {path}",
            RuntimeError,
        )


def _readback_sample_paths(inspection_sample: str) -> list[str]:
    return [
        "README.md",
        f"{CURRENT_PREFIX}/{STRICT_RAW_VIEW}/{RESULTS_FILE}",
        f"{CURRENT_PREFIX}/{STRICT_RAW_VIEW}/protocol.json",
        f"{CURRENT_PREFIX}/{EXACT_FULL_RECORD_DEDUP_VIEW}/{RESULTS_FILE}",
        f"{CURRENT_PREFIX}/{EXACT_FULL_RECORD_DEDUP_VIEW}/protocol.json",
        f"{CURRENT_PREFIX}/corpus-sources.md",
        inspection_sample,
    ]


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
    except Exception as error:  # noqa: BLE001 -- commit transport errors can leave outcome uncertain.
        return (
            _reconcile_or_raise(
                api,
                repo_id,
                recovery,
                error,
            ),
            "reconciled",
        )
    commit_sha = getattr(result, "oid", None)
    return _publish_commit_result(
        commit_sha,
        api,
        repo_id,
        recovery,
    )


def _publish_commit_result(
    commit_sha: t.Any,
    api: t.Any,
    repo_id: str,
    recovery: _PublishRecovery,
) -> tuple[str, str]:
    if not isinstance(commit_sha, str) or len(commit_sha) != 40:
        return (
            _reconcile_or_raise(
                api,
                repo_id,
                recovery,
                ValueError("Hub commit did not return a full commit SHA"),
            ),
            "reconciled",
        )
    return commit_sha, "published"


def _reconcile_or_raise(
    api: t.Any,
    repo_id: str,
    recovery: _PublishRecovery,
    error: Exception,
) -> str:
    reconciled_sha = _try_reconcile_publish(
        api,
        repo_id,
        recovery.desired_files,
        recovery.archive_additions,
        recovery.archived_paths,
    )
    if reconciled_sha is None:
        raise error
    return reconciled_sha


def publish(evidence_dir: Path, repo_id: str, *, api: t.Any) -> dict[str, str]:
    """Publish a fully validated bundle once, with retries reconciled by hash."""
    files, _rows = _bundle_files(evidence_dir, repo_id)
    desired_files = dict(files)
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    parent_sha = _validate_parent_sha(info)
    existing = list(
        api.list_repo_files(repo_id, repo_type="dataset", revision=parent_sha)
    )
    if _repository_matches_bundle(info, set(existing), desired_files):
        return _unchanged_publish_result(parent_sha)
    return _publish_changed_bundle(
        api, repo_id, parent_sha, existing, files, desired_files
    )


def _validate_parent_sha(info: t.Any) -> str:
    parent_sha = getattr(info, "sha", None)
    _require_condition(
        isinstance(parent_sha, str) and len(parent_sha) == 40,
        "dataset repository did not return a full parent commit SHA",
    )
    return t.cast(str, parent_sha)


def _unchanged_publish_result(parent_sha: str) -> dict[str, str]:
    return {"parent_sha": parent_sha, "commit_sha": parent_sha, "status": "unchanged"}


def _publish_changed_bundle(
    api: t.Any,
    repo_id: str,
    parent_sha: str,
    existing: Sequence[str],
    files: dict[str, bytes],
    desired_files: dict[str, bytes],
) -> dict[str, str]:
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
        commit_sha, status = _verify_or_reconcile_publish(
            api, repo_id, commit_sha, files, recovery
        )
    return {"parent_sha": parent_sha, "commit_sha": commit_sha, "status": status}


def _verify_or_reconcile_publish(
    api: t.Any,
    repo_id: str,
    commit_sha: str,
    files: dict[str, bytes],
    recovery: _PublishRecovery,
) -> tuple[str, str]:
    try:
        _verify_publish_readback(
            api, repo_id, commit_sha, files, recovery.archived_paths
        )
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
        return reconciled_sha, "reconciled"
    return commit_sha, "published"


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
