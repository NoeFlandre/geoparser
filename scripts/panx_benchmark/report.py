"""Write a reviewable Markdown report and machine-readable PAN-X evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _number(value: float | int | None) -> str:
    """Format a finite benchmark measurement or mark it unavailable."""
    return "n/a" if value is None else f"{float(value):.4f}"


def _model_summary(model: dict[str, Any]) -> str:
    """Render one model's score and timing row."""
    macro = model["macro"]
    language_count = len(model["documented_languages"])
    coverage = str(language_count) if language_count else "unspecified"
    return (
        f"| `{model['model_id']}` | {coverage} | "
        f"{model['evaluated_examples']} | {_number(macro['precision'])} | "
        f"{_number(macro['recall'])} | {_number(macro['f1'])} | "
        f"{_number(model['steady_examples_per_second'])} | "
        f"{_number(model['checkpoint_download_seconds'])} | "
        f"{_number(model['model_load_seconds'])} |"
    )


def _language_rows(model: dict[str, Any]) -> list[str]:
    """Render measured languages, including explicit untested rows."""
    rows = []
    for language, result in model["per_language"].items():
        metrics = result["metrics"]
        if metrics is None:
            values = ("—", "—", "—", "—", "—")
        else:
            values = (
                str(result["evaluated_examples"]),
                str(metrics["gold_spans"]),
                _number(metrics["precision"]),
                _number(metrics["recall"]),
                _number(metrics["f1"]),
            )
        rows.append(
            f"| `{language}` | {result['status']} | {result['documented_support']} | "
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} |"
        )
    return rows


def _document_lines(result: dict[str, Any]) -> list[str]:
    """Render benchmark identity, source and resource provenance."""
    dataset = result["dataset"]
    kind = result["evaluation_kind"]
    sample_note = (
        "**Feasibility sample only:** metrics below are descriptive of the "
        "bounded prefix, not a full-test quality comparison."
        if kind == "bounded_feasibility_sample"
        else "Full pinned test intersection was evaluated."
    )
    return [
        "# PAN-X / WikiANN place recognition",
        "",
        sample_note,
        "",
        f"- Repository commit: `{result['repository_commit']}`",
        f"- Dataset: `{dataset['id']}` at `{dataset['revision']}`, split `{dataset['split']}`",
        f"- Test intersection: {len(dataset['eligible_languages'])} languages, "
        f"{dataset['source_test_examples']:,} rows; this run used "
        f"{dataset['evaluated_examples']:,} rows",
        f"- Target languages absent from this test split: "
        f"{', '.join(f'`{language}`' for language in dataset['missing_target_languages'])}",
        f"- Hardware: `{result['hardware']['platform']}`, "
        f"{result['hardware']['torch_threads']} CPU threads, "
        f"CUDA available: `{result['hardware']['cuda_available']}`",
        f"- Seed: `{result['seed']}`; batch size: `{result['evaluation']['batch_size']}`; "
        "no training or fine-tuning",
        f"- Dataset acquisition/materialization: {dataset['data_load_seconds']:.2f}s",
        "",
        "The canonical 85-code list is copied from the upstream sentence-splitting "
        "guide at revision "
        f"`{result['language_list']['source_revision']}`. The exact upstream source "
        f"is [{result['language_list']['source_path']}]"
        f"({result['language_list']['source_url']}).",
        "",
        "## Model summary",
        "",
        "| Recognizer | Documented language count | Evaluated sentences | Macro P | Macro R | Macro F1 | "
        "Steady sentences/s | Checkpoint fetch s | Model load s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[_model_summary(model) for model in result["models"]],
        "",
        "Model fetch time, local model load time, warmup, data acquisition, and "
        "steady inference are reported separately. Throughput excludes warmup and "
        "data loading. The GLiNER threshold is fixed at 0.5; no test-set tuning was done.",
        "",
        "## Coverage and overlap",
        "",
    ]


def _model_coverage_lines(model: dict[str, Any]) -> list[str]:
    """Render one model's coverage claims and per-language measurements."""
    return [
        f"### `{model['model_id']}`",
        "",
        f"- Revision: `{model['revision']}`; model card: "
        f"[{model['model_id']}]({model['model_card_url']})",
        f"- Language coverage: {model['coverage_note']}",
        f"- Training data: {model['training_data_note']}",
        f"- WikiANN overlap: {model['training_overlap_note']}",
        f"- Location mapping: {model['location_mapping']}",
        f"- Warmup: {model['warmup_examples']} sentences in "
        f"{model['warmup_seconds']:.4f}s; steady inference: "
        f"{model['steady_inference_seconds']:.4f}s",
        f"- Linear full-matrix inference estimate from this run: "
        f"{_number(model['full_matrix_estimated_inference_seconds'])}s",
        "",
        "| Language | Evaluation status | Coverage status | Sentences | Gold LOC | P | R | F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *_language_rows(model),
        "",
    ]


def _aggregate_lines(result: dict[str, Any]) -> list[str]:
    """Render aggregate scores and full-matrix feasibility context."""
    lines = [
        "## Aggregate scores",
        "",
        "Macro scores are unweighted means over evaluated languages; micro scores "
        "aggregate exact-span counts. Undefined precision/recall/F1 values use zero.",
        "",
        "| Recognizer | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in result["models"]:
        macro, micro = model["macro"], model["micro"]
        lines.append(
            f"| `{model['key']}` | {_number(macro['precision'])} | "
            f"{_number(macro['recall'])} | {_number(macro['f1'])} | "
            f"{_number(micro['precision'])} | {_number(micro['recall'])} | "
            f"{_number(micro['f1'])} |"
        )
    lines.extend(
        [
            "",
            f"Estimated total CPU inference for the complete matrix: "
            f"{_number(result['estimated_full_matrix_inference_seconds'])}s. "
            f"{result['full_matrix_estimate_note']}",
            "",
            "The upstream spaCy model is scored on English only. Its other 81 "
            "available languages are explicitly marked as not evaluated; it is "
            "not treated as a multilingual competitor.",
            "",
        ]
    )
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    """Render the complete result with sample caveats and provenance."""
    lines = _document_lines(result)
    for model in result["models"]:
        lines.extend(_model_coverage_lines(model))
    lines.extend(_aggregate_lines(result))
    return "\n".join(lines)


def write_reports(output_dir: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and Markdown reports without publishing them externally."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path
