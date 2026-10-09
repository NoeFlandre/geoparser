"""Write the MasakhaNER 2.0 inventory as JSON and Markdown.

The report is generated only from the checked-in manifest and the WikiANN split
manifest. It holds no measurements, so it can be regenerated without any data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.masakhaner_benchmark.coverage import (
    coverage_rows,
    coverage_summary,
)

CAVEATS = (
    "Spans are character offsets in the tokens joined by single spaces. The "
    "upstream tokenization pipeline is not documented, so these spans are not "
    "offsets in the original article text.",
    "Only LOC is scored as a location. PER, ORG and DATE tags are parsed but not "
    "scored.",
    "NER annotations carry no coordinates. A coordinate-resolution claim needs a "
    "separate resolver run against a gazetteer.",
    "The dataset license is conflicting. Resolve it before any publication or "
    "commercial use.",
    "Training overlap is not assessed. The upstream repository ships baseline "
    "result files for several encoders, so any evaluated checkpoint must be "
    "checked against its own model card before its scores are reported.",
    "The example counts are README claims. The split files were not downloaded, "
    "so the counts are not checked against their contents.",
    "The Hugging Face loader reads the raw GitHub main branch. Check each file "
    "against its pinned git blob id before use.",
)


def write_report(output_dir: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write report.json and report.md under output_dir and return their paths."""
    rows = coverage_rows(manifest)
    summary = coverage_summary(manifest, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(_payload(manifest, rows, summary), indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(manifest, rows, summary), encoding="utf-8")
    return json_path, markdown_path


def _payload(
    manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Return the machine-readable report."""
    return {
        "schema_version": 1,
        "retrieved_on": manifest["retrieved_on"],
        "dataset": manifest["dataset"],
        "licenses": manifest["licenses"],
        "license_status": manifest["license_status"],
        "annotation_provenance": manifest["annotation_provenance"],
        "provenance": manifest["provenance"],
        "summary": summary,
        "target_rows": rows,
        "configurations": manifest["languages"],
        "caveats": list(CAVEATS),
    }


def render_markdown(
    manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Render the covered and missing table, the counts and the caveats."""
    dataset = manifest["dataset"]
    lines = [
        "# MasakhaNER 2.0 inventory",
        "",
        f"Retrieved {manifest['retrieved_on']} from pinned metadata only. No split "
        "file was downloaded and no model was run. This page does not report any "
        "benchmark measurement.",
        "",
        "## Pins",
        "",
        f"- Hugging Face dataset `{dataset['huggingface_dataset_id']}` at "
        f"`{dataset['huggingface_revision']}`.",
        f"- GitHub repository `{dataset['github_repository']}` at "
        f"`{dataset['github_commit']}` ({dataset['github_commit_date']}).",
        f"- Data root `{dataset['data_root']}`, with one folder per configuration.",
        "",
        "## License",
        "",
        f"Status: {manifest['license_status']}.",
        "",
        "| Source | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {row['source']} | {row['value']} |" for row in manifest["licenses"]]
    lines += _coverage_section(summary, rows)
    lines += _excluded_section(summary)
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in CAVEATS]
    return "\n".join(lines) + "\n"


def _coverage_section(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Render the counts and the full 85-row covered and missing table."""
    counts = summary["examples_by_split"]
    lines = [
        "",
        "## Coverage of the 85 canonical targets",
        "",
        f"- Covered by MasakhaNER 2.0: {summary['covered_targets']} "
        f"({', '.join(summary['covered_codes'])}).",
        f"- Missing: {summary['missing_targets']} of "
        f"{summary['canonical_target_languages']}.",
        f"- WikiANN test split lacks: {', '.join(summary['wikiann_missing_targets'])}. "
        f"MasakhaNER 2.0 supplies: {', '.join(summary['wikiann_gaps_filled'])}.",
        f"- Covered examples across train, validation and test: "
        f"{counts['train']}, {counts['validation']} and {counts['test']}.",
        "",
        "| Code | Status | MasakhaNER config | WikiANN test split | Train | Validation | Test |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(_coverage_line(row) for row in rows)
    return lines


def _coverage_line(row: dict[str, Any]) -> str:
    """Render one canonical target as a Markdown table row."""
    counts = row["readme_counts"]
    cells = [
        row["code"],
        row["status"],
        row["masakhaner_config"] or "-",
        "missing" if row["wikiann_test_missing"] else "present",
    ]
    if counts is None:
        cells += ["-", "-", "-"]
    else:
        cells += [str(counts["train"]), str(counts["validation"]), str(counts["test"])]
    return "| " + " | ".join(cells) + " |"


def _excluded_section(summary: dict[str, Any]) -> list[str]:
    """Render the MasakhaNER configurations that the target set does not include."""
    lines = [
        "",
        "## Excluded configurations",
        "",
        "| Config | ISO 639-3 | ISO 639-1 | Name | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['config']} | {row['iso639_3']} | {row['iso639_1'] or '-'} | "
        f"{row['name']} | {row['reason']} |"
        for row in summary["excluded_configurations"]
    )
    return lines
