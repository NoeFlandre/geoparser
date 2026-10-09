"""Map MasakhaNER 2.0 configurations onto the pinned 85-language target set.

The target list is the upstream sentence-splitting set from #99. It is read from
the PAN-X package so both benchmarks share one canonical list. The WikiANN gaps
come from the PAN-X split manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scripts.panx_benchmark.data import split_manifest, target_languages


def coverage_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one row per canonical target code with its MasakhaNER status."""
    by_code = {row["iso639_1"]: row for row in manifest["languages"] if row["iso639_1"]}
    wikiann_missing = set(split_manifest()["missing_target_languages"])
    rows: list[dict[str, Any]] = []
    for code in target_languages():
        row = by_code.get(code)
        rows.append(
            {
                "code": code,
                "status": "covered" if row else "missing",
                "masakhaner_config": row["config"] if row else None,
                "wikiann_test_missing": code in wikiann_missing,
                "fills_wikiann_gap": row is not None and code in wikiann_missing,
                "readme_counts": row["readme_counts"] if row else None,
            }
        )
    return rows


def excluded_configurations(
    manifest: Mapping[str, Any], included: Sequence[str]
) -> list[dict[str, Any]]:
    """Name each MasakhaNER configuration that the target set does not include."""
    excluded: list[dict[str, Any]] = []
    for row in manifest["languages"]:
        if row["config"] in included:
            continue
        if row["iso639_1"] is None:
            reason = "no ISO 639-1 code, so it cannot match the target list"
        else:
            reason = "its ISO 639-1 code is not in the 85-language target list"
        excluded.append(
            {
                "config": row["config"],
                "iso639_3": row["iso639_3"],
                "iso639_1": row["iso639_1"],
                "name": row["name"],
                "reason": reason,
            }
        )
    return excluded


def coverage_summary(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Count the covered and missing targets and the included configurations."""
    covered = [row for row in rows if row["status"] == "covered"]
    missing = [row for row in rows if row["status"] == "missing"]
    included = [row["masakhaner_config"] for row in covered]
    filled = [row["code"] for row in rows if row["fills_wikiann_gap"]]
    wikiann_gap = [row["code"] for row in rows if row["wikiann_test_missing"]]
    return {
        "canonical_target_languages": len(rows),
        "covered_targets": len(covered),
        "missing_targets": len(missing),
        "covered_codes": [row["code"] for row in covered],
        "missing_codes": [row["code"] for row in missing],
        "masakhaner_configurations": len(manifest["languages"]),
        "included_configurations": included,
        "excluded_configurations": excluded_configurations(manifest, included),
        "wikiann_missing_targets": wikiann_gap,
        "wikiann_gaps_filled": filled,
        "examples_by_split": {
            split: sum(row["readme_counts"][split] for row in covered)
            for split in ("train", "validation", "test")
        },
    }
