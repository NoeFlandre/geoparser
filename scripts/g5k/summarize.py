"""
Aggregate per-corpus benchmark reports into one summary for a run.

The benchmark CLI writes summary.md and summary.json for the corpora of one
invocation only. run_node.sh runs one corpus per invocation so that each
corpus is persisted as soon as it finishes, so the run-level summary is built
here from every corpus folder under the output root, in the same table format.

    PYTHONPATH=<tree> python summarize.py --root DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark import report


def rows_from_reports(root: Path) -> tuple[list[dict], dict, dict]:
    """
    Return one summary row per corpus and pipeline, with the shared provenance.

    Args:
        root: The benchmark output root, holding one folder per corpus

    Returns:
        The rows, the commit of the first report, and its environment
    """
    rows: list[dict] = []
    commit: dict = {}
    environment: dict = {}
    for path in sorted(root.glob("*/benchmark-report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not commit:
            commit = {"commit": data["commit"]}
            environment = data.get("environment", {})
        for pipeline in data["pipelines"]:
            rows.append(
                {
                    "corpus": data["corpus"],
                    "language": data["language"],
                    "documents": data["documents"],
                    "gold_toponyms": data["gold_toponyms"],
                    "pipeline": pipeline["name"],
                    "recognition": pipeline["recognition"],
                    "resolution": pipeline["resolution"],
                    "elapsed_seconds": pipeline["elapsed_seconds"],
                }
            )
    return rows, commit, environment


def main() -> int:
    """Write summary.md and summary.json at the output root."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.expanduser()
    rows, provenance, environment = rows_from_reports(root)
    (root / "summary.md").write_text(report.render_summary(rows), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {**provenance, "environment": environment, "rows": rows}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"summarised {len(rows)} rows from {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
