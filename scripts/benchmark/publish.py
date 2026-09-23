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


def render_card(rows: Sequence[dict[str, t.Any]]) -> str:
    """Return the dataset card, with a results table built from ``rows``."""
    lines = [
        "---",
        f"license: {LICENSE}",
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
        "`runs/` holds each run's reports, checkpoints and Grid'5000 job logs.",
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
