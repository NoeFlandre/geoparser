"""
Run the native spaCy baselines on the pinned WikiANN test splits.

    python -m scripts.spacy_native_baselines.run --cache DIR --wheels DIR --output DIR

For each native language in the roster, the pinned wheel is downloaded (or
reused from --wheels), its size and SHA-256 are checked against the roster, and
only then is it installed with ``--no-deps``. The loaded pipeline must have the
roster's label scheme. The WikiANN test splits are the pinned revision, checked
against the snapshot's row counts by the PAN-X loader. Scoring is the existing
``evaluate_baselines``, which keeps matched, transfer and unsupported apart.

This is a new run of the native baselines. Earlier WikiANN results are not
reproducible (see the protocol), so the output records its own inputs and is
not compared with them.

Refused on any mismatch: a wheel whose size or SHA-256 differs from the roster,
a pipeline whose version or NER labels differ from the roster, or a WikiANN
split whose row count differs from the snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from geoparser.modules.recognizers.spacy_native import NativeSpacyRecognizer
from scripts.panx_benchmark import constants
from scripts.panx_benchmark.data import load_test_examples
from scripts.spacy_native_baselines.evaluation import evaluate_baselines
from scripts.spacy_native_baselines.loading import (
    LabelSchemeError,
    MissingPipelineError,
    load_pipeline,
)
from scripts.spacy_native_baselines.roster import (
    ROSTER_PATH,
    NativePipeline,
    Roster,
    load_roster,
)

DOWNLOAD_TIMEOUT_SECONDS = 600
# A language whose pipeline cannot be fetched, verified, installed or loaded is
# recorded as failed with its reason, and the run goes on. The English control
# is different: without it no transfer result exists, so it still aborts.
RECORDED_FAILURES = (
    WheelMismatchError,
    MissingPipelineError,
    LabelSchemeError,
    ImportError,
    OSError,
    subprocess.CalledProcessError,
)
CHUNK_BYTES = 1 << 20


class WheelMismatchError(RuntimeError):
    """A downloaded wheel does not match the roster."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheel(path: Path, pipeline: NativePipeline) -> str:
    """Return the verified SHA-256 of a wheel, or raise if it is not the pinned one."""
    if pipeline.sha256 is None:
        message = f"{pipeline.package} has no pinned sha256 in the roster"
        raise WheelMismatchError(message)
    size = path.stat().st_size
    if size != pipeline.wheel_bytes:
        message = f"{pipeline.wheel}: {size} bytes, roster pins {pipeline.wheel_bytes}"
        raise WheelMismatchError(message)
    digest = _sha256(path)
    if digest != pipeline.sha256:
        message = f"{pipeline.wheel}: sha256 {digest}, roster pins {pipeline.sha256}"
        raise WheelMismatchError(message)
    return digest


def fetch_wheel(pipeline: NativePipeline, wheels: Path) -> Path:
    """Download a pinned wheel into ``wheels`` unless a verified copy is there."""
    wheels.mkdir(parents=True, exist_ok=True)
    target = wheels / pipeline.wheel
    if target.exists():
        try:
            verify_wheel(target, pipeline)
            return target
        except WheelMismatchError:
            target.unlink()
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(  # noqa: S310 - fixed https URL from the roster
        pipeline.wheel_url, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response, partial.open("wb") as handle:
        for block in iter(lambda: response.read(CHUNK_BYTES), b""):
            handle.write(block)
    try:
        verify_wheel(partial, pipeline)
    except WheelMismatchError:
        partial.unlink()
        raise
    partial.replace(target)
    return target


def install_wheel(path: Path) -> None:
    """Install one verified wheel into this interpreter's environment, without deps."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", str(path)],
        check=True,
    )


def _pipeline_record(pipeline: NativePipeline, wheel: Path) -> dict[str, Any]:
    return {
        "language": pipeline.language,
        "package": pipeline.package,
        "version": pipeline.version,
        "wheel": pipeline.wheel,
        "wheel_url": pipeline.wheel_url,
        "wheel_bytes": pipeline.wheel_bytes,
        "sha256": pipeline.sha256,
        "license": pipeline.license,
        "place_label_map": pipeline.place_label_map,
        "ner_labels": list(pipeline.ner_labels),
        "verified_wheel_sha256": _sha256(wheel),
    }


def _environment() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    import spacy

    facts["spacy"] = spacy.__version__
    return facts


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Native spaCy baselines on WikiANN",
        "",
        f"WikiANN revision `{constants.DATASET_REVISION}`, split `test`.",
        "",
        "## Matched (native pipeline, own language)",
        "",
        "| Language | Sentences | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for language, scores in sorted(result["matched"].items()):
        lines.append(
            f"| {language} | {scores.get('sentences', '-')} "
            f"| {scores['precision']:.3f} | {scores['recall']:.3f} "
            f"| {scores['f1']:.3f} |"
        )
    lines += [
        "",
        "## Transfer (English control, labelled as cross-language)",
        "",
        "| Language | Sentences | F1 |",
        "| --- | ---: | ---: |",
    ]
    for language, scores in sorted(result["transfer"].items()):
        lines.append(f"| {language} | {scores.get('sentences', '-')} | {scores['f1']:.3f} |")
    lines += [
        "",
        f"Unsupported languages with no native pipeline: {len(result['unsupported'])}.",
        "",
        "## Failed (supported by the roster, not scored)",
        "",
    ]
    for language, reason in sorted(result.get("failed", {}).items()):
        lines.append(f"- {language}: {reason}")
    lines.append("")
    return "\n".join(lines)


def run(
    roster: Roster,
    *,
    cache: Path,
    wheels: Path,
    output: Path,
    limit: int | None,
    languages: list[str] | None,
    batch_size: int,
) -> dict[str, Any]:
    """Install, verify and score the native baselines, and write the outputs."""
    started = time.perf_counter()
    selected = languages or sorted(roster.pipelines)
    pipelines = {language: roster.pipelines[language] for language in selected}
    if roster.english_control.language not in pipelines:
        pipelines[roster.english_control.language] = roster.english_control

    recognizers: dict[str, NativeSpacyRecognizer] = {}
    records: list[dict[str, Any]] = []
    failed: dict[str, str] = {}
    control = roster.english_control.language
    for language, pipeline in sorted(pipelines.items()):
        try:
            wheel = fetch_wheel(pipeline, wheels)
            install_wheel(wheel)
            nlp = load_pipeline(pipeline)
        except RECORDED_FAILURES as error:
            if language == control:
                raise
            failed[language] = f"{type(error).__name__}: {error}"
            continue
        records.append(_pipeline_record(pipeline, wheel))
        recognizers[language] = NativeSpacyRecognizer(
            nlp,
            language=language,
            package=pipeline.package,
            version=pipeline.version,
            place_label_map=pipeline.place_label_map,
            batch_size=batch_size,
        )

    dataset = load_test_examples(cache / "wikiann", limit_per_language=limit)
    examples = dataset.examples_by_language
    if languages:
        examples = {language: examples[language] for language in languages}
    english = recognizers[control]
    matched_arms = dict(recognizers)
    result = evaluate_baselines(
        examples,
        matched_arms,
        english,
        roster,
        batch_size=batch_size,
    )
    document = {
        "dataset": {
            "id": constants.DATASET_ID,
            "revision": constants.DATASET_REVISION,
            "split": constants.DATASET_SPLIT,
            "limit_per_language": limit,
            "source_counts": dataset.source_counts,
        },
        "roster_sha256": _sha256(ROSTER_PATH),
        "pipelines": records,
        "failed": failed,
        "environment": _environment(),
        "seconds": round(time.perf_counter() - started, 1),
        **result,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "native-spacy-results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "native-spacy-results.md").write_text(_markdown(document), encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    """Run the native baselines from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--language", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args(argv)
    run(
        load_roster(),
        cache=arguments.cache.expanduser(),
        wheels=arguments.wheels.expanduser(),
        output=arguments.output.expanduser(),
        limit=arguments.limit,
        languages=arguments.language,
        batch_size=arguments.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
