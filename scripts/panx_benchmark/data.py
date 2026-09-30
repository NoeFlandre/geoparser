"""Load the pinned WikiANN test split and align token labels to text spans."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.panx_benchmark.constants import DATASET_ID, DATASET_REVISION

Span = tuple[int, int]
TAG_NAMES = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
PACKAGE_DIR = Path(__file__).parent
TARGET_LANGUAGES_PATH = PACKAGE_DIR / "target_languages.json"
TEST_SPLITS_PATH = PACKAGE_DIR / "wikiann_test_splits.json"


@dataclass(frozen=True, slots=True)
class Example:
    """One test sentence with gold location spans in Python character offsets."""

    language: str
    text: str
    gold_spans: frozenset[Span]
    malformed_location_tags: int = 0


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Materialized test sentences and source row counts."""

    examples_by_language: dict[str, tuple[Example, ...]]
    source_counts: dict[str, int]
    load_seconds: float
    limit_per_language: int | None

    @property
    def evaluated_example_count(self) -> int:
        """Number of examples materialized for this run."""
        return sum(len(examples) for examples in self.examples_by_language.values())

    @property
    def source_example_count(self) -> int:
        """Number of rows in the complete pinned test intersection."""
        return sum(self.source_counts.values())


def read_json(path: Path) -> dict[str, Any]:
    """Read one checked-in benchmark manifest."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        message = f"Expected a JSON object in {path}"
        raise TypeError(message)
    return result


def target_languages() -> tuple[str, ...]:
    """Return the upstream-pinned 85-language sentence-splitting set."""
    manifest = read_json(TARGET_LANGUAGES_PATH)
    languages = manifest.get("language_codes")
    if not isinstance(languages, list) or len(languages) != 85:
        message = "The pinned sentence-splitting list must contain 85 codes"
        raise TypeError(message)
    if len(set(languages)) != len(languages):
        message = "The pinned sentence-splitting list contains duplicates"
        raise ValueError(message)
    return tuple(languages)


def split_manifest() -> dict[str, Any]:
    """Return the test-config and row-count snapshot for the pinned dataset."""
    manifest = read_json(TEST_SPLITS_PATH)
    if manifest.get("dataset_id") != DATASET_ID:
        message = "The WikiANN split manifest names a different dataset"
        raise ValueError(message)
    if manifest.get("dataset_revision") != DATASET_REVISION:
        message = "The WikiANN split manifest names a different revision"
        raise ValueError(message)
    if manifest.get("split") != "test":
        message = "PAN-X evaluation must use the pinned test split"
        raise ValueError(message)
    return manifest


def _span_from_tags(tokens: list[str], tags: list[int]) -> tuple[str, set[Span], int]:
    """Join WikiANN tokens and map its location BIO tags to half-open spans."""
    if len(tokens) != len(tags):
        message = "WikiANN token and tag counts do not match"
        raise ValueError(message)
    if any(not isinstance(token, str) for token in tokens):
        message = "WikiANN tokens must be strings"
        raise TypeError(message)

    text = " ".join(tokens)
    offsets = _token_offsets(tokens)
    expected_length = offsets[-1][1] if offsets else 0
    if len(text) != expected_length:
        message = "WikiANN tokens could not be aligned to the joined text"
        raise ValueError(message)
    spans, malformed = _location_spans(tags, offsets)
    return text, spans, malformed


def _token_offsets(tokens: list[str]) -> list[Span]:
    """Return token offsets for text reconstructed with single spaces."""
    offsets = []
    cursor = 0
    for index, token in enumerate(tokens):
        start = cursor + int(index > 0)
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def _location_spans(tags: list[int], offsets: list[Span]) -> tuple[set[Span], int]:
    """Group location BIO tags, retaining and counting orphan inside tags."""
    spans: set[Span] = set()
    active_start: int | None = None
    active_end = 0
    malformed = 0
    for tag_id, (start, end) in zip(tags, offsets, strict=True):
        if type(tag_id) is not int:
            message = f"WikiANN tag ids must be integers, found {tag_id!r}"
            raise TypeError(message)
        if not 0 <= tag_id < len(TAG_NAMES):
            message = f"Unknown WikiANN tag id: {tag_id!r}"
            raise ValueError(message)
        tag = TAG_NAMES[tag_id]
        if tag == "B-LOC":
            if active_start is not None:
                spans.add((active_start, active_end))
            active_start, active_end = start, end
        elif tag == "I-LOC":
            if active_start is None:
                # Preserve the marked location while reporting the malformed
                # IOB2 transition instead of silently dropping a gold token.
                malformed += 1
                active_start = start
            active_end = end
        elif active_start is not None:
            spans.add((active_start, active_end))
            active_start = None
    if active_start is not None:
        spans.add((active_start, active_end))
    return spans, malformed


def example_from_row(language: str, row: dict[str, Any]) -> Example:
    """Convert one WikiANN row into identical text and location-span input."""
    tokens = row.get("tokens")
    tags = row.get("ner_tags")
    if not isinstance(tokens, list) or not isinstance(tags, list):
        message = "WikiANN row lacks token or NER-tag lists"
        raise TypeError(message)
    text, spans, malformed = _span_from_tags(tokens, tags)
    row_languages = row.get("langs")
    if row_languages and any(code != language for code in row_languages):
        message = f"WikiANN {language!r} row contains a different language"
        raise ValueError(message)
    return Example(language, text, frozenset(spans), malformed)


def load_test_examples(
    cache_dir: Path,
    *,
    limit_per_language: int | None = None,
) -> LoadedDataset:
    """Load the full pinned test splits, optionally retaining a bounded prefix."""
    if limit_per_language is not None and limit_per_language < 1:
        message = "The per-language limit must be positive"
        raise ValueError(message)

    manifest = split_manifest()
    expected_counts = manifest.get("test_examples_by_language")
    languages = manifest.get("eligible_languages")
    if not isinstance(expected_counts, dict) or not isinstance(languages, list):
        message = "The pinned WikiANN test-split manifest is incomplete"
        raise TypeError(message)
    target = set(target_languages())
    if set(languages) - target:
        message = "The WikiANN split manifest contains a non-target language"
        raise ValueError(message)

    from datasets import load_dataset

    started = time.perf_counter()
    examples_by_language: dict[str, tuple[Example, ...]] = {}
    source_counts: dict[str, int] = {}
    for language in languages:
        split = load_dataset(
            DATASET_ID,
            name=language,
            split="test",
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir / "datasets"),
        )
        count = len(split)
        expected_count = expected_counts.get(language)
        if count != expected_count:
            message = (
                f"Pinned WikiANN {language}/test has {count} rows; "
                f"the snapshot records {expected_count}"
            )
            raise ValueError(message)
        source_counts[language] = count
        take = count if limit_per_language is None else min(count, limit_per_language)
        examples_by_language[language] = tuple(
            example_from_row(language, split[index]) for index in range(take)
        )

    return LoadedDataset(
        examples_by_language=examples_by_language,
        source_counts=source_counts,
        load_seconds=time.perf_counter() - started,
        limit_per_language=limit_per_language,
    )
