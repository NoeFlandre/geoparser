"""
Offline checks that a corpus is fit to score, before any model runs.

Every check reads only parsed documents, so the same checks run on synthetic
fixtures and on the output of any adapter. A check reports a problem and never
repairs one: dropping a span changes the gold a run is scored on, so the person
who selects the corpus decides what happens to each problem.

Region geometry is not checked here. A gold point carries no polygon, and a
region check needs a geometry source that is pinned first.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.benchmark.corpus import Document, GoldSpan

COORDINATE = "coordinate"
OFFSET = "offset"
SURFACE = "surface"
DUPLICATE_SPAN = "duplicate_span"
OVERLAPPING_SPAN = "overlapping_span"
MISSING_GOLD = "missing_gold"
SOURCE_COUNT = "source_count"

MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0
CORPUS = "corpus"


@dataclass(frozen=True)
class Problem:
    """One reason a corpus needs a decision before it is scored."""

    kind: str
    document: str
    detail: str


@dataclass(frozen=True)
class CorpusReport:
    """What a corpus holds, and every problem found in it."""

    documents: int
    gold_spans: int
    problems: tuple[Problem, ...]

    def count(self, kind: str) -> int:
        """Return how many problems of one kind were found."""
        return sum(problem.kind == kind for problem in self.problems)

    @property
    def clean(self) -> bool:
        """Whether no problem was found at all."""
        return not self.problems


def coordinate_detail(span: GoldSpan) -> str | None:
    """Describe why a gold coordinate is not a place on Earth, or return None."""
    latitude, longitude = span.latitude, span.longitude
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return f"non-finite coordinate ({latitude}, {longitude})"
    if abs(latitude) > MAX_LATITUDE or abs(longitude) > MAX_LONGITUDE:
        return f"out-of-range coordinate ({latitude}, {longitude})"
    return None


def _span_problems(document: Document, span: GoldSpan) -> list[Problem]:
    """Return the offset, surface and coordinate problems of one gold span."""
    found: list[Problem] = []
    text = document.text
    if span.start < 0 or span.end <= span.start or span.end > len(text):
        detail = f"offsets {span.start}:{span.end} outside text of length {len(text)}"
        found.append(Problem(OFFSET, document.identifier, detail))
    elif text[span.start : span.end] != span.name:
        found_text = text[span.start : span.end]
        detail = f"{span.start}:{span.end} reads {found_text!r}, not {span.name!r}"
        found.append(Problem(SURFACE, document.identifier, detail))
    coordinate = coordinate_detail(span)
    if coordinate is not None:
        found.append(Problem(COORDINATE, document.identifier, coordinate))
    return found


def _duplicate_problems(document: Document) -> list[Problem]:
    """Flag every gold span whose offsets repeat an earlier one."""
    seen: set[tuple[int, int]] = set()
    found: list[Problem] = []
    for span in document.gold:
        key = (span.start, span.end)
        if key in seen:
            detail = f"{span.start}:{span.end} {span.name!r} repeats an earlier span"
            found.append(Problem(DUPLICATE_SPAN, document.identifier, detail))
        seen.add(key)
    return found


def _overlap_problems(document: Document) -> list[Problem]:
    """Flag every distinct span that starts inside an earlier span."""
    found: list[Problem] = []
    reach = -1
    owner: tuple[int, int] | None = None
    for start, end in sorted({(span.start, span.end) for span in document.gold}):
        if owner is not None and start < reach:
            detail = f"{start}:{end} overlaps {owner[0]}:{owner[1]}"
            found.append(Problem(OVERLAPPING_SPAN, document.identifier, detail))
        if end > reach:
            reach, owner = end, (start, end)
    return found


def document_problems(document: Document) -> list[Problem]:
    """Return every problem found in one document's gold."""
    if not document.gold:
        detail = "no gold toponyms"
        return [Problem(MISSING_GOLD, document.identifier, detail)]
    found: list[Problem] = []
    for span in document.gold:
        found.extend(_span_problems(document, span))
    found.extend(_duplicate_problems(document))
    found.extend(_overlap_problems(document))
    return found


def _source_count_problems(
    documents: int,
    gold_spans: int,
    expected_documents: int | None,
    expected_gold: int | None,
) -> list[Problem]:
    """Compare the loaded totals with the totals a source publishes."""
    found: list[Problem] = []
    if expected_documents is not None and documents != expected_documents:
        detail = f"{documents} documents loaded, source publishes {expected_documents}"
        found.append(Problem(SOURCE_COUNT, CORPUS, detail))
    if expected_gold is not None and gold_spans != expected_gold:
        detail = f"{gold_spans} gold spans loaded, source publishes {expected_gold}"
        found.append(Problem(SOURCE_COUNT, CORPUS, detail))
    return found


def check_corpus(
    documents: Iterable[Document],
    *,
    expected_documents: int | None = None,
    expected_gold: int | None = None,
) -> CorpusReport:
    """
    Check every document of a corpus and its totals against the source.

    Totals are compared only when the source publishes them, so a corpus
    without a published count is never reported as wrong.

    Args:
        documents: The parsed documents, in corpus order
        expected_documents: The number of documents the source publishes
        expected_gold: The number of gold spans the source publishes

    Returns:
        The totals and every problem found
    """
    loaded = list(documents)
    problems = [
        problem for document in loaded for problem in document_problems(document)
    ]
    gold_spans = sum(len(document.gold) for document in loaded)
    problems.extend(
        _source_count_problems(
            len(loaded), gold_spans, expected_documents, expected_gold
        )
    )
    return CorpusReport(len(loaded), gold_spans, tuple(problems))
