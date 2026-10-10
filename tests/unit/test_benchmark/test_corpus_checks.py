"""
Tests for the offline corpus checks, on tiny synthetic documents.

The text is fixed and its offsets are written out by hand, so an expected
problem never comes from the code under test. "Paris" sits at 0:5 and 29:34,
"Versailles" at 14:24, and the text is 35 characters long.
"""

import math

import pytest

from scripts.benchmark import corpus_checks
from scripts.benchmark.corpus import Document, GoldSpan

TEXT = "Paris is near Versailles and Paris."
PARIS = GoldSpan(0, 5, "Paris", 48.85, 2.35)
VERSAILLES = GoldSpan(14, 24, "Versailles", 48.80, 2.13)
PARIS_AGAIN = GoldSpan(29, 34, "Paris", 48.85, 2.35)


def _document(*gold: GoldSpan, text: str = TEXT) -> Document:
    """Build one synthetic document with the given gold spans."""
    return Document("doc-1", text, tuple(gold))


def _kinds(report: corpus_checks.CorpusReport) -> list[str]:
    """The kinds of every problem, in the order they were found."""
    return [problem.kind for problem in report.problems]


class TestCoordinates:
    """A gold coordinate must name a point on the globe."""

    @pytest.mark.parametrize(
        ("latitude", "longitude"),
        [(90.0, 180.0), (-90.0, -180.0), (0.0, 0.0), (48.85, 2.35)],
    )
    def test_accepts_points_on_the_globe_including_its_edges(self, latitude, longitude):
        """The poles and antimeridian are valid coordinates."""
        span = GoldSpan(0, 5, "Paris", latitude, longitude)

        assert corpus_checks.coordinate_detail(span) is None

    @pytest.mark.parametrize(
        ("latitude", "longitude"),
        [(90.5, 2.35), (-90.01, 2.35), (48.85, 180.5), (48.85, -181.0)],
    )
    def test_rejects_coordinates_beyond_the_range(self, latitude, longitude):
        """A latitude past a pole or a longitude past the antimeridian is wrong."""
        span = GoldSpan(0, 5, "Paris", latitude, longitude)

        assert corpus_checks.coordinate_detail(span) is not None

    @pytest.mark.parametrize(
        ("latitude", "longitude"),
        [(math.nan, 2.35), (48.85, math.nan), (math.inf, 2.35), (48.85, -math.inf)],
    )
    def test_rejects_non_finite_coordinates(self, latitude, longitude):
        """NaN and infinity pass a plain range test, so they are checked first."""
        span = GoldSpan(0, 5, "Paris", latitude, longitude)

        assert corpus_checks.coordinate_detail(span) is not None

    def test_a_document_reports_its_bad_coordinate(self):
        """The problem names the document it belongs to."""
        bad = GoldSpan(0, 5, "Paris", 123.0, 2.35)

        report = corpus_checks.check_corpus([_document(bad)])

        assert _kinds(report) == [corpus_checks.COORDINATE]
        assert report.problems[0].document == "doc-1"


class TestSpanAlignment:
    """A gold span must point at its own surface form in the text."""

    def test_an_aligned_span_is_clean(self):
        """Offsets that read the surface name pass."""
        report = corpus_checks.check_corpus([_document(PARIS, VERSAILLES)])

        assert report.clean

    def test_a_negative_start_is_an_offset_problem(self):
        """Python slicing would silently read from the end of the text."""
        span = GoldSpan(-1, 4, "Pari", 48.85, 2.35)

        report = corpus_checks.check_corpus([_document(span)])

        assert _kinds(report) == [corpus_checks.OFFSET]

    def test_an_end_past_the_text_is_an_offset_problem(self):
        """A span that runs off the end of the text cannot be aligned."""
        span = GoldSpan(30, 40, "Paris.", 48.85, 2.35)

        report = corpus_checks.check_corpus([_document(span)])

        assert _kinds(report) == [corpus_checks.OFFSET]

    def test_an_empty_span_is_an_offset_problem(self):
        """A span must cover at least one character."""
        span = GoldSpan(5, 5, "", 48.85, 2.35)

        report = corpus_checks.check_corpus([_document(span)])

        assert _kinds(report) == [corpus_checks.OFFSET]

    def test_offsets_that_read_another_word_are_a_surface_problem(self):
        """The offsets land inside the text but on the wrong word."""
        shifted = GoldSpan(14, 24, "Paris", 48.85, 2.35)

        report = corpus_checks.check_corpus([_document(shifted)])

        assert _kinds(report) == [corpus_checks.SURFACE]
        assert "'Versailles'" in report.problems[0].detail


class TestDuplicatesAndOverlaps:
    """Repeated and overlapping spans make the gold ambiguous."""

    def test_a_repeated_span_is_one_duplicate(self):
        """The second copy of the same offsets is the problem, not the first."""
        report = corpus_checks.check_corpus([_document(PARIS, PARIS)])

        assert _kinds(report) == [corpus_checks.DUPLICATE_SPAN]

    def test_the_same_word_at_two_offsets_is_not_a_duplicate(self):
        """Two mentions of one name are two gold spans, as in the fixture text."""
        report = corpus_checks.check_corpus([_document(PARIS, PARIS_AGAIN)])

        assert report.clean

    def test_a_partial_overlap_is_flagged(self):
        """Two distinct spans that share characters cannot both be scored."""
        first = GoldSpan(0, 5, "Paris", 48.85, 2.35)
        second = GoldSpan(3, 8, "is ne", 0.0, 0.0)

        report = corpus_checks.check_corpus([_document(first, second)])

        assert corpus_checks.OVERLAPPING_SPAN in _kinds(report)

    def test_a_nested_span_is_flagged_even_when_not_adjacent(self):
        """A long span overlaps every span inside it, not only its neighbour."""
        outer = GoldSpan(0, 10, "x", 1.0, 1.0)
        first_inner = GoldSpan(2, 4, "x", 1.0, 1.0)
        second_inner = GoldSpan(5, 7, "x", 1.0, 1.0)

        report = corpus_checks.check_corpus(
            [_document(outer, first_inner, second_inner)]
        )

        assert report.count(corpus_checks.OVERLAPPING_SPAN) == 2

    def test_touching_spans_do_not_overlap(self):
        """End and start equal means adjacent words, not shared characters."""
        left = GoldSpan(0, 5, "Paris", 48.85, 2.35)
        right = GoldSpan(5, 9, " is ", 1.0, 1.0)

        report = corpus_checks.check_corpus([_document(left, right)])

        assert report.count(corpus_checks.OVERLAPPING_SPAN) == 0


class TestMissingGold:
    """A document without gold cannot tell a hit from a miss."""

    def test_a_document_without_gold_is_reported(self):
        """Missing gold is a problem, and it is counted as a document."""
        report = corpus_checks.check_corpus([_document()])

        assert _kinds(report) == [
            corpus_checks.MISSING_GOLD,
            corpus_checks.EMPTY_CORPUS,
        ]
        assert report.documents == 1
        assert report.gold_spans == 0

    def test_one_empty_document_among_good_ones_is_still_reported(self):
        """Only the document without gold is flagged."""
        report = corpus_checks.check_corpus(
            [_document(PARIS), Document("doc-2", TEXT, ())]
        )

        assert [problem.document for problem in report.problems] == ["doc-2"]


class TestSourceCounts:
    """Loaded totals are compared with what the source publishes."""

    def test_totals_are_counted_from_the_documents(self):
        """Two documents with two and one gold spans make 2 documents, 3 spans."""
        report = corpus_checks.check_corpus(
            [_document(PARIS, VERSAILLES), Document("doc-2", TEXT, (PARIS_AGAIN,))]
        )

        assert report.documents == 2
        assert report.gold_spans == 3

    def test_an_unpublished_count_is_never_compared(self):
        """With no expected totals, nothing is reported as a count mismatch."""
        report = corpus_checks.check_corpus([_document(PARIS)])

        assert corpus_checks.SOURCE_COUNT not in _kinds(report)

    def test_matching_published_totals_are_clean(self):
        """Loaded totals equal to the published ones add no problem."""
        report = corpus_checks.check_corpus(
            [_document(PARIS, VERSAILLES)], expected_documents=1, expected_gold=2
        )

        assert report.clean

    def test_each_mismatched_total_is_reported_on_its_own(self):
        """A short document count and a short gold count are two problems."""
        report = corpus_checks.check_corpus(
            [_document(PARIS)], expected_documents=2, expected_gold=3
        )

        assert _kinds(report) == [corpus_checks.SOURCE_COUNT] * 2
        assert "1 documents loaded, source publishes 2" in report.problems[0].detail
        assert "1 gold spans loaded, source publishes 3" in report.problems[1].detail

    def test_an_empty_corpus_is_a_problem_not_a_clean_report(self):
        """Zero documents has nothing to score, so it can never pass the gate."""
        report = corpus_checks.check_corpus([])

        assert (report.documents, report.gold_spans, report.clean) == (0, 0, False)
        assert [(p.kind, p.detail) for p in report.problems] == [
            (corpus_checks.EMPTY_CORPUS, "no documents")
        ]

    def test_documents_without_any_gold_span_are_an_empty_corpus(self):
        """Every document lacking gold is reported, and the corpus is empty too."""
        report = corpus_checks.check_corpus(
            [Document("doc-1", TEXT, ()), Document("doc-2", TEXT, ())]
        )

        assert report.gold_spans == 0
        assert report.count(corpus_checks.MISSING_GOLD) == 2
        assert report.count(corpus_checks.EMPTY_CORPUS) == 1
        assert corpus_checks.EMPTY_CORPUS in _kinds(report)

    def test_a_corpus_with_gold_is_not_empty(self):
        """One gold span anywhere is enough to leave the empty-corpus check."""
        report = corpus_checks.check_corpus([_document(PARIS)])

        assert corpus_checks.EMPTY_CORPUS not in _kinds(report)


class TestDocumentIdentifiers:
    """Each document needs its own identifier, or predictions overwrite one another."""

    def test_distinct_identifiers_are_clean(self):
        """Documents with different identifiers raise no duplicate problem."""
        report = corpus_checks.check_corpus(
            [Document("a", TEXT, (PARIS,)), Document("b", TEXT, (PARIS_AGAIN,))]
        )

        assert corpus_checks.DUPLICATE_DOCUMENT not in _kinds(report)

    def test_a_repeated_identifier_is_one_problem_naming_its_count(self):
        """Three documents under one identifier give one problem: it appears 3 times."""
        report = corpus_checks.check_corpus(
            [_document(PARIS), _document(VERSAILLES), _document(PARIS_AGAIN)]
        )

        assert report.count(corpus_checks.DUPLICATE_DOCUMENT) == 1
        assert report.problems[-1].detail == "identifier appears 3 times"
