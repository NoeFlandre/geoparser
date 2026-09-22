"""
Tests for scoring and rendering a benchmark run.

A bug here does not crash, it prints a plausible number, so the round trip and
the arithmetic are pinned rather than eyeballed.
"""

from geoparser.evaluation import Annotation
from scripts.benchmark.report import (
    PipelineResult,
    deserialize,
    render_markdown,
    score_recognition,
    score_resolution,
    serialize,
)

ZURICH = (47.3769, 8.5417)


def located(start, end, latitude, longitude, document="d"):
    """Build an annotation that places a span."""
    return Annotation(start, end, None, document, latitude, longitude)


class TestSerialization:
    """Annotations survive the checkpoint round trip."""

    def test_round_trips_a_located_annotation(self):
        """Test that nothing is lost through JSON."""
        annotation = Annotation(1, 5, "geonames:42", "doc7", *ZURICH)

        assert deserialize(serialize(annotation)) == annotation

    def test_round_trips_an_unresolved_annotation(self):
        """Test the common case of a recognized but unplaced span."""
        annotation = Annotation(1, 5, None, "doc7")

        assert deserialize(serialize(annotation)) == annotation

    def test_serializes_to_plain_data(self):
        """Test that the payload is JSON-safe."""
        payload = serialize(Annotation(1, 5, None, "d", *ZURICH))

        assert set(payload) == {
            "start",
            "end",
            "identifier",
            "document_id",
            "latitude",
            "longitude",
        }


class TestScoring:
    """The two score bundles the report is built from."""

    def test_recognition_reports_the_three_figures(self):
        """Test a perfect recognition run."""
        gold = [Annotation(0, 5, None, "d")]

        scores = score_recognition(gold, gold)

        assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_resolution_reports_the_four_figures(self):
        """Test a perfect resolution run."""
        gold = [located(0, 5, *ZURICH)]

        scores = score_resolution(gold, gold)

        assert scores["accuracy_at_161km"] == 1.0
        assert scores["mean_error_km"] == 0.0
        assert scores["median_error_km"] == 0.0
        assert scores["auc"] == 0.0

    def test_resolution_charges_an_unplaced_toponym(self):
        """Test that abstaining is not free."""
        scores = score_resolution([located(0, 5, *ZURICH)], [])

        assert scores["accuracy_at_161km"] == 0.0
        assert scores["auc"] == 1.0


class TestRenderMarkdown:
    """The report a reader actually sees."""

    def result(self, name="swapped"):
        """Build a fully populated result."""
        return PipelineResult(
            name=name,
            models={"resolver": "jina", "reranker": "jina-rerank"},
            recognition={"precision": 0.8, "recall": 0.9, "f1": 0.85},
            resolution={
                "accuracy_at_161km": 0.5,
                "mean_error_km": 123.4,
                "median_error_km": 12.3,
                "auc": 0.25,
            },
            elapsed_seconds=42.0,
            device="cuda",
        )

    def render(self, results):
        """Render with fixed header values."""
        return render_markdown(
            results,
            corpus_name="GeoVirus",
            documents=2,
            gold_toponyms=14,
            gazetteer="geonames",
            min_similarity=0.0,
        )

    def test_states_the_run_parameters(self):
        """Test that the header carries what the numbers depend on."""
        text = self.render([self.result()])

        assert "- Documents: 2" in text
        assert "- Gold toponyms: 14" in text
        assert "- Gazetteer: geonames" in text
        assert "- Shared abstention threshold: 0.0" in text

    def test_lists_each_pipeline_in_both_tables(self):
        """Test that two pipelines are comparable side by side."""
        text = self.render([self.result("upstream"), self.result("swapped")])

        assert text.count("| upstream |") == 2
        assert text.count("| swapped |") == 2

    def test_lists_the_hybrid_pipeline_in_both_tables(self):
        """Test that the new pipeline name renders in both metric tables."""
        text = self.render([self.result("upstream"), self.result("hybrid")])

        assert text.count("| upstream |") == 2
        assert text.count("| hybrid |") == 2

    def test_names_the_models(self):
        """Test that the report says what actually ran."""
        text = self.render([self.result()])

        assert "resolver=jina" in text
        assert "reranker=jina-rerank" in text

    def test_omits_a_table_row_for_a_phase_that_did_not_run(self):
        """Test a resolution-only run."""
        result = self.result()
        result.recognition = {}

        text = self.render([result])

        assert "| swapped | cuda | 0.500" in text
        assert "| swapped | cuda | 0.800" not in text

    def test_explains_the_abstention_threshold(self):
        """Test that the reader is told model score scales may differ."""
        text = self.render([self.result()])

        assert "different similarity scales" in text

    def test_says_resolution_is_scored_on_gold_spans(self):
        """Test that the experimental design is stated, not implied."""
        text = self.render([self.result()])

        assert "gold spans" in text
