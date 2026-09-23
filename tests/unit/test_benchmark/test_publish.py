"""
Tests for publishing benchmark evidence to a Hugging Face dataset.

The Hub client is replaced by a recorder, so these tests say what would be
uploaded without any network access or token.
"""

import json

import pytest

from scripts.benchmark.publish import (
    CHART_FILES,
    LICENSE,
    PIPELINE_DESCRIPTIONS,
    RESULTS_FILE,
    collect_rows,
    latest_results,
    mark_best,
    publish,
    render_card,
    render_csv,
)


def report(tmp_path, folder, corpus="geovirus", language="en"):
    """Write one corpus report and return its folder."""
    path = tmp_path / folder
    path.mkdir(parents=True)
    (path / "benchmark-report.json").write_text(
        json.dumps(
            {
                "corpus": corpus,
                "language": language,
                "documents": 2,
                "gold_toponyms": 5,
                "commit": "abc1234",
                "pipelines": [
                    {
                        "name": "hybrid",
                        "models": {"recognizer": "r-model", "resolver": "s-model"},
                        "recognition": {"f1": 0.5},
                        "resolution": {"accuracy_at_161km": 0.75, "auc": 0.25},
                        "elapsed_seconds": 3.0,
                    }
                ],
            }
        )
    )
    return path


class FakeApi:
    """Records what would have been sent to the Hub."""

    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs["repo_id"], kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs["repo_id"], kwargs))


class TestCollectRows:
    """Reading every report under an evidence folder."""

    def test_reads_nested_reports_with_their_run(self, tmp_path):
        """Test that each row names the run and corpus it came from."""
        report(tmp_path, "run-a")
        report(tmp_path, "run-b/hipe2020-de", corpus="hipe2020-de", language="de")

        rows = collect_rows(tmp_path)

        assert [(r["run"], r["corpus"], r["language"]) for r in rows] == [
            ("run-a", "geovirus", "en"),
            ("run-b", "hipe2020-de", "de"),
        ]
        assert rows[0]["f1"] == 0.5

    def test_skips_baseline_copies(self, tmp_path):
        """Test that a historical report kept for comparison is not a new row."""
        report(tmp_path, "run-a/baseline-jina")

        assert collect_rows(tmp_path) == []


class TestRenderCard:
    """The dataset card."""

    def test_declares_the_share_alike_license(self):
        """Test that HIPE's CC-BY-NC-SA terms carry over to the results."""
        card = render_card([])

        assert card.startswith("---\n")
        assert f"license: {LICENSE}" in card
        assert LICENSE == "cc-by-nc-sa-4.0"

    def test_shows_one_compact_table_per_metric(self):
        """The card has a small table per metric, not one row per run."""
        rows = [
            _row("r1", "geovirus", "upstream", f1=0.8, acc=0.83),
            _row("r1", "geovirus", "hybrid", f1=0.9, acc=0.83),
        ]

        card = render_card(rows)

        assert "### Resolution: Acc@161km" in card
        assert "### Recognition: F1" in card
        assert "| Corpus | Lang | upstream | hybrid |" in card
        assert "| r1 |" not in card


def _row(run, corpus, pipeline, *, f1, acc, started="2026-01-01T00:00:00"):
    """One collected result row."""
    return {
        "run": run,
        "corpus": corpus,
        "language": "en",
        "documents": 2,
        "gold_toponyms": 5,
        "pipeline": pipeline,
        "recognizer": "",
        "resolver": "",
        "reranker": "",
        "f1": f1,
        "accuracy_at_161km": acc,
        "auc": 0.3,
        "elapsed_seconds": 1.0,
        "commit": "abc",
        "started_at": started,
    }


class TestLatestResults:
    """Which run a cell of the compact tables comes from."""

    def test_the_most_recently_started_run_wins(self):
        """A later rerun replaces an earlier one, whatever its folder name."""
        old = _row("z-partial", "geovirus", "hybrid", f1=0.1, acc=0.1, started="1")
        new = _row("a-full", "geovirus", "hybrid", f1=0.9, acc=0.9, started="2")

        assert latest_results([old, new]) == {("geovirus", "hybrid"): new}

    def test_corpus_names_are_compared_case_insensitively(self):
        """GeoVirus from an older report and geovirus are one corpus."""
        old = _row("a", "GeoVirus", "hybrid", f1=0.1, acc=0.1, started="1")
        new = _row("b", "geovirus", "hybrid", f1=0.9, acc=0.9, started="2")

        assert list(latest_results([old, new])) == [("geovirus", "hybrid")]


class TestMarkBest:
    """Bold for the best score, underline for the second."""

    def test_bolds_the_best_and_underlines_the_second(self):
        """Higher is better by default."""
        assert mark_best([0.5, 0.9, 0.7]) == ["0.500", "**0.900**", "<u>0.700</u>"]

    def test_lower_can_be_better(self):
        """For an error measure the smallest value is best."""
        assert mark_best([0.5, 0.9, 0.7], higher_is_better=False) == [
            "**0.500**",
            "0.900",
            "<u>0.700</u>",
        ]

    def test_ties_share_the_mark(self):
        """Equal scores, as rounded, get the same emphasis."""
        assert mark_best([0.9, 0.9, 0.7]) == ["**0.900**", "**0.900**", "<u>0.700</u>"]

    def test_a_missing_score_is_a_dash(self):
        """A pipeline that did not run on a corpus is shown as such."""
        assert mark_best([None, 0.9]) == ["-", "**0.900**"]

    def test_a_single_score_is_only_bold(self):
        """With one value there is no second best."""
        assert mark_best([0.9]) == ["**0.900**"]


class TestPublish:
    """What reaches the Hub."""

    def test_creates_a_public_dataset_and_uploads_evidence_and_card(self, tmp_path):
        """Test that the repo is public and gets the evidence plus a README."""
        report(tmp_path, "run-a")
        api = FakeApi()

        publish(tmp_path, "me/results", api=api)

        kinds = [call[0] for call in api.calls]
        assert kinds == [
            "create_repo",
            "upload_folder",
            *["upload_file"] * (2 + len(CHART_FILES)),
        ]
        assert [c[2]["path_in_repo"] for c in api.calls[2:]] == [
            RESULTS_FILE,
            *CHART_FILES.values(),
            "README.md",
        ]
        create = api.calls[0][2]
        assert create == {"repo_type": "dataset", "private": False, "exist_ok": True}
        folder = api.calls[1][2]
        assert folder["folder_path"] == str(tmp_path)
        assert folder["path_in_repo"] == "runs"
        assert folder["repo_type"] == "dataset"

    def test_refuses_an_empty_evidence_folder(self, tmp_path):
        """Test that nothing is published when there is nothing to show."""
        with pytest.raises(ValueError):
            publish(tmp_path, "me/results", api=FakeApi())


class TestViewer:
    """What the Hub's dataset viewer reads."""

    def test_card_points_the_viewer_at_the_flat_table(self):
        """Test that the viewer reads one CSV, not the mixed JSON files."""
        card = render_card([])

        assert "configs:" in card
        assert f"path: {RESULTS_FILE}" in card
        assert RESULTS_FILE == "results.csv"

    def test_csv_has_one_row_per_result_with_models(self, tmp_path):
        """Test that each row carries the models that produced it."""
        report(tmp_path, "run-a")

        lines = render_csv(collect_rows(tmp_path)).splitlines()

        assert lines[0].startswith("run,corpus,language,")
        assert "recognizer" in lines[0] and "resolver" in lines[0]
        assert len(lines) == 2
        assert "r-model" in lines[1] and "s-model" in lines[1]


class TestPipelineDescriptions:
    """Saying what each candidate is."""

    def test_every_pipeline_is_described(self):
        """Test that every benchmark pipeline has an explanation."""
        assert set(PIPELINE_DESCRIPTIONS) == {"upstream", "swapped", "hybrid", "prior"}

    def test_card_explains_each_pipeline_and_its_models(self):
        """The pipelines table names every pipeline and its real models."""
        card = render_card([])

        assert "## Pipelines" in card
        assert "**hybrid**" in card
        assert "fastino/gliner2.5-multi-v1" in card
        assert "dguzh/geo-all-MiniLM-L6-v2" in card


class TestCharts:
    """The card shows the results as charts, above the tables."""

    def test_embeds_a_chart_per_metric_when_the_repo_is_known(self):
        """Each chart is linked from the repo it is uploaded to."""
        card = render_card(
            [_row("r1", "geovirus", "hybrid", f1=0.9, acc=0.8)], repo_id="me/results"
        )

        for path in CHART_FILES.values():
            assert (
                f"https://huggingface.co/datasets/me/results/resolve/main/{path}"
                in card
            )

    def test_uploaded_charts_are_svg(self, tmp_path):
        """What is uploaded under the chart paths is an SVG document."""
        report(tmp_path, "run-a")
        api = FakeApi()

        publish(tmp_path, "me/results", api=api)

        uploads = {c[2]["path_in_repo"]: c[2]["path_or_fileobj"] for c in api.calls[2:]}
        for path in CHART_FILES.values():
            assert uploads[path].startswith(b"<svg")


class TestMinimalCard:
    """A short card: benchmarks, pipelines, results."""

    def test_describes_each_benchmark_family_once(self):
        """Every corpus family that ran gets one line, with its languages."""
        rows = [
            _row("r", "geovirus", "hybrid", f1=0.9, acc=0.8),
            _row("r", "hipe2020-de", "hybrid", f1=0.9, acc=0.8),
            _row("r", "hipe2020-fr", "hybrid", f1=0.9, acc=0.8),
        ]
        rows[1]["language"], rows[2]["language"] = "de", "fr"

        card = render_card(rows)

        assert "## Benchmarks" in card
        assert card.count("| GeoVirus |") == 1
        assert card.count("| HIPE-2020 |") == 1
        assert "| HIPE-2020 | de, fr |" in card

    def test_lists_pipelines_as_a_table(self):
        """One row per pipeline, naming its recognizer and resolver."""
        card = render_card([_row("r", "geovirus", "hybrid", f1=0.9, acc=0.8)])

        assert "| Pipeline | Recognition | Resolution |" in card

    def test_stays_short(self):
        """The card is a summary; the detail lives in the files."""
        rows = [
            _row("r", corpus, pipeline, f1=0.9, acc=0.8)
            for corpus in ("geovirus", "hipe2020-de", "newseye-fi")
            for pipeline in ("upstream", "hybrid", "prior")
        ]

        assert len(render_card(rows).splitlines()) < 90


class TestMetricCoverage:
    """The card plots the metrics that matter, not accuracy alone."""

    def test_charts_cover_accuracy_auc_median_error_and_f1(self):
        """Four charts, each a separate file."""
        assert set(CHART_FILES) == {
            "accuracy_at_161km",
            "auc",
            "median_error_km",
            "f1",
        }

    def test_rows_carry_the_distance_errors(self, tmp_path):
        """Median and mean errors are collected for the charts and the CSV."""
        report(tmp_path, "run-a")

        (row,) = collect_rows(tmp_path)

        assert "median_error_km" in row and "mean_error_km" in row

    def test_auc_is_marked_lower_is_better(self):
        """In the AUC table the smallest value is bold."""
        rows = [
            _row("r", "geovirus", "upstream", f1=0.8, acc=0.8),
            _row("r", "geovirus", "hybrid", f1=0.8, acc=0.8),
        ]
        rows[0]["auc"], rows[1]["auc"] = 0.2, 0.4

        card = render_card(rows)

        assert "| geovirus | en | **0.200** | <u>0.400</u> |" in card


def test_every_registered_corpus_family_is_described():
    """A new corpus cannot reach the card without a description."""
    from scripts.benchmark.corpora import CORPORA
    from scripts.benchmark.publish import BENCHMARKS

    assert {name.split("-")[0] for name in CORPORA} <= set(BENCHMARKS)
