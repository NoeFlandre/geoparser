"""
Tests for publishing benchmark evidence to a Hugging Face dataset.

The Hub client is replaced by a recorder, so these tests say what would be
uploaded without any network access or token.
"""

import json

import pytest

from scripts.benchmark.publish import (
    LICENSE,
    PIPELINE_DESCRIPTIONS,
    RESULTS_FILE,
    collect_rows,
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

    def test_tabulates_rows(self):
        """Test that the card shows every result."""
        row = {
            "run": "r",
            "corpus": "c",
            "language": "de",
            "documents": 2,
            "gold_toponyms": 5,
            "pipeline": "hybrid",
            "f1": 0.5,
            "accuracy_at_161km": 0.75,
            "auc": 0.25,
            "elapsed_seconds": 3.0,
            "commit": "abc1234",
        }

        assert (
            "| r | c | de | 2 | 5 | hybrid | 0.500 | 0.750 | 0.250 | 3.0 | abc1234 |"
            in (render_card([row]))
        )


class TestPublish:
    """What reaches the Hub."""

    def test_creates_a_public_dataset_and_uploads_evidence_and_card(self, tmp_path):
        """Test that the repo is public and gets the evidence plus a README."""
        report(tmp_path, "run-a")
        api = FakeApi()

        publish(tmp_path, "me/results", api=api)

        kinds = [call[0] for call in api.calls]
        assert kinds == ["create_repo", "upload_folder", "upload_file", "upload_file"]
        assert [c[2]["path_in_repo"] for c in api.calls[2:]] == [
            RESULTS_FILE,
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
        """Test that upstream, swapped and hybrid each have an explanation."""
        assert set(PIPELINE_DESCRIPTIONS) == {"upstream", "swapped", "hybrid"}

    def test_card_explains_each_pipeline_and_its_models(self):
        """Test that the card lists every pipeline with its models."""
        row = {
            "run": "r",
            "corpus": "c",
            "language": "de",
            "documents": 2,
            "gold_toponyms": 5,
            "pipeline": "hybrid",
            "f1": 0.5,
            "accuracy_at_161km": 0.75,
            "auc": 0.25,
            "elapsed_seconds": 3.0,
            "commit": "abc1234",
            "recognizer": "r-model",
            "resolver": "s-model",
            "reranker": "",
        }

        card = render_card([row])

        assert "## Pipelines" in card
        assert "**hybrid**" in card
        assert "r-model" in card and "s-model" in card
