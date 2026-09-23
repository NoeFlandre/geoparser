"""
Tests for publishing benchmark evidence to a Hugging Face dataset.

The Hub client is replaced by a recorder, so these tests say what would be
uploaded without any network access or token.
"""

import json

import pytest

from scripts.benchmark.publish import (
    LICENSE,
    collect_rows,
    publish,
    render_card,
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
        assert kinds == ["create_repo", "upload_folder", "upload_file"]
        create = api.calls[0][2]
        assert create == {"repo_type": "dataset", "private": False, "exist_ok": True}
        folder = api.calls[1][2]
        assert folder["folder_path"] == str(tmp_path)
        assert folder["path_in_repo"] == "runs"
        assert folder["repo_type"] == "dataset"
        assert api.calls[2][2]["path_in_repo"] == "README.md"

    def test_refuses_an_empty_evidence_folder(self, tmp_path):
        """Test that nothing is published when there is nothing to show."""
        with pytest.raises(ValueError):
            publish(tmp_path, "me/results", api=FakeApi())
