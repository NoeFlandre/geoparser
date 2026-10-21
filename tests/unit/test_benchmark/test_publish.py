"""Publication gates, versioned artifacts, and single-commit Hub behavior."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

from scripts.benchmark import missing_matrix, pipelines, publish
from scripts.benchmark.corpora import EXACT_FULL_RECORD_DEDUP_VIEW, STRICT_RAW_VIEW

PARENT = "a" * 40
HEAD = "b" * 40


def _expected_totals(view):
    if view == STRICT_RAW_VIEW:
        return {"geovirus": (1, 1)}
    return {"newsli-ar": (1, 1)}


def _write_evidence(root: Path, view: str, corpus_name: str) -> Path:
    """Create a complete tiny current cell with all role pins and inspections."""
    run = root / f"run-{view}"
    cell = run / corpus_name
    cell.mkdir(parents=True)
    spec = publish.CORPORA[corpus_name]
    source_sha = "c" * 64
    archive_sha = source_sha if spec.kind == "newsli" else None
    coordinate_sha = None
    corpus_digest = "corpus-digest"
    pipelines_payload = []
    model_runs = {}
    cell_pipelines = {}
    for name in publish.MAIN_PIPELINES:
        pins = pipelines.pipeline_model_pins(name)
        pipelines_payload.append(
            {
                "name": name,
                "models": pins,
                "recognition": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                "resolution": {
                    "accuracy_at_161km": 0.5,
                    "auc": 0.5,
                    "median_error_km": 100.0,
                    "mean_error_km": 100.0,
                },
                "elapsed_seconds": 1.0,
            }
        )
        model_runs[name] = {
            "model_pins": pins,
            "execution_settings": {
                "document_chunk_size": 16,
                "recognition": {"method": "fixture"},
                "resolution": {"batch_size": 16},
            },
        }
        cell_pipelines[name] = {
            "status": "complete",
            "phase_status": {"recognition": "complete", "resolution": "complete"},
            "model_pins": pins,
        }
        record = {
            "schema": "geoparser-prediction-inspection-v1",
            "pipeline": name,
            "scoring_view": view,
            "document_id": "doc-1",
            "reference": [
                {"start": 0, "end": 4, "name": "city", "latitude": 1, "longitude": 2}
            ],
            "resolution_input": "gold_reference",
            "prediction": {"recognition": [], "resolution": []},
            "phase_status": {"recognition": "complete", "resolution": "complete"},
            "failures": {},
        }
        (cell / f"inspection-{name}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
    removed = (
        missing_matrix.NEWSLI_EXACT_DUPLICATES_REMOVED["ar"]
        if view == EXACT_FULL_RECORD_DEDUP_VIEW
        else 0
    )
    report_data = {
        "scoring_view": view,
        "corpus": corpus_name,
        "language": spec.language,
        "limit": None,
        "partial": False,
        "documents": 1,
        "gold_toponyms": 1,
        "corpus_digest": corpus_digest,
        "source_url": spec.url,
        "source_content_sha256": source_sha,
        "source_archive_sha256": archive_sha,
        "coordinate_cache_sha256": coordinate_sha,
        "duplicate_records_removed": removed,
        "gazetteer": pipelines.GAZETTEER_NAME,
        "gazetteer_sha256": "f" * 64,
        "min_similarity": 0.0,
        "commit": "d" * 40,
        "environment": {"started_at": "2026-10-21T00:00:00Z"},
        "pipelines": pipelines_payload,
    }
    (cell / publish.REPORT_NAME).write_text(json.dumps(report_data), encoding="utf-8")
    (cell / "benchmark-report.md").write_text("report\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "full_commit": "d" * 40,
        "dependency_lock_sha256": "e" * 64,
        "protocol": {
            "scoring_view": view,
            "phases": ["recognition", "resolution"],
            "gazetteer": pipelines.GAZETTEER_NAME,
            "gazetteer_artifact_sha256": "f" * 64,
            "min_similarity": 0.0,
            "chunk_size": 16,
            "device": "cuda",
            "accelerator": "T4",
            "pipelines": model_runs,
            "corpora": {
                corpus_name: {
                    "status": "complete",
                    "partial": False,
                    "documents": 1,
                    "gold_toponyms": 1,
                    "source_url": spec.url,
                    "source_content_sha256": source_sha,
                    "source_archive_sha256": archive_sha,
                    "coordinate_cache_sha256": coordinate_sha,
                    "corpus_digest": corpus_digest,
                    "pipelines": cell_pipelines,
                }
            },
        },
    }
    (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cell / publish.REPORT_NAME


class FakeApi:
    """In-memory HF repository that records one atomic commit and its readback."""

    def __init__(self, tmp_path):
        self.files = {
            "README.md": b"old card",
            "results.csv": b"old results",
            "current/v2/strict-raw-v1/results.csv": b"previous current results",
            "current/v2/obsolete.txt": b"stale current artifact",
            "runs/old/report.json": b"historical report",
            ".gitattributes": b"*.json filter=lfs diff=lfs merge=lfs -text\n",
            "archive/README.md": b"prior archive info",
        }
        self.tmp_path = tmp_path
        self.commit_calls = []

    def repo_info(self, repo_id, *, repo_type, revision=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=HEAD if revision == HEAD else PARENT)

    def list_repo_files(self, repo_id, *, repo_type, revision):
        assert repo_type == "dataset"
        assert revision in {PARENT, HEAD}
        return sorted(self.files)

    def create_commit(self, **kwargs):
        self.commit_calls.append(kwargs)
        assert kwargs["parent_commit"] == PARENT
        assert kwargs["repo_type"] == "dataset"
        original = self.files.copy()
        for operation in kwargs["operations"]:
            if isinstance(operation, CommitOperationCopy):
                self.files[operation.path_in_repo] = original[
                    operation.src_path_in_repo
                ]
            elif isinstance(operation, CommitOperationAdd):
                self.files[operation.path_in_repo] = operation.path_or_fileobj
            elif isinstance(operation, CommitOperationDelete):
                self.files.pop(operation.path_in_repo, None)
        return SimpleNamespace(oid=HEAD)

    def hf_hub_download(self, repo_id, *, filename, repo_type, revision):
        assert (repo_type, revision) == ("dataset", HEAD)
        path = self.tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return str(path)


@pytest.fixture
def tiny_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    _write_evidence(tmp_path, EXACT_FULL_RECORD_DEDUP_VIEW, "newsli-ar")
    return tmp_path


def test_registry_has_only_the_six_exact_current_combinations():
    assert tuple(publish.PIPELINE_DESCRIPTIONS) == missing_matrix.CURRENT_PIPELINES
    assert len(publish.PIPELINE_DESCRIPTIONS) == 6
    assert set(missing_matrix.LEGACY_PIPELINE_ID_MAP) >= {
        "swapped",
        "hybrid",
        "prior",
        "otter-cross-minilm",
    }


def test_card_is_minimal_and_keeps_viewer_configs_separate():
    card = publish.render_card([])
    assert "config_name: strict_raw" in card
    assert "config_name: newsli_exact_record_dedup" in card
    assert "current/v2/strict-raw-v1/results.csv" in card
    assert "current/v2/newsli-exact-full-record-dedup-v1/results.csv" in card
    assert "gold toponym spans" in card
    assert "end-to-end score is reported" in card
    assert "population prior" not in card
    assert "inflection" not in card


def test_report_validation_requires_every_expected_role_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    report_path = _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    data = json.loads(report_path.read_text())
    manifest_path = report_path.parent.parent / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["protocol"]["pipelines"]["gliner-jina"]["model_pins"]["reranker"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing or mispins a model role"):
        publish._validate_current_report(report_path, data)


def test_report_validation_rejects_inspection_text(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    report_path = _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    data = json.loads(report_path.read_text())
    inspection = report_path.parent / "inspection-gliner-minilm.jsonl"
    record = json.loads(inspection.read_text())
    record["text"] = "article body"
    inspection.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="includes private text"):
        publish._validate_current_report(report_path, data)


@pytest.mark.parametrize("location", ["record", "prediction"])
def test_report_validation_rejects_unknown_inspection_fields(
    tmp_path, monkeypatch, location
):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    report_path = _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    data = json.loads(report_path.read_text())
    inspection = report_path.parent / "inspection-gliner-minilm.jsonl"
    record = json.loads(inspection.read_text())
    if location == "record":
        record["document_text"] = "private source text"
    else:
        record["prediction"]["recognition"].append(
            {
                "start": 0,
                "end": 4,
                "identifier": None,
                "document_id": "doc-1",
                "latitude": None,
                "longitude": None,
                "document_text": "private source text",
            }
        )
    inspection.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="unsupported inspection fields"):
        publish._validate_current_report(report_path, data)


def test_bundle_writes_distinct_metrics_and_protocol_paths(tiny_evidence):
    files, rows = publish._bundle_files(tiny_evidence, "me/results")

    strict = "current/v2/strict-raw-v1/results.csv"
    dedup = "current/v2/newsli-exact-full-record-dedup-v1/results.csv"
    assert strict in files and dedup in files
    assert len(rows) == 12
    assert b"strict-raw-v1" in files[strict]
    assert b"newsli-exact-full-record-dedup-v1" in files[dedup]
    assert not any("checkpoint-" in path for path in files)
    assert "README.md" in files
    assert "current/v2/corpus-sources.md" in files


def test_publisher_uses_one_parent_guarded_commit_archives_and_reads_back(
    tiny_evidence, tmp_path
):
    api = FakeApi(tmp_path)

    result = publish.publish(tiny_evidence, "me/results", api=api)

    assert result == {"parent_sha": PARENT, "commit_sha": HEAD}
    assert len(api.commit_calls) == 1
    call = api.commit_calls[0]
    assert call["parent_commit"] == PARENT
    operations = call["operations"]
    assert any(
        isinstance(operation, CommitOperationCopy)
        and operation.src_path_in_repo == "runs/old/report.json"
        and operation.path_in_repo.endswith("/legacy/runs/old/report.json")
        for operation in operations
    )
    assert any(
        isinstance(operation, CommitOperationDelete)
        and operation.path_in_repo == "runs/old/report.json"
        for operation in operations
    )
    assert any(isinstance(operation, CommitOperationAdd) for operation in operations)
    assert "runs/old/report.json" not in api.files
    assert any(path.startswith("archive/snapshots/") for path in api.files)
    assert "current/v2/strict-raw-v1/results.csv" in api.files
    assert "current/v2/newsli-exact-full-record-dedup-v1/results.csv" in api.files
    assert (
        api.files["current/v2/strict-raw-v1/results.csv"] != b"previous current results"
    )
    assert "current/v2/obsolete.txt" not in api.files
    assert (
        api.files[f"archive/snapshots/{PARENT}/current/v2/strict-raw-v1/results.csv"]
        == b"previous current results"
    )
    assert (
        api.files[f"archive/snapshots/{PARENT}/current/v2/obsolete.txt"]
        == b"stale current artifact"
    )
    assert "README.md" in api.files


def test_incomplete_bundle_fails_before_repo_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    api = FakeApi(tmp_path)
    before = api.files.copy()

    with pytest.raises(ValueError, match="cannot publish"):
        publish.publish(tmp_path, "me/results", api=api)

    assert api.files == before
    assert api.commit_calls == []
