"""Publication gates, versioned artifacts, and single-commit Hub behavior."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

from scripts.benchmark import checkpoint, missing_matrix, pipelines, publish, runner
from scripts.benchmark.corpora import EXACT_FULL_RECORD_DEDUP_VIEW, STRICT_RAW_VIEW
from scripts.benchmark.corpus import Document, GoldSpan

PARENT = "a" * 40
HEAD = "b" * 40
NEXT_HEAD = "c" * 40


def _expected_totals(view):
    if view == STRICT_RAW_VIEW:
        return {"geovirus": (1, 1)}
    return {"newsli-ar": (1, 1)}


def _has_legacy_report_copy(operations):
    return any(
        isinstance(operation, CommitOperationCopy)
        and operation.src_path_in_repo == "runs/old/report.json"
        and operation.path_in_repo.endswith("/legacy/runs/old/report.json")
        for operation in operations
    )


def _has_legacy_report_delete(operations):
    return any(
        isinstance(operation, CommitOperationDelete)
        and operation.path_in_repo == "runs/old/report.json"
        for operation in operations
    )


def _archive_snapshot_paths(files):
    return {path for path in files if path.startswith("archive/snapshots/")}


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
        "scoring_spans": 1,
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
                    "scoring_spans": 1,
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
        self.head_sha = PARENT
        self.revisions = {PARENT: self.files.copy()}
        self.fail_next_readback = False

    def repo_info(self, repo_id, *, repo_type, revision=None, files_metadata=False):
        assert repo_type == "dataset"
        sha = revision or self.head_sha
        contents = self.revisions[sha]
        siblings = [
            SimpleNamespace(
                rfilename=path,
                blob_id=hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest(),
                lfs=None,
            )
            for path, content in contents.items()
        ]
        return SimpleNamespace(sha=sha, siblings=siblings)

    def list_repo_files(self, repo_id, *, repo_type, revision):
        assert repo_type == "dataset"
        return sorted(self.revisions[revision])

    def create_commit(self, **kwargs):
        self.commit_calls.append(kwargs)
        assert (kwargs["parent_commit"], kwargs["repo_type"]) == (
            self.head_sha,
            "dataset",
        )
        original = self.revisions[kwargs["parent_commit"]]
        contents = original.copy()
        for operation in kwargs["operations"]:
            self._apply_operation(contents, original, operation)
        new_sha = HEAD if len(self.commit_calls) == 1 else NEXT_HEAD
        self.files = contents
        self.head_sha = new_sha
        self.revisions[new_sha] = contents.copy()
        return SimpleNamespace(oid=new_sha)

    @staticmethod
    def _apply_operation(contents, original, operation):
        if isinstance(operation, CommitOperationCopy):
            contents[operation.path_in_repo] = original[operation.src_path_in_repo]
        elif isinstance(operation, CommitOperationAdd):
            contents[operation.path_in_repo] = operation.path_or_fileobj
        elif isinstance(operation, CommitOperationDelete):
            contents.pop(operation.path_in_repo, None)

    def hf_hub_download(self, repo_id, *, filename, repo_type, revision):
        assert repo_type == "dataset"
        if self.fail_next_readback:
            self.fail_next_readback = False
            raise OSError("temporary readback outage")
        path = self.tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.revisions[revision][filename])
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
    fragments = (
        "config_name: strict_raw",
        "config_name: newsli_exact_record_dedup",
        "current/v2/strict-raw-v1/results.csv",
        "current/v2/newsli-exact-full-record-dedup-v1/results.csv",
        "gold toponym spans",
        "distinct (document ID, start, end) spans",
        "metric denominator",
        "end-to-end score is reported",
    )
    assert tuple(fragment in card for fragment in fragments) == (True,) * len(fragments)
    assert tuple(term not in card for term in ("population prior", "inflection")) == (
        True,
        True,
    )
    protocol = json.loads(publish.render_protocol_manifest(STRICT_RAW_VIEW))
    assert (
        protocol["scoring"]["unit"],
        "count once" in protocol["scoring"]["duplicate_identity_policy"],
    ) == (
        "distinct (document_id, start, end) span identity",
        True,
    )


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


def test_report_validation_checks_scoring_span_count_against_inspections(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    report_path = _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    data = json.loads(report_path.read_text())
    data["scoring_spans"] = 2
    report_path.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = report_path.parent.parent / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"]["corpora"]["geovirus"]["scoring_spans"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its inspections"):
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


@pytest.mark.parametrize(
    "location",
    ["record", "prediction", "prediction_value", "prediction_document_id"],
)
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
    elif location == "prediction":
        record["prediction"]["recognition"].append(
            {
                "start": 0,
                "end": 4,
                "identifier": None,
                "latitude": None,
                "longitude": None,
                "document_text": "private source text",
            }
        )
    elif location == "prediction_document_id":
        record["prediction"]["recognition"].append(
            {
                "start": 0,
                "end": 4,
                "identifier": None,
                "document_id": "doc-1",
                "latitude": None,
                "longitude": None,
            }
        )
    else:
        record["prediction"]["recognition"].append(
            {
                "start": 0,
                "end": 4,
                "identifier": {"document_text": "private source text"},
                "latitude": None,
                "longitude": None,
            }
        )
    inspection.write_text(json.dumps(record) + "\n")

    expected_error = (
        "non-scalar"
        if location == "prediction_value"
        else "unsupported inspection fields"
    )
    with pytest.raises(ValueError, match=expected_error):
        publish._validate_current_report(report_path, data)


def test_publisher_accepts_nonempty_generated_inspection_records(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    report_path = _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    data = json.loads(report_path.read_text())
    pipeline = "gliner-minilm"
    document = Document(
        "doc-1",
        "city",
        (GoldSpan(0, 4, "city", 1.0, 2.0),),
    )
    state = checkpoint.Checkpoint(
        checkpoint.RunIdentity(
            pipeline=pipeline,
            corpus_digest="corpus-pin",
            gazetteer="geonames",
            min_similarity=0.0,
            limit=None,
            commit="code-pin",
        )
    )
    state.record(
        runner.RECOGNITION,
        "doc-1",
        [
            {
                "start": 0,
                "end": 4,
                "identifier": None,
                "document_id": "doc-1",
                "latitude": None,
                "longitude": None,
            }
        ],
    )
    state.record(
        runner.RESOLUTION,
        "doc-1",
        [
            {
                "start": 0,
                "end": 4,
                "identifier": "geonames:1",
                "document_id": "doc-1",
                "latitude": 1.0,
                "longitude": 2.0,
            }
        ],
    )

    record = runner.inspection_records(
        [document],
        state,
        pipeline=pipeline,
        scoring_view=STRICT_RAW_VIEW,
    )[0]
    assert record["prediction"][runner.RECOGNITION] == [
        {
            "start": 0,
            "end": 4,
            "identifier": None,
            "latitude": None,
            "longitude": None,
        }
    ]
    assert record["prediction"][runner.RESOLUTION][0]["identifier"] == ("geonames:1")
    inspection = report_path.parent / f"inspection-{pipeline}.jsonl"
    inspection.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert publish._validate_current_report(report_path, data) == STRICT_RAW_VIEW


def test_bundle_writes_distinct_metrics_and_protocol_paths(tiny_evidence):
    files, rows = publish._bundle_files(tiny_evidence, "me/results")

    strict = "current/v2/strict-raw-v1/results.csv"
    dedup = "current/v2/newsli-exact-full-record-dedup-v1/results.csv"
    assert (
        strict in files and dedup in files,
        len(rows),
        b"strict-raw-v1" in files[strict],
        b"newsli-exact-full-record-dedup-v1" in files[dedup],
        any("checkpoint-" in path for path in files),
        "README.md" in files,
        "current/v2/corpus-sources.md" in files,
    ) == (True, 12, True, True, False, True, True)


def test_publisher_uses_one_parent_guarded_commit_archives_and_reads_back(
    tiny_evidence, tmp_path
):
    api = FakeApi(tmp_path)

    result = publish.publish(tiny_evidence, "me/results", api=api)

    assert result == {"parent_sha": PARENT, "commit_sha": HEAD, "status": "published"}
    call = api.commit_calls[0]
    operations = call["operations"]
    observed = (
        len(api.commit_calls),
        call["parent_commit"],
        _has_legacy_report_copy(operations),
        _has_legacy_report_delete(operations),
        any(isinstance(operation, CommitOperationAdd) for operation in operations),
        "runs/old/report.json" in api.files,
        bool(_archive_snapshot_paths(api.files)),
        "current/v2/strict-raw-v1/results.csv" in api.files,
        "current/v2/newsli-exact-full-record-dedup-v1/results.csv" in api.files,
        api.files["current/v2/strict-raw-v1/results.csv"]
        == b"previous current results",
        "current/v2/obsolete.txt" in api.files,
        api.files[f"archive/snapshots/{PARENT}/current/v2/strict-raw-v1/results.csv"],
        api.files[f"archive/snapshots/{PARENT}/current/v2/obsolete.txt"],
        "README.md" in api.files,
    )
    assert observed == (
        1,
        PARENT,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        b"previous current results",
        b"stale current artifact",
        True,
    )


def test_identical_publish_is_a_noop_and_does_not_add_an_archive(
    tiny_evidence, tmp_path
):
    api = FakeApi(tmp_path)

    publish.publish(tiny_evidence, "me/results", api=api)
    archived_paths = {path for path in api.files if path.startswith("archive/")}
    result = publish.publish(tiny_evidence, "me/results", api=api)

    assert (
        result,
        len(api.commit_calls),
        {path for path in api.files if path.startswith("archive/")},
    ) == (
        {"parent_sha": HEAD, "commit_sha": HEAD, "status": "unchanged"},
        1,
        archived_paths,
    )


def test_changed_bundle_creates_one_new_snapshot(tiny_evidence, tmp_path):
    api = FakeApi(tmp_path)
    publish.publish(tiny_evidence, "me/results", api=api)
    report_path = (
        tiny_evidence / f"run-{STRICT_RAW_VIEW}" / "geovirus" / publish.REPORT_NAME
    )
    data = json.loads(report_path.read_text())
    data["pipelines"][0]["recognition"]["f1"] = 0.6
    report_path.write_text(json.dumps(data), encoding="utf-8")

    result = publish.publish(tiny_evidence, "me/results", api=api)

    assert result == {
        "parent_sha": HEAD,
        "commit_sha": NEXT_HEAD,
        "status": "published",
    }
    assert len(api.commit_calls) == 2
    assert api.commit_calls[1]["parent_commit"] == HEAD
    assert f"archive/snapshots/{HEAD}/ARCHIVE.json" in api.files


def test_readback_failure_reconciles_successful_commit_and_retry_is_unchanged(
    tiny_evidence, tmp_path
):
    api = FakeApi(tmp_path)
    api.fail_next_readback = True

    result = publish.publish(tiny_evidence, "me/results", api=api)
    retry = publish.publish(tiny_evidence, "me/results", api=api)

    assert result == {
        "parent_sha": PARENT,
        "commit_sha": HEAD,
        "status": "reconciled",
    }
    assert retry == {
        "parent_sha": HEAD,
        "commit_sha": HEAD,
        "status": "unchanged",
    }
    assert len(api.commit_calls) == 1
    assert f"archive/snapshots/{PARENT}/ARCHIVE.json" in api.files


def test_incomplete_bundle_fails_before_repo_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_expected_corpus_totals", _expected_totals)
    _write_evidence(tmp_path, STRICT_RAW_VIEW, "geovirus")
    api = FakeApi(tmp_path)
    before = api.files.copy()

    with pytest.raises(ValueError, match="cannot publish"):
        publish.publish(tmp_path, "me/results", api=api)

    assert api.files == before
    assert api.commit_calls == []
