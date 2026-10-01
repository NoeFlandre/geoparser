import json

import pytest

from scripts.panx_benchmark.checkpoint import (
    ModelLanguageCheckpoints,
    require_clean_commit,
)


def test_clean_commit_is_required_for_an_immutable_run_snapshot():
    assert require_clean_commit("abc123") == "abc123"

    for commit in ("unknown", "abc123-dirty"):
        with pytest.raises(ValueError, match="clean, identifiable"):
            require_clean_commit(commit)


def test_language_checkpoints_are_atomic_and_resumable(tmp_path):
    identity = {"repository_commit": "abc123", "dataset_revision": "f0a3"}
    store = ModelLanguageCheckpoints(tmp_path, identity)
    result = {
        "status": "evaluated",
        "evaluated_examples": 2,
        "metrics": {"f1": 0.5},
    }

    assert store.load_language("gliner", "org/model", "model-sha", "en", 10) is None
    store.save_language("gliner", "org/model", "model-sha", "en", 10, result)

    resumed = ModelLanguageCheckpoints(tmp_path, identity)
    path = resumed.directory / "languages" / "gliner" / "en.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert resumed.snapshot_id == store.snapshot_id
    assert (
        record["snapshot_id"],
        resumed.load_language("gliner", "org/model", "model-sha", "en", 10),
    ) == (store.snapshot_id, result)


def test_checkpoint_rejects_a_language_record_from_different_inputs(tmp_path):
    store = ModelLanguageCheckpoints(tmp_path, {"repository_commit": "abc123"})
    store.save_language(
        "gliner", "org/model", "model-sha", "en", 10, {"status": "evaluated"}
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        store.load_language("gliner", "org/model", "model-sha", "en", 11)


def test_checkpoint_rejects_a_manifest_that_does_not_match_its_snapshot(tmp_path):
    store = ModelLanguageCheckpoints(tmp_path, {"repository_commit": "abc123"})
    manifest = store.directory / "snapshot.json"
    manifest.write_text('{"repository_commit":"edited"}', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest does not match"):
        ModelLanguageCheckpoints(tmp_path, {"repository_commit": "abc123"})


def test_checkpoint_rejects_orphaned_language_files(tmp_path):
    store = ModelLanguageCheckpoints(tmp_path, {"repository_commit": "abc123"})
    (store.directory / "snapshot.json").unlink()
    (store.directory / "orphan.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no run snapshot manifest"):
        ModelLanguageCheckpoints(tmp_path, {"repository_commit": "abc123"})
