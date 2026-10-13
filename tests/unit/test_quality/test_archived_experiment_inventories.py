"""The experiment cleanup retains source inventories and completed evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[3]
INDEX = ROOT / "benchmark-evidence/archives/inventory-index.json"
EXPECTED_ARCHIVES = {
    "scripts/multiconer_benchmark/multiconer_manifest.json": (
        "benchmark-evidence/archives/multiconer/manifest.json"
    ),
    "scripts/masakhaner_benchmark/manifest.json": (
        "benchmark-evidence/archives/masakhaner/manifest.json"
    ),
    "scripts/uner_benchmark/sources.json": (
        "benchmark-evidence/archives/uner/sources.json"
    ),
    "scripts/unitoprank_benchmark/pins.py": (
        "benchmark-evidence/archives/unitoprank/pins.py"
    ),
    "scripts/unitoprank_benchmark/ranker-requirements.txt": (
        "benchmark-evidence/archives/unitoprank/ranker-requirements.txt"
    ),
    "scripts/reranker_isolation/pins.py": (
        "benchmark-evidence/archives/reranker-isolation/pins.py"
    ),
}


def test_archived_inventories_match_their_recorded_hashes() -> None:
    """Every moved inventory retains its exact original bytes."""
    index = json.loads(INDEX.read_text(encoding="utf-8"))

    assert index["schema_version"] == 1
    assert {
        entry["original_path"]: entry["archive_path"] for entry in index["files"]
    } == EXPECTED_ARCHIVES
    assert not (ROOT / "scripts/reranker_isolation/pins.py").exists()
    for entry in index["files"]:
        archived = ROOT / entry["archive_path"]
        payload = archived.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_completed_and_inventory_evidence_remains_present() -> None:
    """The cleanup keeps published and locally recorded evidence in place."""
    retained = (
        "benchmark-evidence/2026-09-23-hybrid/baseline-jina/benchmark-report.md",
        "benchmark-evidence/masakhaner/inventory-2026-10-09/report.json",
        "benchmark-evidence/masakhaner/inventory-2026-10-09/report.md",
        "benchmark-evidence/uner/adapter-source-audit.json",
    )

    assert all((ROOT / path).is_file() for path in retained)


def test_uner_source_audit_path_maps_to_archived_inventory() -> None:
    """The unchanged historical audit path resolves through the archive index."""
    audit = json.loads(
        (ROOT / "benchmark-evidence/uner/adapter-source-audit.json").read_text(
            encoding="utf-8"
        )
    )
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in index["files"]
        if entry["original_path"] == audit["source_snapshot"]
    ]

    assert len(matches) == 1
    assert (ROOT / matches[0]["archive_path"]).is_file()
