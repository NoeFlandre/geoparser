"""The experiment cleanup retains source inventories and completed evidence."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
INDEX = ROOT / "benchmark-evidence/archives/inventory-index.json"
pytestmark = pytest.mark.skipif(
    ROOT.name == "mutants",
    reason="The mutation workspace copies tests but not archived evidence files",
)


def test_archived_inventories_match_their_recorded_hashes() -> None:
    """Every moved inventory retains its exact original bytes."""
    index = json.loads(INDEX.read_text(encoding="utf-8"))

    assert index["schema_version"] == 1
    assert len(index["files"]) == 6
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
