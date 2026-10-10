"""
The recorded benchmark totals agree with the checked-in reports they came from.

These tests read the reports under benchmark-evidence/. The mutation sandbox
does not copy that folder, so they live here, where CI runs them, rather than
beside the unit tests of the same totals.
"""

import json
from pathlib import Path

import pytest

from scripts.benchmark import corpora

EVIDENCE = Path(__file__).resolve().parents[2] / "benchmark-evidence"
NEWSLI_EVIDENCE = EVIDENCE / "2026-09-23-newsli"
MULTILINGUAL_EVIDENCE = EVIDENCE / "2026-09-23-multilingual"


def _report(directory: Path, name: str) -> dict:
    """Read the benchmark report one corpus wrote into a checked-in folder."""
    path = directory / name / "benchmark-report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_newsli_totals_are_the_checked_in_report_counts():
    """Each language's documents, and gold when uncapped, are the report's own counts."""
    for language in corpora.NEWSLI_LANGUAGES:
        report = _report(NEWSLI_EVIDENCE, f"newsli-{language}")
        documents, gold = corpora.NEWSLI_RECORDED_TOTALS[language]
        assert documents == report["documents"], language
        uncapped = report["documents"] < corpora.MAX_DOCUMENTS
        assert gold == (report["gold_toponyms"] if uncapped else None), language


def _registered_hipe_splits() -> set:
    """The names of every corpus the registry declares as a HIPE split."""
    return {name for name, spec in corpora.CORPORA.items() if spec.kind == corpora.HIPE}


def test_every_registered_hipe_split_has_recorded_totals():
    """No HIPE split is registered without the totals its report recorded."""
    assert set(corpora.HIPE_RECORDED_TOTALS) == _registered_hipe_splits()


@pytest.mark.parametrize("name", sorted(corpora.HIPE_RECORDED_TOTALS))
def test_a_hipe_split_carries_its_recorded_report_counts(name):
    """The split's documents and gold are the counts its report recorded."""
    documents, gold = corpora.HIPE_RECORDED_TOTALS[name]
    report = _report(MULTILINGUAL_EVIDENCE, name)
    assert (documents, gold) == (report["documents"], report["gold_toponyms"])
