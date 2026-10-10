"""
The recorded benchmark totals agree with the checked-in reports they came from.

These tests read the reports under benchmark-evidence/. The mutation sandbox
does not copy that folder, so they live here, where CI runs them, rather than
beside the unit tests of the same totals.
"""

import json
from pathlib import Path

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


def test_every_hipe_split_carries_its_recorded_report_counts():
    """Each registered HIPE split has the documents and gold its report recorded."""
    hipe = {name for name, spec in corpora.CORPORA.items() if spec.kind == corpora.HIPE}
    assert set(corpora.HIPE_RECORDED_TOTALS) == hipe
    for name, (documents, gold) in corpora.HIPE_RECORDED_TOTALS.items():
        report = _report(MULTILINGUAL_EVIDENCE, name)
        assert (documents, gold) == (report["documents"], report["gold_toponyms"]), name
