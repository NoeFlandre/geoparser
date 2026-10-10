"""
Tests for the pinned inventory of the UniTopRank release and TopoResolve.

The expected totals are written out here, not computed from the snapshot, so
a mistyped count in the snapshot fails a test instead of passing silently. The
totals come from the member listing of the release archive on 2026-10-09.
"""

import json
import re
from pathlib import Path

import pytest

from scripts.benchmark import corpora

SNAPSHOT = Path(corpora.__file__).with_name("data") / "geographic-corpora-snapshot.json"
NEWSLI_EVIDENCE = (
    Path(__file__).resolve().parents[3] / "benchmark-evidence" / "2026-09-23-newsli"
)
GUIDE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "guides"
    / ("geographic-corpora-inventory.md")
)
DATASET_FOLDERS = {
    "19th",
    "ITA-DSTR",
    "LDC",
    "TUD",
    "ar_geotoponyms",
    "de_geotoponyms",
    "es_geotoponyms",
    "fa_geotoponyms",
    "fingernews_gold",
    "fingertweets_gold",
    "geocorpora",
    "geovirus",
    "gwn",
    "ja_geotoponyms",
    "lgl",
    "pl_geotoponyms",
    "ro_geotoponyms",
    "semeval",
    "sr_geotoponyms",
    "ta_geotoponyms",
    "tr_geotoponyms",
    "trnews",
    "uk_geotoponyms",
    "wiktor",
    "wotr",
}
NEWSLI_LANGUAGES = {"ar", "de", "es", "fa", "ja", "pl", "ro", "sr", "ta", "tr", "uk"}
STATUSES = {"registered", "duplicate", "possible_duplicate", "candidate"}
HEX_MD5 = re.compile(r"^[0-9a-f]{32}$")
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def snapshot() -> dict:
    """The pinned snapshot, parsed once for the whole module."""
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def folders(snapshot: dict) -> dict:
    """The per-folder rows of the UniTopRank release."""
    return snapshot["unitoprank"]["folders"]


class TestReleaseListing:
    """The snapshot agrees with the member listing of the archive."""

    def test_the_release_holds_twenty_five_dataset_folders(self, folders):
        """The issue's 25 datasets are the top-level folders of the archive."""
        assert set(folders) == DATASET_FOLDERS
        assert len(folders) == 25

    def test_members_add_up_to_the_listed_total(self, snapshot, folders):
        """Folder entries plus one gold JSON per folder make 59,519 members."""
        central = snapshot["unitoprank"]["central_directory"]
        folder_entries = sum(row["entries"] for row in folders.values())

        assert central["members"] == 59519
        assert folder_entries + central["gold_json_files"] == 59519

    def test_the_central_directory_counts_twenty_five_folders_and_gold_files(
        self, snapshot
    ):
        """The archive holds one gold JSON file beside each of its 25 folders."""
        central = snapshot["unitoprank"]["central_directory"]

        assert central["dataset_folders"] == 25
        assert central["gold_json_files"] == 25

    def test_text_files_never_exceed_entries(self, folders):
        """A folder cannot hold more text files than entries."""
        for name, row in folders.items():
            assert 0 <= row["text_files"] <= row["entries"], name

    def test_the_geovirus_folder_holds_the_229_articles_of_geovirus(self, folders):
        """229 is the article count that corpus.py documents for GeoVirus."""
        assert folders["geovirus"]["text_files"] == 229

    def test_the_data_file_is_pinned_by_size_and_checksum(self, snapshot):
        """The archive is identified by its figshare file id, size and MD5."""
        data_file = snapshot["unitoprank"]["data_file"]

        assert data_file["figshare_file_id"] == 59465342
        assert data_file["bytes"] == 63356728
        assert HEX_MD5.match(data_file["md5"])


class TestStatuses:
    """Every folder is classified, and registered folders point at registry keys."""

    def test_every_folder_has_a_known_status(self, folders):
        """No folder is left unclassified or given a free-form status."""
        for name, row in folders.items():
            assert row["status"] in STATUSES, name

    def test_every_folder_without_a_registration_gives_a_reason(self, folders):
        """A candidate, duplicate or excluded folder must say why."""
        for name, row in folders.items():
            if row["status"] != "registered":
                assert row["reason"].strip(), name

    def test_registered_folders_name_registered_corpora(self, folders):
        """A registered folder points at a corpus the harness really offers."""
        for name, row in folders.items():
            if row["status"] == "registered":
                assert row["registered_as"] in corpora.CORPORA, name

    def test_candidates_are_not_registered_under_any_name(self, folders):
        """A candidate has no registry key yet, so it cannot be run by accident."""
        for name, row in folders.items():
            if row["status"] == "candidate":
                assert row["registered_as"] is None, name
                assert row["language"], name

    def test_geovirus_is_a_possible_duplicate_until_its_content_is_compared(
        self, folders
    ):
        """GeoVirus is registered, but the name and count alone do not prove a match."""
        assert folders["geovirus"]["status"] == "possible_duplicate"
        assert folders["geovirus"]["registered_as"] == "geovirus"
        assert "geovirus" in corpora.CORPORA

    def test_nineteenth_century_folder_is_not_yet_resolved_as_a_duplicate(
        self, folders
    ):
        """TopRes19th may match the registered HIPE split; that is not decided yet."""
        assert folders["19th"]["status"] == "possible_duplicate"
        assert folders["19th"]["registered_as"] == "topres19th-en"
        assert "topres19th-en" in corpora.CORPORA


class TestNewsliAgreement:
    """The NewsLi folders are exactly the languages the adapter registers."""

    def test_geotoponym_folders_are_the_eleven_newsli_languages(self, folders):
        """Each *_geotoponyms folder is one NewsLi language, and no more."""
        languages = {
            name.removesuffix("_geotoponyms")
            for name in folders
            if name.endswith("_geotoponyms")
        }

        assert languages == NEWSLI_LANGUAGES

    def test_the_registry_offers_the_same_eleven_languages(self, folders):
        """The registry and the release agree on the NewsLi languages."""
        registered = {
            spec.language
            for name, spec in corpora.CORPORA.items()
            if name.startswith("newsli-")
        }

        assert registered == NEWSLI_LANGUAGES

    def test_each_newsli_folder_is_registered_under_its_own_language(self, folders):
        """ar_geotoponyms is newsli-ar, and so on for every language."""
        for language in NEWSLI_LANGUAGES:
            row = folders[f"{language}_geotoponyms"]
            assert row["status"] == "registered"
            assert row["registered_as"] == f"newsli-{language}"

    def test_the_newsli_totals_are_the_checked_in_report_counts(self):
        """Each language's documents, and gold when uncapped, are the report's own counts."""
        for language in NEWSLI_LANGUAGES:
            report = json.loads(
                (
                    NEWSLI_EVIDENCE / f"newsli-{language}" / "benchmark-report.json"
                ).read_text(encoding="utf-8")
            )
            documents, gold = corpora.NEWSLI_RECORDED_TOTALS[language]
            assert documents == report["documents"], language
            uncapped = report["documents"] < corpora.MAX_DOCUMENTS
            assert gold == (report["gold_toponyms"] if uncapped else None), language

    def test_the_raw_text_file_count_is_not_the_newsli_expected_total(self, folders):
        """Text files outnumber the articles the adapter keeps, so they are not the total."""
        for language in ("fa", "pl", "ro", "uk"):
            text_files = folders[f"{language}_geotoponyms"]["text_files"]
            documents, _gold = corpora.NEWSLI_RECORDED_TOTALS[language]

            assert corpora.CORPORA[f"newsli-{language}"].documents == documents
            assert documents < text_files, language


class TestToporesolve:
    """TopoResolve is pinned, and it stays excluded while it has no licence."""

    def test_the_repository_commit_is_pinned(self, snapshot):
        """The commit is a full 40-character SHA."""
        commit = snapshot["toporesolve"]["commit"]

        assert HEX_SHA1.match(commit)

    def test_no_licence_is_recorded_because_none_was_found(self, snapshot):
        """Nothing in the repository states a licence, so none is claimed."""
        assert snapshot["toporesolve"]["licence_as_found"] is None

    def test_the_three_gold_files_are_pinned_by_blob(self, snapshot):
        """GPE, LOC and FAC each carry a blob SHA and a byte size."""
        gold = snapshot["toporesolve"]["gold_files"]

        assert [row["path"].rsplit("/", 1)[-1].split("_", 1)[0] for row in gold] == [
            "GPE",
            "LOC",
            "FAC",
        ]
        for row in gold:
            assert HEX_SHA1.match(row["blob_sha"]), row["path"]
            assert row["bytes"] > 0

    def test_the_evaluation_radius_is_recorded(self, snapshot):
        """The README radius is kept beside the gold, not hidden in code."""
        assert snapshot["toporesolve"]["evaluation_radius_miles"] == 25


class TestGuide:
    """The guide names every folder and gold file the snapshot pins."""

    def test_the_guide_names_every_folder(self, folders):
        """A folder missing from the guide would be silently unreviewed."""
        text = GUIDE.read_text(encoding="utf-8")

        for name in folders:
            assert f"`{name}`" in text, name

    def test_the_guide_names_every_toporesolve_gold_file(self, snapshot):
        """Each gold file is named in the guide by its stem."""
        text = GUIDE.read_text(encoding="utf-8")

        for row in snapshot["toporesolve"]["gold_files"]:
            stem = row["path"].rsplit("/", 1)[-1].removesuffix(".jsonl")
            assert stem in text, stem
