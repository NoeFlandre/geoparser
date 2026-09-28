import sys
from types import SimpleNamespace

import scripts.changed_mutation_patterns as changed_patterns


def _stub_mutmut(monkeypatch) -> None:
    """Keep selector tests independent of mutmut's generated source tree."""

    def mutate_file_contents(filename: str, source_text: str) -> SimpleNamespace:
        mutant_names = () if filename.endswith("/__init__.py") else ("mutant",)
        return SimpleNamespace(mutant_names=mutant_names)

    monkeypatch.setattr(changed_patterns, "mutate_file_contents", mutate_file_contents)


def test_mutation_patterns_select_only_mutable_changed_geoparser_modules(
    monkeypatch,
) -> None:
    _stub_mutmut(monkeypatch)
    assert changed_patterns.module_patterns_for_paths(
        [
            "geoparser/modules/recognizers/manual.py",
            "geoparser/db/__init__.py",
            "tests/unit/test_manual.py",
            "scripts/quality_gauntlet.py",
            "geoparser/py.typed",
        ]
    ) == ["geoparser.modules.recognizers.manual.*"]


def test_mutation_patterns_skip_modules_excluded_by_mutmut_config(monkeypatch) -> None:
    _stub_mutmut(monkeypatch)
    assert changed_patterns.module_patterns_for_paths(
        ["geoparser/cli/parse.py", "geoparser/db/db.py"]
    ) == ["geoparser.db.db.*"]


def test_mutation_patterns_skip_deleted_package_modules(monkeypatch) -> None:
    _stub_mutmut(monkeypatch)

    assert (
        changed_patterns.module_patterns_for_paths(["geoparser/modules/removed.py"])
        == []
    )


def test_changed_paths_lists_all_added_or_modified_files(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append((command, check, capture_output, text))
        return SimpleNamespace(
            stdout="geoparser/modules/manual.py\ntests/unit/test_manual.py\n"
        )

    monkeypatch.setattr(
        changed_patterns,
        "subprocess",
        SimpleNamespace(run=fake_run),
        raising=False,
    )

    assert changed_patterns.changed_paths("base", "head") == [
        "geoparser/modules/manual.py",
        "tests/unit/test_manual.py",
    ]
    assert calls == [
        (
            (
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRTD",
                "base...head",
            ),
            True,
            True,
            True,
        )
    ]


def test_changed_paths_includes_deleted_files(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        return SimpleNamespace(stdout="tests/unit/test_removed_assertion.py\n")

    monkeypatch.setattr(
        changed_patterns,
        "subprocess",
        SimpleNamespace(run=fake_run),
        raising=False,
    )

    assert changed_patterns.changed_paths("base", "head") == [
        "tests/unit/test_removed_assertion.py"
    ]
    assert "--diff-filter=ACMRTD" in calls[0]


def test_main_prints_the_patterns_for_the_requested_commit_range(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        changed_patterns,
        "changed_paths",
        lambda base, head: ["geoparser/db/db.py"],
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["changed_mutation_patterns.py", "base", "head"])

    changed_patterns.main()

    assert capsys.readouterr().out == "geoparser.db.db.*\n"


def test_main_requests_full_mutation_for_test_only_changes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        changed_patterns,
        "changed_paths",
        lambda base, head: ["tests/unit/test_db/test_crud/test_document.py"],
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["changed_mutation_patterns.py", "base", "head"])

    changed_patterns.main()

    assert capsys.readouterr().out == "FULL_MUTATION\n"


def test_main_requests_full_mutation_for_mutation_configuration_changes(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        changed_patterns,
        "changed_paths",
        lambda base, head: ["pyproject.toml"],
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["changed_mutation_patterns.py", "base", "head"])

    changed_patterns.main()

    assert capsys.readouterr().out == "FULL_MUTATION\n"
