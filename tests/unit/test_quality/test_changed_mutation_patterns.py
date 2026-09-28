import sys
from types import SimpleNamespace

import scripts.changed_mutation_patterns as changed_patterns


def test_mutation_patterns_select_only_mutable_changed_geoparser_modules() -> None:
    assert changed_patterns.module_patterns_for_paths(
        [
            "geoparser/modules/recognizers/manual.py",
            "geoparser/db/__init__.py",
            "tests/unit/test_manual.py",
            "scripts/quality_gauntlet.py",
            "geoparser/py.typed",
        ]
    ) == ["geoparser.modules.recognizers.manual.*"]


def test_mutation_patterns_skip_modules_excluded_by_mutmut_config() -> None:
    assert changed_patterns.module_patterns_for_paths(
        ["geoparser/cli/parse.py", "geoparser/db/db.py"]
    ) == ["geoparser.db.db.*"]


def test_changed_paths_requests_only_added_or_modified_package_files(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append((command, check, capture_output, text))
        return SimpleNamespace(stdout="geoparser/modules/manual.py\n")

    monkeypatch.setattr(
        changed_patterns,
        "subprocess",
        SimpleNamespace(run=fake_run),
        raising=False,
    )

    assert changed_patterns.changed_paths("base", "head") == [
        "geoparser/modules/manual.py"
    ]
    assert calls == [
        (
            (
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRT",
                "base...head",
                "--",
                "geoparser",
            ),
            True,
            True,
            True,
        )
    ]


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
