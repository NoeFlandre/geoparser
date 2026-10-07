"""The dry-run boundary must never invoke a model, network, or output writer."""

import json
import subprocess
import sys

from scripts.benchmark_protocol import __main__ as cli
from tests.unit.test_benchmark_protocol.test_protocol import experiment_payload


def test_dry_run_validates_and_prints_full_inventory(tmp_path, capsys):
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps(experiment_payload()))
    before = sorted(tmp_path.iterdir())
    assert cli.main(["--dry-run", str(manifest)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["configuration_count"] == 2
    assert result["status_counts"] == {"planned": 2}
    assert result["included_languages"] == ["en", "fr"]
    assert len(result["missing_target_languages"]) == 83
    assert sorted(tmp_path.iterdir()) == before


def test_dry_run_error_is_actionable(tmp_path, capsys):
    manifest = tmp_path / "bad.json"
    manifest.write_text('{"protocol":null}')
    assert cli.main(["--dry-run", str(manifest)]) == 2
    assert "protocol" in capsys.readouterr().err


def test_dry_run_missing_file_is_actionable(tmp_path, capsys):
    assert cli.main(["--dry-run", str(tmp_path / "absent")]) == 2
    assert "absent" in capsys.readouterr().err


def test_schema_is_machine_readable(capsys):
    assert cli.main(["--schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Protocol"]["properties"]["schema_version"]["const"] == "1.0"


def test_fresh_process_dry_run_never_imports_model_or_network_stack(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(experiment_payload()))
    script = """
import importlib.abc
import runpy
import sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','spacy','transformers','datasets','requests','httpx','huggingface_hub'}:
            raise AssertionError('forbidden dry-run import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
sys.argv = ['benchmark_protocol', '--dry-run', sys.argv[1]]
runpy.run_module('scripts.benchmark_protocol', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["configuration_count"] == 2
