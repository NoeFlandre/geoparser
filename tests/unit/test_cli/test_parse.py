"""
Unit tests for geoparser/cli/parse.py

The models are mocked: these check the command's wiring, input handling and
output formats, not the quality of the recognizer or resolver.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from shapely.geometry import Point
from typer.testing import CliRunner

from geoparser.cli import parse as parse_module
from geoparser.cli.app import app

runner = CliRunner()


def _feature(identifier="2988507", crs="EPSG:4326", geometry=None):
    return SimpleNamespace(
        gazetteer_name="geonames",
        identifier=identifier,
        crs=crs,
        geometry=Point(2.35, 48.85) if geometry is None else geometry,
    )


def _reference(start, end, text, feature):
    return SimpleNamespace(start=start, end=end, text=text, location=feature)


def _document(text, references):
    return SimpleNamespace(text=text, toponyms=references)


@pytest.fixture(autouse=True)
def _restore_logging():
    """parse reconfigures logging globally; undo it for the next test."""
    root = logging.getLogger()
    package = logging.getLogger("geoparser")
    handlers, level = root.handlers[:], package.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            root.removeHandler(handler)
    root.handlers[:] = handlers
    package.setLevel(level)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Pretend 'geonames' is installed, in a throwaway data directory."""
    monkeypatch.setattr("geoparser.gazetteer.artifact.gazetteers_dir", lambda: tmp_path)
    (tmp_path / "geonames.db").write_bytes(b"")
    return tmp_path


@pytest.fixture
def pipeline():
    """Mock the model builders and the Geoparser they feed."""
    paris = _reference(0, 5, "Paris", _feature())
    atlantis = _reference(9, 17, "Atlantis", None)
    with (
        patch.object(parse_module, "_build_recognizer") as recognizer,
        patch.object(parse_module, "_build_resolver") as resolver,
        patch("geoparser.geoparser.Geoparser") as geoparser,
    ):
        geoparser.return_value.parse.side_effect = lambda texts: [
            _document(text, [paris, atlantis]) for text in texts
        ]
        yield SimpleNamespace(
            recognizer=recognizer, resolver=resolver, geoparser=geoparser
        )


@pytest.mark.unit
@pytest.mark.usefixtures("installed")
class TestParseCommand:
    """Test the parse command end to end through the Typer app."""

    def test_stdin_to_jsonl(self, pipeline):
        result = runner.invoke(app, ["parse", "-"], input="Paris or Atlantis")

        assert result.exit_code == 0, result.output
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "-"
        assert record["text"] == "Paris or Atlantis"
        paris, atlantis = record["toponyms"]
        assert paris["identifier"] == "2988507"
        assert paris["gazetteer"] == "geonames"
        assert paris["geometry"]["type"] == "Point"
        assert atlantis["identifier"] is None
        assert atlantis["geometry"] is None

    def test_no_arguments_reads_stdin(self, pipeline):
        result = runner.invoke(app, ["parse"], input="Paris")

        assert result.exit_code == 0, result.output
        pipeline.geoparser.return_value.parse.assert_called_once_with(["Paris"])

    def test_files_are_one_document_each(self, pipeline, tmp_path):
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text("Paris")
        second.write_text("Zurich")

        result = runner.invoke(app, ["parse", str(first), str(second)])

        assert result.exit_code == 0, result.output
        sources = [json.loads(line)["source"] for line in result.stdout.splitlines()]
        assert sources == [str(first), str(second)]

    def test_options_select_modules(self, pipeline):
        result = runner.invoke(
            app,
            [
                "parse",
                "--gazetteer",
                "geonames",
                "--recognizer",
                "gliner",
                "--recognizer-model",
                "r-model",
                "--resolver",
                "jina",
                "--model",
                "m-model",
            ],
            input="Paris",
        )

        assert result.exit_code == 0, result.output
        pipeline.recognizer.assert_called_once_with(
            parse_module.RecognizerChoice.gliner, "r-model"
        )
        pipeline.resolver.assert_called_once_with(
            parse_module.ResolverChoice.jina, "geonames", "m-model"
        )

    def test_json_format(self, pipeline):
        result = runner.invoke(app, ["parse", "--format", "json"], input="Paris")

        assert result.exit_code == 0, result.output
        records = json.loads(result.stdout)
        assert isinstance(records, list)
        assert records[0]["text"] == "Paris"

    def test_geojson_format(self, pipeline):
        result = runner.invoke(app, ["parse", "--format", "geojson"], input="Paris")

        assert result.exit_code == 0, result.output
        collection = json.loads(result.stdout)
        assert collection["type"] == "FeatureCollection"
        paris = collection["features"][0]
        assert paris["geometry"]["coordinates"] == [2.35, 48.85]
        assert paris["properties"]["text"] == "Paris"
        assert "geometry" not in paris["properties"]

    def test_output_file(self, pipeline, tmp_path):
        target = tmp_path / "out.jsonl"

        result = runner.invoke(app, ["parse", "-o", str(target)], input="Paris")

        assert result.exit_code == 0, result.output
        assert result.stdout == ""
        assert json.loads(target.read_text())["text"] == "Paris"

    @pytest.mark.parametrize(
        ("flag", "level"),
        [("-q", logging.WARNING), ("-v", logging.DEBUG), (None, logging.INFO)],
    )
    def test_verbosity_flags(self, pipeline, flag, level):
        arguments = ["parse"] if flag is None else ["parse", flag]

        result = runner.invoke(app, arguments, input="Paris")

        assert result.exit_code == 0, result.output
        assert logging.getLogger("geoparser").level == level


@pytest.mark.unit
def test_missing_gazetteer_exits_2_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr("geoparser.gazetteer.artifact.gazetteers_dir", lambda: tmp_path)
    with patch.object(parse_module, "_build_recognizer") as recognizer:
        result = runner.invoke(app, ["parse", "--gazetteer", "nope"], input="x")

    assert result.exit_code == 2
    assert "geoparser install nope" in result.stderr
    recognizer.assert_not_called()


@pytest.mark.unit
class TestBuilders:
    """Test that each choice builds the matching module class."""

    @pytest.mark.parametrize(
        ("choice", "target"),
        [
            ("spacy", "geoparser.modules.recognizers.spacy.SpacyRecognizer"),
            ("gliner", "geoparser.modules.recognizers.gliner.GLiNER2Recognizer"),
        ],
    )
    def test_recognizers(self, choice, target):
        with patch(target) as module:
            built = parse_module._build_recognizer(
                parse_module.RecognizerChoice(choice), None
            )
            parse_module._build_recognizer(
                parse_module.RecognizerChoice(choice), "custom"
            )

        assert built is module.return_value
        assert module.call_args_list[0].kwargs == {}
        assert module.call_args_list[1].kwargs == {"model_name": "custom"}

    @pytest.mark.parametrize(
        ("choice", "target"),
        [
            ("prior", "geoparser.modules.resolvers.prior.PriorResolver"),
            (
                "sentencetransformer",
                "geoparser.modules.resolvers.sentencetransformer."
                "SentenceTransformerResolver",
            ),
            ("jina", "geoparser.modules.resolvers.jina.JinaResolver"),
        ],
    )
    def test_resolvers(self, choice, target):
        with patch(target) as module:
            built = parse_module._build_resolver(
                parse_module.ResolverChoice(choice), "geonames", None
            )
            parse_module._build_resolver(
                parse_module.ResolverChoice(choice), "geonames", "custom"
            )

        assert built is module.return_value
        assert module.call_args_list[0].kwargs == {"gazetteer_name": "geonames"}
        assert module.call_args_list[1].kwargs == {
            "gazetteer_name": "geonames",
            "model_name": "custom",
        }


@pytest.mark.unit
class TestGeometry:
    """Test the WGS 84 conversion of feature geometries."""

    def test_missing_geometry_is_null(self):
        feature = Mock(geometry=None)

        assert parse_module._wgs84_geometry(feature) is None

    def test_projected_geometry_is_reprojected(self):
        # Bern in Swiss LV95 (EPSG:2056)
        feature = _feature(crs="EPSG:2056", geometry=Point(2600000, 1200000))

        geometry = parse_module._wgs84_geometry(feature)

        longitude, latitude = geometry["coordinates"]
        assert longitude == pytest.approx(7.44, abs=0.01)
        assert latitude == pytest.approx(46.95, abs=0.01)


@pytest.mark.unit
class TestListJson:
    """Test ``geoparser list --json``."""

    def test_lists_names_and_sizes(self, installed):
        (installed / "geonames.db").write_bytes(b"x" * 10)

        result = runner.invoke(app, ["list", "--json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == [{"name": "geonames", "size_bytes": 10}]

    def test_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "geoparser.gazetteer.artifact.gazetteers_dir", lambda: tmp_path
        )

        result = runner.invoke(app, ["list", "--json"])

        assert json.loads(result.stdout) == []
