"""The ``geoparser parse`` command: text in, linked toponyms out."""

from __future__ import annotations

import enum
import json
import logging
import sys
import typing as t
from pathlib import Path

import typer

if t.TYPE_CHECKING:
    from geoparser.db.models import Document
    from geoparser.modules.recognizers import Recognizer
    from geoparser.modules.resolvers import Resolver

STDIN = "-"


class RecognizerChoice(str, enum.Enum):
    """Recognizers the command can build."""

    spacy = "spacy"
    gliner = "gliner"


class ResolverChoice(str, enum.Enum):
    """Resolvers the command can build."""

    prior = "prior"
    sentencetransformer = "sentencetransformer"
    jina = "jina"


class OutputFormat(str, enum.Enum):
    """Serializations the command can write."""

    jsonl = "jsonl"
    json = "json"
    geojson = "geojson"


def _build_recognizer(choice: RecognizerChoice, model: str | None) -> Recognizer:
    """
    Instantiate the chosen recognizer.

    Args:
        choice: Which recognizer to build.
        model: Model name to load instead of the recognizer's default.

    Returns:
        The recognizer.
    """
    kwargs = {} if model is None else {"model_name": model}
    if choice is RecognizerChoice.gliner:
        from geoparser.modules.recognizers.gliner import GLiNER2Recognizer

        return GLiNER2Recognizer(**kwargs)
    from geoparser.modules.recognizers.spacy import SpacyRecognizer

    return SpacyRecognizer(**kwargs)


def _build_resolver(
    choice: ResolverChoice, gazetteer: str, model: str | None
) -> Resolver:
    """
    Instantiate the chosen resolver against a gazetteer.

    Args:
        choice: Which resolver to build.
        gazetteer: Name of the installed gazetteer to resolve against.
        model: Model name to load instead of the resolver's default.

    Returns:
        The resolver.
    """
    kwargs: dict[str, t.Any] = {"gazetteer_name": gazetteer}
    if model is not None:
        kwargs["model_name"] = model
    if choice is ResolverChoice.jina:
        from geoparser.modules.resolvers.jina import JinaResolver

        return JinaResolver(**kwargs)
    if choice is ResolverChoice.sentencetransformer:
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        return SentenceTransformerResolver(**kwargs)
    from geoparser.modules.resolvers.prior import PriorResolver

    return PriorResolver(**kwargs)


def _read_inputs(inputs: list[str]) -> list[tuple[str, str]]:
    """
    Read each input as one document.

    Args:
        inputs: File paths, or ``-`` for standard input.

    Returns:
        (source, text) pairs in input order.
    """
    documents = []
    for source in inputs:
        if source == STDIN:
            documents.append((source, sys.stdin.read()))
        else:
            documents.append((source, Path(source).read_text(encoding="utf-8")))
    return documents


def _wgs84_geometry(feature: t.Any) -> dict[str, t.Any] | None:
    """
    Return a feature's geometry as a GeoJSON geometry in WGS 84.

    Args:
        feature: The gazetteer feature.

    Returns:
        GeoJSON geometry mapping, or None if the feature has none.
    """
    geometry = feature.geometry
    if geometry is None:
        return None
    from shapely.geometry import mapping

    if feature.crs.upper() != "EPSG:4326":
        from pyproj import Transformer
        from shapely.ops import transform

        project = Transformer.from_crs(
            feature.crs, "EPSG:4326", always_xy=True
        ).transform
        geometry = transform(project, geometry)
    return dict(mapping(geometry))


def _toponym_record(reference: t.Any) -> dict[str, t.Any]:
    """
    Serialize one recognized toponym and the feature it resolved to.

    Args:
        reference: The reference to serialize.

    Returns:
        JSON-serializable record.
    """
    feature = reference.location
    return {
        "start": reference.start,
        "end": reference.end,
        "text": reference.text,
        "gazetteer": None if feature is None else feature.gazetteer_name,
        "identifier": None if feature is None else feature.identifier,
        "geometry": None if feature is None else _wgs84_geometry(feature),
    }


def _document_record(source: str, document: Document) -> dict[str, t.Any]:
    """
    Serialize a parsed document.

    Args:
        source: Where the text came from (a path or ``-``).
        document: The parsed document.

    Returns:
        JSON-serializable record.
    """
    return {
        "source": source,
        "text": document.text,
        "toponyms": [_toponym_record(ref) for ref in document.toponyms],
    }


def _geojson(records: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    """
    Turn document records into one FeatureCollection of resolved toponyms.

    Args:
        records: Records from :func:`_document_record`.

    Returns:
        GeoJSON FeatureCollection; unresolved toponyms have a null geometry.
    """
    features = [
        {
            "type": "Feature",
            "geometry": toponym["geometry"],
            "properties": {
                "source": record["source"],
                **{k: v for k, v in toponym.items() if k != "geometry"},
            },
        }
        for record in records
        for toponym in record["toponyms"]
    ]
    return {"type": "FeatureCollection", "features": features}


def _serialize(records: list[dict[str, t.Any]], output_format: OutputFormat) -> str:
    """
    Render records in the requested format.

    Args:
        records: Records from :func:`_document_record`.
        output_format: The serialization to produce.

    Returns:
        The rendered output, ending in a newline.
    """
    if output_format is OutputFormat.json:
        return json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    if output_format is OutputFormat.geojson:
        return json.dumps(_geojson(records), ensure_ascii=False, indent=2) + "\n"
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def _configure_logging(quiet: bool, verbose: bool) -> None:
    """
    Send library progress to stderr so stdout carries only the results.

    Args:
        quiet: Show warnings and errors only.
        verbose: Show debug messages too.
    """
    level = logging.WARNING if quiet else logging.DEBUG if verbose else logging.INFO
    # Configuring the root logger makes the package's default stdout handler
    # stand aside, so progress cannot interleave with the JSON on stdout.
    logging.basicConfig(stream=sys.stderr, format="%(message)s", force=True)
    logging.getLogger("geoparser").setLevel(level)


def parse_cli(
    inputs: t.Annotated[
        list[str] | None,
        typer.Argument(
            help="Text files to parse, one document each. '-' (the default) "
            "reads a single document from standard input.",
            show_default=False,
        ),
    ] = None,
    gazetteer: t.Annotated[
        str, typer.Option(help="Installed gazetteer to resolve against.")
    ] = "geonames",
    recognizer: t.Annotated[
        RecognizerChoice, typer.Option(help="Recognizer that finds toponyms.")
    ] = RecognizerChoice.spacy,
    resolver: t.Annotated[
        ResolverChoice, typer.Option(help="Resolver that links toponyms.")
    ] = ResolverChoice.prior,
    model: t.Annotated[
        str | None,
        typer.Option(help="Model for the resolver instead of its default."),
    ] = None,
    recognizer_model: t.Annotated[
        str | None,
        typer.Option(help="Model for the recognizer instead of its default."),
    ] = None,
    output_format: t.Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.jsonl,
    output: t.Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
    quiet: t.Annotated[
        bool, typer.Option("--quiet", "-q", help="Only report warnings and errors.")
    ] = False,
    verbose: t.Annotated[
        bool, typer.Option("--verbose", "-v", help="Report debug messages too.")
    ] = False,
):
    """
    Parse text into toponyms linked to a gazetteer.

    Results are written as JSON Lines to stdout by default, one record per
    input document; progress messages go to stderr.
    """
    from geoparser.gazetteer.artifact import artifact_path

    # Checked first: loading the models takes far longer than this, and would
    # only end in the same error.
    if not artifact_path(gazetteer).exists():
        typer.secho(
            f"Gazetteer '{gazetteer}' is not installed. "
            f"Install it with: geoparser install {gazetteer}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    _configure_logging(quiet, verbose)
    documents = _read_inputs(inputs or [STDIN])

    from geoparser.geoparser import Geoparser

    geoparser = Geoparser(
        recognizer=_build_recognizer(recognizer, recognizer_model),
        resolver=_build_resolver(resolver, gazetteer, model),
    )
    parsed = geoparser.parse([text for _, text in documents])
    records = [
        _document_record(source, document)
        for (source, _), document in zip(documents, parsed, strict=True)
    ]

    rendered = _serialize(records, output_format)
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")
