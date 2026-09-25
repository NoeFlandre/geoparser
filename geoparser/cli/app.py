import typing as t
from importlib.metadata import version

import typer

from geoparser.cli.annotator import annotator_cli
from geoparser.cli.download import download_cli
from geoparser.cli.install import install_cli, list_cli, uninstall_cli
from geoparser.cli.parse import parse_cli

app = typer.Typer(no_args_is_help=True)


def _version_callback(value: bool) -> None:
    """Print the installed package version and stop."""
    if value:
        typer.echo(version("geoparser"))
        raise typer.Exit


@app.callback()
def main(
    show_version: t.Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed geoparser version and exit.",
        ),
    ] = False,
) -> None:
    """Irchel Geoparser: gazetteer management, annotation and parsing."""


app.command("annotator")(annotator_cli)
app.command("parse")(parse_cli)
app.command("install")(install_cli)
app.command("list")(list_cli)
app.command("uninstall")(uninstall_cli)
app.command("download", deprecated=True)(download_cli)
