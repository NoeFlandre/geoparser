import typing as t

import typer


def download_cli(
    config: t.Annotated[
        str,
        typer.Argument(
            help="Gazetteer name (e.g. geonames) or path to a custom YAML config."
        ),
    ],
) -> None:
    """
    [Deprecated] This command was renamed to install. It does not download
    anything. It prints the matching install command and exits with status 1.

    For example, run: geoparser install geonames
    """
    typer.secho(
        f"Use 'install' instead:\n  geoparser install {config}",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=1)
