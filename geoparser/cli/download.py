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
    [Deprecated] This command has been renamed to ``install``.

    Use ``install`` instead::

        geoparser install geonames

    Args:
        config: Either a gazetteer name (e.g., 'geonames', 'swissnames3d') or
                a path to a custom YAML configuration file.
    """
    typer.secho(
        f"Use 'install' instead:\n  geoparser install {config}",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=1)
