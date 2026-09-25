import typing as t
from importlib.resources import files
from pathlib import Path

import typer


def _get_builtin_gazetteers() -> dict[str, Path]:
    """
    Discover all built-in gazetteer configurations.

    Returns:
        Dictionary mapping gazetteer names to their config file paths.
    """
    configs_dir = files("geoparser.gazetteer") / "configs"

    # Get all yaml files in the configs directory
    gazetteers = {}
    for config_file in configs_dir.iterdir():
        if config_file.name.endswith(".yaml"):
            gazetteer_name = config_file.name[:-5]  # Remove .yaml extension
            gazetteers[gazetteer_name] = Path(str(config_file))

    return gazetteers


def _is_installed(config_path: Path) -> bool:
    """
    Report (and say) whether the gazetteer a config describes is installed.

    Args:
        config_path: Path to the gazetteer's YAML configuration.

    Returns:
        True if an artifact with the config's name already exists.
    """
    from geoparser.gazetteer.artifact import artifact_path
    from geoparser.gazetteer.build.schema import GazetteerConfig

    name = GazetteerConfig.from_yaml(config_path).name
    if not artifact_path(name).exists():
        return False
    typer.echo(f"Gazetteer '{name}' is already installed. Use --force to rebuild it.")
    return True


def install_cli(
    config: t.Annotated[
        str,
        typer.Argument(
            help="A built-in gazetteer name (e.g. 'geonames') or a path to a YAML config."
        ),
    ],
    keep_downloads: t.Annotated[
        bool,
        typer.Option(help="Keep the downloaded source files after the build."),
    ] = False,
    force: t.Annotated[
        bool,
        typer.Option(help="Rebuild the gazetteer even if it is already installed."),
    ] = False,
    verbose: t.Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show the full traceback on failure."),
    ] = False,
):
    """
    Install a gazetteer from a configuration file.

    Args:
        config: Either a gazetteer name (e.g., 'geonames', 'swissnames3d') or
                a path to a custom YAML configuration file.
        keep_downloads: Keep the downloaded source files after the build.
        force: Rebuild even when the gazetteer is already installed.
        verbose: Re-raise build failures with their traceback instead of
                 printing a one-line error.
    """
    # Check if config is a built-in gazetteer name
    config_path = Path(config)

    if not config_path.exists():
        # Get available built-in gazetteers
        builtin_gazetteers = _get_builtin_gazetteers()

        if config in builtin_gazetteers:
            config_path = builtin_gazetteers[config]
        else:
            available = "\n".join(
                f"  - {name}" for name in sorted(builtin_gazetteers.keys())
            )
            typer.secho(
                f"Gazetteer config not found: {config}\n"
                f"Available built-in gazetteer configs:\n{available}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    # Imported per command: the build pipeline pulls in the heavy stack, and
    # `list` and `uninstall` have no reason to wait for it.
    from geoparser.gazetteer.build import GazetteerBuilder

    try:
        if not force and _is_installed(config_path):
            return
        GazetteerBuilder().build(config_path, keep_downloads=keep_downloads)
    except Exception as error:
        # A failed download or a full disk is a user-facing condition, not a
        # bug report: one line by default, the traceback on request.
        if verbose:
            raise
        typer.secho(
            f"Failed to install gazetteer from {config}: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo("Re-run with --verbose for the full traceback.", err=True)
        raise typer.Exit(code=1) from error


def list_cli():
    """
    List installed gazetteers.
    """
    from geoparser.gazetteer.artifact import artifact_path, list_artifacts

    names = list_artifacts()
    if not names:
        typer.echo("No gazetteers installed.")
        return
    for name in names:
        size = artifact_path(name).stat().st_size
        typer.echo(f"{name}  ({size / 1024 / 1024:.1f} MB)")


def uninstall_cli(
    name: t.Annotated[str, typer.Argument(help="Name of the gazetteer to remove.")],
    yes: t.Annotated[
        bool, typer.Option("--yes", "-y", help="Remove without asking first.")
    ] = False,
):
    """
    Remove an installed gazetteer.

    Args:
        name: Name of the gazetteer to remove.
        yes: Skip the confirmation prompt.
    """
    from geoparser.gazetteer.artifact import artifact_path
    from geoparser.gazetteer.build.builder import uninstall

    # Only ask when there is something to delete; a missing gazetteer is
    # reported below either way.
    if not yes and artifact_path(name).exists():
        typer.confirm(f"Remove gazetteer '{name}'?", abort=True)

    if uninstall(name):
        typer.echo(f"Removed gazetteer '{name}'.")
    else:
        typer.secho(
            f"Gazetteer '{name}' is not installed.", fg=typer.colors.YELLOW, err=True
        )
        raise typer.Exit(code=1)
