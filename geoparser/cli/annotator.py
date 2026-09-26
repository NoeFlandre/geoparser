import typing as t

import typer


def annotator_cli(
    host: t.Annotated[
        str,
        typer.Option(
            help="Interface to bind to. Use 0.0.0.0 to expose the server on "
            "every network interface (it has no authentication)."
        ),
    ] = "127.0.0.1",
    port: t.Annotated[int, typer.Option(help="Port to listen on.")] = 5000,
    browser: t.Annotated[
        bool, typer.Option(help="Open the annotator in a web browser on start.")
    ] = True,
    reload: t.Annotated[
        bool, typer.Option(help="Restart the server when source files change.")
    ] = False,
):
    """Launch the Irchel Geoparser Annotator web application."""
    # Imported here rather than at module scope: building the FastAPI app
    # pulls in spaCy and torch, which would make every CLI invocation --
    # including `--help` -- pay for a web server nobody asked to start.
    from geoparser.annotator.app import run

    run(use_reloader=reload, host=host, port=port, open_browser=browser)
