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
    # Keep application imports lazy so CLI help does not import the web stack.
    from geoparser.annotator.server import run

    run(use_reloader=reload, host=host, port=port, open_browser=browser)
