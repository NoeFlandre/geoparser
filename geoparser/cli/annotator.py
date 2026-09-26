def annotator_cli():
    """Launch the Irchel Geoparser Annotator web application."""
    # Keep application imports lazy so CLI help does not import the web stack.
    from geoparser.annotator.server import run

    run()
