"""Process launcher for the annotator web application."""

import threading
import webbrowser

import uvicorn

from geoparser.annotator.app import app
from geoparser.annotator.db.db import create_db_and_tables, engine


def run(
    use_reloader: bool = False,
    host: str = "0.0.0.0",
    port: int = 5000,
    open_browser: bool = True,
) -> None:  # pragma: no cover
    """Initialize the annotator database and start Uvicorn."""

    def launch_browser() -> None:
        webbrowser.open_new(f"http://127.0.0.1:{port}/")

    create_db_and_tables(engine)
    if open_browser:
        threading.Timer(1.0, launch_browser).start()
    uvicorn.run(app, host=host, port=port, reload=use_reloader)
