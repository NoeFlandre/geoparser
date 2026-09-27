"""Process launcher for the annotator web application."""

import threading
import webbrowser

import uvicorn

from geoparser.annotator.app import app
from geoparser.annotator.db.db import create_db_and_tables


def run(
    use_reloader: bool = False,  # noqa: FBT001, FBT002 - positional bool kept for API compatibility; make keyword-only in the next major release
    host: str = "127.0.0.1",
    port: int = 5000,
    open_browser: bool = True,  # noqa: FBT001, FBT002 - positional bool kept for API compatibility; make keyword-only in the next major release
) -> None:  # pragma: no cover
    """
    Initialize the annotator database and start Uvicorn.

    Args:
        use_reloader: Enable auto-reload for development (default: False)
        host: Host to bind the server to (default: "127.0.0.1", local only)
        port: Port to run the server on (default: 5000)
        open_browser: Automatically open browser on startup (default: True)
    """

    def launch_browser() -> None:
        browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host  # noqa: S104 - only maps a wildcard bind to a browsable loopback URL
        webbrowser.open_new(f"http://{browser_host}:{port}/")

    create_db_and_tables()
    if open_browser:
        threading.Timer(1.0, launch_browser).start()
    # uvicorn can only reload an application it imports itself.
    target = "geoparser.annotator.app:app" if use_reloader else app
    uvicorn.run(target, host=host, port=port, reload=use_reloader)
