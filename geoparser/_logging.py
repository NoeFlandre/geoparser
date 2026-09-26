"""Progress messages for the library, routed through :mod:`logging`.

Library code reports progress (model downloads, fine-tuning steps, saved
projects) through loggers under ``geoparser``. So that those messages still
reach people who never configure logging, the ``geoparser`` logger defaults
to INFO and carries a handler that prints the bare message to stdout, exactly
as the former ``print()`` calls did. The handler stands aside as soon as the
application configures the root logger, so configured apps get each message
once, through their own handlers.
"""

import logging
import sys

PACKAGE_LOGGER = "geoparser"


class _DefaultStdoutHandler(logging.Handler):
    """Print records to the current stdout while logging is unconfigured."""

    def emit(self, record: logging.LogRecord) -> None:
        if logging.getLogger().handlers:
            return
        try:
            # Written to the stream directly: print()'s implicit stdout would
            # make `file=sys.stdout` a no-op mutation nothing could detect.
            sys.stdout.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001 - a logging handler must never raise; errors go to handleError
            self.handleError(record)


def _configure_package_logger() -> None:
    logger = logging.getLogger(PACKAGE_LOGGER)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    if not any(isinstance(h, _DefaultStdoutHandler) for h in logger.handlers):
        logger.addHandler(_DefaultStdoutHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a geoparser module, with the default output set up."""
    _configure_package_logger()
    return logging.getLogger(name)
