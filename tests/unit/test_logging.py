"""The default output behind geoparser's progress messages."""

import logging

import pytest

from geoparser._logging import PACKAGE_LOGGER, _DefaultStdoutHandler, get_logger


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        "geoparser.x", logging.INFO, __file__, 1, message, (), None
    )


@pytest.mark.unit
class TestDefaultOutput:
    def test_package_logger_defaults_to_info_with_one_handler(self):
        get_logger("geoparser.a")
        logger = get_logger("geoparser.b")

        package = logging.getLogger(PACKAGE_LOGGER)
        assert logger.name == "geoparser.b"
        assert package.level == logging.INFO
        handlers = [h for h in package.handlers if isinstance(h, _DefaultStdoutHandler)]
        assert len(handlers) == 1

    def test_keeps_a_level_the_application_chose(self, monkeypatch):
        package = logging.getLogger(PACKAGE_LOGGER)
        monkeypatch.setattr(package, "level", logging.WARNING)

        get_logger("geoparser.c")

        assert package.level == logging.WARNING

    def test_prints_bare_message_when_logging_is_unconfigured(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(logging.getLogger(), "handlers", [])

        _DefaultStdoutHandler().emit(_record("Starting model fine-tuning..."))

        assert capsys.readouterr().out == "Starting model fine-tuning...\n"

    def test_stands_aside_once_root_logging_is_configured(self, monkeypatch, capsys):
        monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])

        _DefaultStdoutHandler().emit(_record("hidden"))

        assert capsys.readouterr().out == ""

    def test_reports_emit_errors_through_logging(self, monkeypatch):
        monkeypatch.setattr(logging.getLogger(), "handlers", [])
        handler = _DefaultStdoutHandler()
        errors = []
        monkeypatch.setattr(handler, "format", lambda record: 1 / 0)
        monkeypatch.setattr(handler, "handleError", errors.append)
        record = _record("x")

        handler.emit(record)

        assert errors == [record]
