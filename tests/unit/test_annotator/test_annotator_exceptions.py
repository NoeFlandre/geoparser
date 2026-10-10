"""Stable HTTP mappings for annotator CRUD exceptions."""

import json
from typing import Any, cast

import pytest

from geoparser.annotator import exceptions as annotator_exceptions
from geoparser.annotator.exceptions import (
    InvalidUploadException,
    SessionSettingsNotFoundException,
    ToponymOverlapException,
    invalid_upload_exception_handler,
    sessionsettings_exception_handler,
    toponym_overlap_exception_handler,
)


@pytest.mark.parametrize(
    ("legacy_name", "current_name"),
    [
        ("SessionNotFoundException", "SessionNotFoundError"),
        ("SessionSettingsNotFoundException", "SessionSettingsNotFoundError"),
        ("DocumentNotFoundException", "DocumentNotFoundError"),
        ("ToponymNotFoundException", "ToponymNotFoundError"),
        ("ToponymOverlapException", "ToponymOverlapError"),
        ("InvalidUploadException", "InvalidUploadError"),
    ],
)
def test_legacy_exception_names_alias_the_current_classes(legacy_name, current_name):
    """Each *Exception name is the same class as its *Error name."""
    legacy = getattr(annotator_exceptions, legacy_name)
    current = getattr(annotator_exceptions, current_name)

    assert legacy is current
    assert issubclass(current, Exception)


def test_missing_settings_handler_returns_not_found():
    """A missing settings row maps to the documented 404 response."""
    response = sessionsettings_exception_handler(
        cast(Any, None), SessionSettingsNotFoundException()
    )

    assert response.status_code == 404
    assert json.loads(bytes(response.body)) == {
        "status": "error",
        "message": "Settings not found.",
    }


def test_toponym_overlap_handler_returns_unprocessable_content():
    """An overlapping span maps to the documented 422 response."""
    response = toponym_overlap_exception_handler(
        cast(Any, None), ToponymOverlapException()
    )

    assert response.status_code == 422
    assert json.loads(bytes(response.body)) == {
        "status": "error",
        "message": "Overlap with existing toponym.",
    }


def test_invalid_upload_handler_returns_unprocessable_content():
    """An invalid upload maps to a clear 422 response."""
    response = invalid_upload_exception_handler(
        cast(Any, None), InvalidUploadException("Invalid UTF-8 session file.")
    )

    assert response.status_code == 422
    assert json.loads(bytes(response.body)) == {
        "status": "error",
        "message": "Invalid UTF-8 session file.",
    }
