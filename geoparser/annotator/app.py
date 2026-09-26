"""FastAPI application setup for the annotation interface."""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from geoparser.annotator.exceptions import (
    DocumentNotFoundException,
    InvalidUploadException,
    SessionNotFoundException,
    SessionSettingsNotFoundException,
    ToponymNotFoundException,
    ToponymOverlapException,
    document_exception_handler,
    invalid_upload_exception_handler,
    session_exception_handler,
    sessionsettings_exception_handler,
    toponym_exception_handler,
    toponym_overlap_exception_handler,
)
from geoparser.annotator.metadata import tags_metadata
from geoparser.annotator.routes import annotations, documents, pages, sessions, settings

app = FastAPI(
    title="Irchel Geoparser",
    summary="API docs for the Irchel Geoparser annotator.",
    openapi_tags=tags_metadata,
)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# Starlette types handlers as Exception while these handlers accept their
# registered exception subclasses.
app.add_exception_handler(
    SessionNotFoundException,
    session_exception_handler,  # ty: ignore[invalid-argument-type]
)
app.add_exception_handler(
    SessionSettingsNotFoundException,
    sessionsettings_exception_handler,  # ty: ignore[invalid-argument-type]
)
app.add_exception_handler(
    DocumentNotFoundException,
    document_exception_handler,  # ty: ignore[invalid-argument-type]
)
app.add_exception_handler(
    ToponymNotFoundException,
    toponym_exception_handler,  # ty: ignore[invalid-argument-type]
)
app.add_exception_handler(
    ToponymOverlapException,
    toponym_overlap_exception_handler,  # ty: ignore[invalid-argument-type]
)
app.add_exception_handler(
    InvalidUploadException,
    invalid_upload_exception_handler,  # ty: ignore[invalid-argument-type]
)

for route_module in (pages, sessions, documents, annotations, settings):
    app.include_router(route_module.router)
