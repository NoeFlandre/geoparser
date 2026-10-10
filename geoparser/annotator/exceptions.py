from fastapi import Request, status
from fastapi.responses import JSONResponse

from geoparser.annotator.models.api import BaseResponse


class SessionNotFoundError(Exception):
    pass


class SessionSettingsNotFoundError(Exception):
    pass


class DocumentNotFoundError(Exception):
    pass


class ToponymNotFoundError(Exception):
    pass


class ToponymOverlapError(Exception):
    pass


class InvalidUploadError(Exception):
    """An uploaded or legacy file cannot be decoded or validated."""


# The *Exception names are kept as aliases of the classes above. Existing
# imports, raises, and except clauses keep working unchanged.
SessionNotFoundException = SessionNotFoundError
SessionSettingsNotFoundException = SessionSettingsNotFoundError
DocumentNotFoundException = DocumentNotFoundError
ToponymNotFoundException = ToponymNotFoundError
ToponymOverlapException = ToponymOverlapError
InvalidUploadException = InvalidUploadError


def session_exception_handler(
    request: Request, exc: SessionNotFoundError
) -> JSONResponse:
    return JSONResponse(
        content={
            **BaseResponse(status="error", message="Session not found.").model_dump()
        },
        status_code=status.HTTP_404_NOT_FOUND,
    )


def sessionsettings_exception_handler(
    request: Request, exc: SessionSettingsNotFoundError
) -> JSONResponse:
    return JSONResponse(
        content={
            **BaseResponse(status="error", message="Settings not found.").model_dump()
        },
        status_code=status.HTTP_404_NOT_FOUND,
    )


def document_exception_handler(
    request: Request, exc: DocumentNotFoundError
) -> JSONResponse:
    return JSONResponse(
        content={
            **BaseResponse(
                status="error", message="Invalid document index."
            ).model_dump()
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


def toponym_exception_handler(
    request: Request, exc: ToponymNotFoundError
) -> JSONResponse:
    return JSONResponse(
        content={
            **BaseResponse(status="error", message="Toponym not found.").model_dump()
        },
        status_code=status.HTTP_404_NOT_FOUND,
    )


def toponym_overlap_exception_handler(
    request: Request, exc: ToponymOverlapError
) -> JSONResponse:
    return JSONResponse(
        content={
            **BaseResponse(
                status="error", message="Overlap with existing toponym."
            ).model_dump()
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


def invalid_upload_exception_handler(
    request: Request, exc: InvalidUploadError
) -> JSONResponse:
    return JSONResponse(
        content={
            **BaseResponse(status="error", message=str(exc)).model_dump(),
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
