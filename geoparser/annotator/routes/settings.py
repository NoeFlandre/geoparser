"""Session settings routes."""

import typing as t

from fastapi import APIRouter, Depends
from sqlmodel import Session as DBSession

from geoparser.annotator.db.crud import SessionSettingsRepository
from geoparser.annotator.db.db import get_db
from geoparser.annotator.db.models import (
    AnnotatorSession,
    AnnotatorSessionSettings,
    AnnotatorSessionSettingsBase,
    AnnotatorSessionSettingsUpdate,
)
from geoparser.annotator.dependencies import get_session
from geoparser.annotator.models.api import BaseResponse

router = APIRouter()


@router.get("/session/{session_id}/settings", tags=["settings"])
def get_session_settings(
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
) -> AnnotatorSessionSettings:
    return session.settings


@router.put("/session/{session_id}/settings", tags=["settings"])
def put_session_settings(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    session_settings: AnnotatorSessionSettingsBase,
) -> BaseResponse:
    SessionSettingsRepository.update(
        db,
        AnnotatorSessionSettingsUpdate(
            id=session.settings.id, **session_settings.model_dump()
        ),
    )
    return BaseResponse()
