"""HTML page routes for the annotation interface."""

import os
import typing as t
import uuid

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from spacy.util import get_installed_models
from sqlmodel import Session as DBSession

from geoparser.annotator.constants import GAZETTEERS
from geoparser.annotator.db.crud import DocumentRepository, SessionRepository
from geoparser.annotator.db.db import get_db
from geoparser.annotator.db.models import AnnotatorSessionForTemplate
from geoparser.annotator.dependencies import get_document, get_session
from geoparser.annotator.exceptions import (
    DocumentNotFoundException,
    SessionNotFoundException,
)

router = APIRouter()
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "../templates")
)
spacy_models = list(get_installed_models())


@router.get("/", tags=["pages"])
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="html/index.html")


@router.get("/start_new_session", tags=["pages"])
def start_new_session(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="html/start_new_session.html",
        context={"gazetteers": GAZETTEERS, "spacy_models": spacy_models},
    )


@router.get("/continue_session", tags=["pages"])
def continue_session(
    db: t.Annotated[DBSession, Depends(get_db)], request: Request
) -> HTMLResponse:
    cached_sessions = [
        AnnotatorSessionForTemplate(
            **session.model_dump(), num_documents=len(session.documents)
        )
        for session in SessionRepository.read_all(db)
    ]
    return templates.TemplateResponse(
        request=request,
        name="html/continue_session.html",
        context={"cached_sessions": cached_sessions},
    )


@router.get("/session/{session_id}/document/{doc_index}/annotate", tags=["pages"])
def annotate(
    request: Request,
    db: t.Annotated[DBSession, Depends(get_db)],
    session_id: uuid.UUID,
    doc_index: int = 0,
) -> Response:
    try:
        session = get_session(db, session_id)
    except SessionNotFoundException:
        return RedirectResponse(
            url=request.app.url_path_for("index"),
            status_code=status.HTTP_302_FOUND,
        )
    try:
        doc = get_document(session, doc_index)
    except DocumentNotFoundException:
        if doc_index > 0:
            return RedirectResponse(
                url=request.app.url_path_for(
                    "annotate", session_id=session.id, doc_index=0
                ),
                status_code=status.HTTP_302_FOUND,
            )
        return templates.TemplateResponse(
            request=request,
            name="html/annotate.html",
            context={
                "doc": None,
                "doc_index": None,
                "pre_annotated_text": None,
                "total_docs": 0,
                "gazetteer": session.gazetteer,
                "documents": [],
                "total_toponyms": 0,
                "annotated_toponyms": 0,
                "session_id": session.id,
                "spacy_models": spacy_models,
            },
        )

    pre_annotated_text = DocumentRepository.get_pre_annotated_text(db, doc.id)
    documents = DocumentRepository.get_progress(db, session_id=session.id)
    total_toponyms = len(doc.toponyms)
    annotated_toponyms = sum(toponym.loc_id != "" for toponym in doc.toponyms)
    return templates.TemplateResponse(
        request=request,
        name="html/annotate.html",
        context={
            "doc": doc,
            "doc_index": doc_index,
            "pre_annotated_text": pre_annotated_text,
            "total_docs": len(session.documents),
            "gazetteer": session.gazetteer,
            "documents": documents,
            "session_id": session.id,
            "total_toponyms": total_toponyms,
            "annotated_toponyms": annotated_toponyms,
            "spacy_models": spacy_models,
        },
    )
