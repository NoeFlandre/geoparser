"""
Unit tests for geoparser/db/crud/document.py

Tests the DocumentRepository class with custom query methods.
"""

import uuid

import pytest
from sqlmodel import Session

from geoparser.db.crud import DocumentRepository
from geoparser.db.models import DocumentCreate, Project, ProjectCreate

CHUNK_SIZE = 500


def _project(session: Session, name: str) -> Project:
    """Persist a project and return it."""
    project = Project(**ProjectCreate(name=name).model_dump())
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def _documents(session: Session, project: Project, count: int):
    """Persist `count` documents in a project and return their ids."""
    return [
        DocumentRepository.create(
            session, DocumentCreate(text=f"doc {index}", project_id=project.id)
        ).id
        for index in range(count)
    ]


@pytest.mark.unit
class TestDocumentRepositoryGetByProject:
    """Test the get_by_project method of DocumentRepository."""

    def test_returns_documents_for_project(
        self,
        test_session: Session,
        project_factory,
        document_factory,
    ):
        """Test that get_by_project returns all documents for a project."""
        # Arrange
        project = project_factory()
        doc1 = document_factory(text="Document 1", project_id=project.id)
        doc2 = document_factory(text="Document 2", project_id=project.id)

        # Act
        documents = DocumentRepository.get_by_project(test_session, project.id)

        # Assert
        assert len(documents) == 2
        document_ids = [d.id for d in documents]
        assert doc1.id in document_ids
        assert doc2.id in document_ids

    def test_returns_empty_list_for_project_without_documents(
        self, test_session: Session, project_factory
    ):
        """Test that get_by_project returns empty list for project without documents."""
        # Arrange
        project = project_factory()

        # Act
        documents = DocumentRepository.get_by_project(test_session, project.id)

        # Assert
        assert documents == []

    def test_filters_by_project(
        self,
        test_session: Session,
        project_factory,
        document_factory,
    ):
        """Test that get_by_project only returns documents from specified project."""
        # Arrange
        project1 = project_factory(name="Project 1")
        project2 = project_factory(name="Project 2")

        # Documents in project1
        doc1_proj1 = document_factory(text="Doc in Project 1", project_id=project1.id)

        # Documents in project2
        document_factory(text="Doc in Project 2", project_id=project2.id)

        # Act - Get documents from project1
        documents = DocumentRepository.get_by_project(test_session, project1.id)

        # Assert - Should only contain doc from project1
        assert len(documents) == 1
        assert documents[0].id == doc1_proj1.id
        assert documents[0].text == "Doc in Project 1"


@pytest.mark.unit
class TestChunking:
    """Requests that span more than one chunk."""

    def test_a_request_larger_than_one_chunk_returns_every_document(self, test_session):
        project = _project(test_session, "big")
        ids = _documents(test_session, project, CHUNK_SIZE + 1)

        found = DocumentRepository.get_by_ids(test_session, project.id, ids)

        assert {document.id for document in found} == set(ids)

    def test_the_second_chunk_starts_where_the_first_ended(self, test_session):
        project = _project(test_session, "big")
        ids = _documents(test_session, project, CHUNK_SIZE + 3)

        found = DocumentRepository.get_by_ids(test_session, project.id, ids)

        returned = [document.id for document in found]
        assert len(returned) == len(set(returned)) == CHUNK_SIZE + 3

    def test_a_request_that_fits_in_one_chunk_is_unaffected(self, test_session):
        project = _project(test_session, "small")
        ids = _documents(test_session, project, 3)

        found = DocumentRepository.get_by_ids(test_session, project.id, ids)

        assert {document.id for document in found} == set(ids)

    def test_no_ids_means_no_query_and_no_documents(self, test_session):
        project = _project(test_session, "empty")
        _documents(test_session, project, 2)

        assert DocumentRepository.get_by_ids(test_session, project.id, []) == []


@pytest.mark.unit
class TestGetByIdsScoping:
    """Which documents a request is allowed to see."""

    def test_another_projects_document_is_not_returned(self, test_session):
        mine = _project(test_session, "mine")
        theirs = _project(test_session, "theirs")
        (my_id,) = _documents(test_session, mine, 1)
        (their_id,) = _documents(test_session, theirs, 1)

        found = DocumentRepository.get_by_ids(test_session, mine.id, [my_id, their_id])

        assert [document.id for document in found] == [my_id]

    def test_the_project_filter_survives_every_chunk(self, test_session):
        mine = _project(test_session, "mine")
        theirs = _project(test_session, "theirs")
        my_ids = _documents(test_session, mine, CHUNK_SIZE + 1)
        their_ids = _documents(test_session, theirs, 2)

        found = DocumentRepository.get_by_ids(
            test_session, mine.id, [*my_ids, *their_ids]
        )

        assert {document.id for document in found} == set(my_ids)

    def test_an_unknown_id_is_ignored_rather_than_reported(self, test_session):
        project = _project(test_session, "mine")
        (known,) = _documents(test_session, project, 1)

        found = DocumentRepository.get_by_ids(
            test_session, project.id, [known, uuid.uuid4()]
        )

        assert [document.id for document in found] == [known]
