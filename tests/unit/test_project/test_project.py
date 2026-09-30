"""
Unit tests for geoparser/project/project.py

Tests the Project class with mocked dependencies.
"""

import builtins
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, Mock, patch
from uuid import UUID, uuid4

import pytest

from geoparser.project.project import Project


@pytest.mark.unit
class TestProjectInitialization:
    """Test Project initialization."""

    @patch("geoparser.project.project.ProjectRepository")
    def test_creates_new_project_when_doesnt_exist(self, mock_project_repo):
        """Test that Project creates a new project record when it doesn't exist."""
        # Arrange
        mock_project_repo.get_by_name.return_value = None  # Project doesn't exist

        mock_created_project = Mock()
        mock_created_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.create.return_value = mock_created_project

        # Act
        project = Project("NewProject")

        # Assert
        assert project.name == "NewProject"
        assert project.id == UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.create.assert_called_once()

    @patch("geoparser.project.project.ProjectRepository")
    def test_loads_existing_project_when_exists(self, mock_project_repo):
        """Test that Project loads existing project record when it exists."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("87654321-4321-8765-4321-876543218765")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        # Act
        project = Project("ExistingProject")

        # Assert
        assert project.name == "ExistingProject"
        assert project.id == UUID("87654321-4321-8765-4321-876543218765")
        mock_project_repo.create.assert_not_called()


@pytest.mark.unit
class TestProjectCreateDocuments:
    """Test Project create_documents method."""

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_creates_single_document(self, mock_doc_repo, mock_project_repo):
        """Test that create_documents creates a single document from a one-text list."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        project = Project("TestProject")

        # Act
        project.create_documents(["Test document text"])

        # Assert
        mock_doc_repo.create_many.assert_called_once()
        document_creates = mock_doc_repo.create_many.call_args.args[1]
        assert len(document_creates) == 1
        assert document_creates[0].text == "Test document text"
        assert document_creates[0].project_id == project.id

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_rejects_a_bare_string(self, mock_doc_repo, mock_project_repo):
        """Test that create_documents rejects a string instead of splitting it up."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        project = Project("TestProject")

        # Act & Assert
        with pytest.raises(TypeError, match="expects a sequence of texts"):
            project.create_documents("Test document text")

        mock_doc_repo.create_many.assert_not_called()

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_creates_multiple_documents(self, mock_doc_repo, mock_project_repo):
        """Test that create_documents creates multiple documents from a list."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        project = Project("TestProject")

        # Act
        project.create_documents(["Doc 1", "Doc 2", "Doc 3"])

        # Assert
        mock_doc_repo.create_many.assert_called_once()
        document_creates = mock_doc_repo.create_many.call_args.args[1]
        call_args_list = [document_create.text for document_create in document_creates]
        assert "Doc 1" in call_args_list
        assert "Doc 2" in call_args_list
        assert "Doc 3" in call_args_list

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_returns_document_ids_in_input_order(
        self, mock_doc_repo, mock_project_repo
    ):
        """Test that create_documents returns the new IDs in the order of the texts."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        created_ids = [uuid4(), uuid4(), uuid4()]
        mock_doc_repo.create_many.return_value = list(created_ids)

        project = Project("TestProject")

        # Act
        document_ids = project.create_documents(["Doc 1", "Doc 2", "Doc 3"])

        # Assert
        assert document_ids == created_ids


@pytest.mark.unit
class TestProjectGetDocumentsByIds:
    """Test Project get_documents method when specific IDs are requested."""

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_returns_documents_in_requested_order(
        self, mock_doc_repo, mock_project_repo, mock_context
    ):
        """Test that get_documents returns documents in the order the IDs were given."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        first_id, second_id = uuid4(), uuid4()
        mock_doc1 = Mock(id=first_id, references=[])
        mock_doc2 = Mock(id=second_id, references=[])

        # The repository makes no promises about ordering
        mock_doc_repo.get_by_ids.return_value = [mock_doc2, mock_doc1]

        mock_context_instance = Mock()
        mock_context_instance.get_recognizer_context.return_value = None
        mock_context_instance.get_resolver_context.return_value = None
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        # Act
        documents = project.get_documents(ids=[first_id, second_id])

        # Assert
        assert documents == [mock_doc1, mock_doc2]
        mock_doc_repo.get_by_ids.assert_called_once_with(
            ANY, project.id, [first_id, second_id]
        )

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_accepts_ids_as_strings(
        self, mock_doc_repo, mock_project_repo, mock_context
    ):
        """Test that get_documents accepts IDs given as strings."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        document_id = uuid4()
        mock_doc = Mock(id=document_id, references=[])
        mock_doc_repo.get_by_ids.return_value = [mock_doc]

        mock_context_instance = Mock()
        mock_context_instance.get_recognizer_context.return_value = None
        mock_context_instance.get_resolver_context.return_value = None
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        # Act
        documents = project.get_documents(ids=str(document_id))

        # Assert
        assert documents == [mock_doc]
        mock_doc_repo.get_by_ids.assert_called_once_with(ANY, project.id, [document_id])

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_raises_for_unknown_id(
        self, mock_doc_repo, mock_project_repo, mock_context
    ):
        """Test that get_documents raises if an ID is not in the project."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        unknown_id = uuid4()
        mock_doc_repo.get_by_ids.return_value = []

        mock_context_instance = Mock()
        mock_context_instance.get_recognizer_context.return_value = None
        mock_context_instance.get_resolver_context.return_value = None
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        # Act & Assert
        with pytest.raises(ValueError, match=str(unknown_id)):
            project.get_documents(ids=[unknown_id])

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_raises_for_value_that_is_not_an_id(self, mock_doc_repo, mock_project_repo):
        """Test that a value that isn't an ID is reported with a hint about tags."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        project = Project("TestProject")

        # Act & Assert
        with pytest.raises(ValueError, match="not a valid document ID"):
            project.get_documents(ids="baseline")


@pytest.mark.unit
class TestProjectGetDocuments:
    """Test Project get_documents method."""

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_retrieves_documents_for_project(
        self, mock_doc_repo, mock_project_repo, mock_context
    ):
        """Test that get_documents retrieves all documents for the project."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_doc1 = Mock()
        mock_doc1.references = []
        mock_doc2 = Mock()
        mock_doc2.references = []
        mock_doc_repo.get_by_project.return_value = [mock_doc1, mock_doc2]

        # Mock context to return None for both IDs
        mock_context_instance = Mock()
        mock_context_instance.get_recognizer_context.return_value = None
        mock_context_instance.get_resolver_context.return_value = None
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        # Act
        documents = project.get_documents()

        # Assert
        assert len(documents) == 2
        mock_doc_repo.get_by_project.assert_called_once_with(ANY, project.id)

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_sets_recognizer_context_on_documents(
        self, mock_doc_repo, mock_project_repo
    ):
        """Test that get_documents sets recognizer context on documents."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_doc = Mock()
        mock_doc.references = []
        mock_doc_repo.get_by_project.return_value = [mock_doc]

        project = Project("TestProject")

        # Act - Using tag to retrieve documents
        # Note: The implementation retrieves IDs from context, but we're testing the context setting behavior
        with (
            patch.object(
                project.context,
                "get_recognizer_context",
                return_value="test_recognizer",
            ),
            patch.object(project.context, "get_resolver_context", return_value=None),
        ):
            project.get_documents(tag="test_tag")

        # Assert
        mock_doc._set_recognizer_context.assert_called_once_with("test_recognizer")

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.DocumentRepository")
    def test_sets_resolver_context_on_references(
        self, mock_doc_repo, mock_project_repo
    ):
        """Test that get_documents sets resolver context on references."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_ref1 = Mock()
        mock_ref2 = Mock()
        mock_doc = Mock()
        mock_doc.references = [mock_ref1, mock_ref2]
        mock_doc_repo.get_by_project.return_value = [mock_doc]

        project = Project("TestProject")

        # Act - Using tag to retrieve documents
        with (
            patch.object(project.context, "get_recognizer_context", return_value=None),
            patch.object(
                project.context, "get_resolver_context", return_value="test_resolver"
            ),
        ):
            project.get_documents(tag="test_tag")

        # Assert
        mock_ref1._set_resolver_context.assert_called_once_with("test_resolver")
        mock_ref2._set_resolver_context.assert_called_once_with("test_resolver")


@pytest.mark.unit
class TestProjectRunRecognizer:
    """Test Project run_recognizer method."""

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.RecognitionService")
    @patch("geoparser.project.project.DocumentRepository")
    def test_runs_recognizer_on_all_documents(
        self, mock_doc_repo, mock_recognition_service, mock_project_repo, mock_context
    ):
        """Test that run_recognizer runs recognizer on all project documents."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_doc1 = Mock()
        mock_doc1.references = []
        mock_doc2 = Mock()
        mock_doc2.references = []
        mock_doc_repo.get_by_project.return_value = [mock_doc1, mock_doc2]

        mock_recognizer = Mock()
        mock_recognizer.id = "test_recognizer_id"
        mock_service_instance = Mock()
        mock_recognition_service.return_value = mock_service_instance

        # Mock context
        mock_context_instance = Mock()
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        # Act
        project.run_recognizer(mock_recognizer)

        # Assert
        mock_recognition_service.assert_called_once_with(mock_recognizer)
        mock_service_instance.predict.assert_called_once()
        called_docs = mock_service_instance.predict.call_args[0][0]
        assert len(called_docs) == 2


@pytest.mark.unit
class TestProjectRunResolver:
    """Test Project run_resolver method."""

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.ResolutionService")
    @patch("geoparser.project.project.DocumentRepository")
    def test_runs_resolver_on_all_documents(
        self, mock_doc_repo, mock_resolution_service, mock_project_repo, mock_context
    ):
        """Test that run_resolver runs resolver on all project documents."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_doc1 = Mock()
        mock_doc1.references = []
        mock_doc2 = Mock()
        mock_doc2.references = []
        mock_doc_repo.get_by_project.return_value = [mock_doc1, mock_doc2]

        mock_resolver = Mock()
        mock_resolver.id = "test_resolver_id"
        mock_service_instance = Mock()
        mock_resolution_service.return_value = mock_service_instance

        # Mock context
        mock_context_instance = Mock()
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        # Act
        project.run_resolver(mock_resolver)

        # Assert
        mock_resolution_service.assert_called_once_with(mock_resolver)
        mock_service_instance.predict.assert_called_once()
        called_docs = mock_service_instance.predict.call_args[0][0]
        assert len(called_docs) == 2


@pytest.mark.unit
class TestProjectTrainResolver:
    """Test training a resolver from a selected project context."""

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.ResolutionService")
    def test_trains_resolver_with_tag_and_options(
        self, mock_resolution_service, mock_project_repo
    ):
        """The selected documents and caller options reach the service."""
        project_record = Mock(id=UUID("12345678-1234-5678-1234-567812345678"))
        mock_project_repo.get_by_name.return_value = project_record
        resolver = Mock()
        documents = [Mock(), Mock()]
        service = Mock()
        mock_resolution_service.return_value = service
        project = Project("TrainingProject")

        with patch.object(project, "get_documents", return_value=documents) as get_docs:
            project.train_resolver(resolver, tag="training", epochs=3)

        get_docs.assert_called_once_with(tag="training")
        mock_resolution_service.assert_called_once_with(resolver)
        service.fit.assert_called_once_with(documents, epochs=3)


@pytest.mark.unit
class TestProjectDelete:
    """Test Project delete method."""

    @patch("geoparser.project.project.ProjectRepository")
    def test_deletes_project_from_database(self, mock_project_repo):
        """Test that delete removes the project from the database."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        project = Project("TestProject")

        # Act
        project.delete()

        # Assert
        mock_project_repo.delete.assert_called_once_with(ANY, id=project.id)


@pytest.mark.unit
class TestProjectCreateReferences:
    """Test Project create_references method."""

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.ManualRecognizer")
    @patch("geoparser.project.project.RecognitionService")
    @patch("geoparser.project.project.DocumentRepository")
    def test_creates_manual_recognizer_and_runs_it(
        self,
        mock_doc_repo,
        mock_recognition_service,
        mock_manual_recognizer,
        mock_project_repo,
        mock_context,
    ):
        """Test that create_references creates ManualRecognizer and runs it."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_doc = Mock()
        mock_doc.references = []
        mock_doc_repo.get_by_project.return_value = [mock_doc]

        mock_recognizer_instance = Mock()
        mock_recognizer_instance.id = "test_recognizer_id"
        mock_manual_recognizer.return_value = mock_recognizer_instance

        mock_service_instance = Mock()
        mock_recognition_service.return_value = mock_service_instance

        # Mock context
        mock_context_instance = Mock()
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        texts = ["Doc 1"]
        references = [[(0, 5)]]

        # Act
        project.create_references(texts, references, tag="test_tag")

        # Assert
        mock_manual_recognizer.assert_called_once_with(
            label="test_tag", texts=texts, references=references
        )
        mock_recognition_service.assert_called_once_with(mock_recognizer_instance)
        mock_service_instance.predict.assert_called_once()


@pytest.mark.unit
class TestProjectCreateReferents:
    """Test Project create_referents method."""

    @patch("geoparser.project.project.Context")
    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.ManualResolver")
    @patch("geoparser.project.project.ResolutionService")
    @patch("geoparser.project.project.DocumentRepository")
    def test_creates_manual_resolver_and_runs_it(
        self,
        mock_doc_repo,
        mock_resolution_service,
        mock_manual_resolver,
        mock_project_repo,
        mock_context,
    ):
        """Test that create_referents creates ManualResolver and runs it."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        mock_doc = Mock()
        mock_doc.references = []
        mock_doc_repo.get_by_project.return_value = [mock_doc]

        mock_resolver_instance = Mock()
        mock_resolver_instance.id = "test_resolver_id"
        mock_manual_resolver.return_value = mock_resolver_instance

        mock_service_instance = Mock()
        mock_resolution_service.return_value = mock_service_instance

        # Mock context
        mock_context_instance = Mock()
        mock_context.return_value = mock_context_instance

        project = Project("TestProject")

        texts = ["Doc 1"]
        references = [[(0, 5)]]
        referents = [[("geonames", "123")]]

        # Act
        project.create_referents(texts, references, referents, tag="test_tag")

        # Assert
        mock_manual_resolver.assert_called_once_with(
            label="test_tag",
            texts=texts,
            references=references,
            referents=referents,
        )
        mock_resolution_service.assert_called_once_with(mock_resolver_instance)
        mock_service_instance.predict.assert_called_once()


@pytest.mark.unit
class TestProjectLoadAnnotations:
    """Test Project load_annotations method."""

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.open", create=True)
    def test_loads_annotations_from_json_file(self, mock_file_open, mock_project_repo):
        """Test that load_annotations loads and parses JSON file."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        json_data = {
            "gazetteer": "geonames",
            "documents": [
                {
                    "text": "Paris is beautiful",
                    "toponyms": [
                        {"start": 0, "end": 5, "loc_id": "123"},
                        {"start": 10, "end": 19, "loc_id": ""},  # Not geocoded
                    ],
                }
            ],
        }

        mock_file_open.return_value.__enter__.return_value.read.return_value = str(
            json_data
        )

        # Mock json.load
        with (
            patch("geoparser.project.project.json.load", return_value=json_data),
            patch.object(Project, "create_references"),
            patch.object(Project, "create_referents"),
        ):
            project = Project("TestProject")

            # Act
            project.load_annotations("test.json", tag="test_tag")

            # Assert - Verify file was opened
            mock_file_open.assert_called_once()

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.open", create=True)
    def test_creates_documents_when_requested(self, mock_file_open, mock_project_repo):
        """Test that load_annotations creates documents when create_documents=True."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        json_data = {
            "gazetteer": "geonames",
            "documents": [{"text": "Paris", "toponyms": []}],
        }

        with (
            patch("geoparser.project.project.json.load", return_value=json_data),
            patch.object(Project, "create_documents") as mock_create_docs,
            patch.object(Project, "create_references"),
            patch.object(Project, "create_referents"),
        ):
            project = Project("TestProject")

            # Act
            project.load_annotations("test.json", tag="test_tag", create_documents=True)

            # Assert
            mock_create_docs.assert_called_once_with(["Paris"])

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.open", create=True)
    def test_skips_document_creation_by_default(
        self, mock_file_open, mock_project_repo
    ):
        """Test that load_annotations doesn't create documents by default."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        json_data = {
            "gazetteer": "geonames",
            "documents": [{"text": "Paris", "toponyms": []}],
        }

        with (
            patch("geoparser.project.project.json.load", return_value=json_data),
            patch.object(Project, "create_documents") as mock_create_docs,
            patch.object(Project, "create_references"),
            patch.object(Project, "create_referents"),
        ):
            project = Project("TestProject")

            # Act
            project.load_annotations("test.json", tag="test_tag")

            # Assert
            mock_create_docs.assert_not_called()

    @patch("geoparser.project.project.ProjectRepository")
    @patch("geoparser.project.project.open", create=True)
    def test_filters_out_non_geocoded_referents(
        self, mock_file_open, mock_project_repo
    ):
        """Test that load_annotations filters out non-geocoded referents (empty or null loc_id)."""
        # Arrange

        mock_existing_project = Mock()
        mock_existing_project.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_project_repo.get_by_name.return_value = mock_existing_project

        json_data = {
            "gazetteer": "geonames",
            "documents": [
                {
                    "text": "Paris and London",
                    "toponyms": [
                        {"start": 0, "end": 5, "loc_id": "123"},  # Geocoded
                        {"start": 10, "end": 16, "loc_id": ""},  # Not geocoded (empty)
                    ],
                }
            ],
        }

        with (
            patch("geoparser.project.project.json.load", return_value=json_data),
            patch.object(Project, "create_references"),
            patch.object(Project, "create_referents") as mock_create_ref,
        ):
            project = Project("TestProject")

            # Act
            project.load_annotations("test.json", tag="test_tag")

            # Assert
            # Should create referents with None for non-geocoded
            call_args = mock_create_ref.call_args[0]
            referents = call_args[2]  # Third argument is referents
            assert referents == [[("geonames", "123"), None]]


@pytest.mark.unit
class TestNormalizeDocumentIds:
    """Accepting IDs as UUIDs, strings, or sequences of either."""

    def test_wraps_a_single_uuid(self):
        """One UUID becomes a one-element list."""
        # Arrange
        one = uuid.uuid4()

        # Act & Assert
        assert Project._normalize_document_ids(one) == [one]

    def test_wraps_and_parses_a_single_string(self):
        """One string ID is parsed into a UUID."""
        # Arrange
        one = uuid.uuid4()

        # Act & Assert
        assert Project._normalize_document_ids(str(one)) == [one]

    def test_keeps_the_order_of_a_sequence(self):
        """A sequence comes back in the order it was given."""
        # Arrange
        first, second = uuid.uuid4(), uuid.uuid4()

        # Act
        normalized = Project._normalize_document_ids([str(second), first])

        # Assert
        assert normalized == [second, first]

    def test_rejects_a_value_that_is_not_an_id(self):
        """Anything unparseable is reported rather than silently skipped."""
        # Act & Assert
        with pytest.raises(ValueError):
            Project._normalize_document_ids(["not-a-uuid"])


@pytest.mark.unit
class TestCreateDocuments:
    """Turning texts into document rows."""

    @staticmethod
    def _project() -> Project:
        """A Project with its database interactions stubbed out."""
        project = Project.__new__(Project)
        project.name = "demo"
        project.id = uuid.uuid4()
        return project

    def test_rejects_a_bare_string(self):
        """
        A single string would be iterated character by character.

        The guard exists so that create_documents("hello") does not silently
        create five documents.
        """
        # Arrange
        project = self._project()

        # Act & Assert
        with pytest.raises(TypeError, match="create_documents"):
            project.create_documents("hello")

    def test_creates_one_document_per_text_under_this_project(self):
        """Each text becomes its own row, carrying this project's id."""
        # Arrange
        project = self._project()
        created = []

        def _create_many(session, document_creates):
            created.extend(document_creates)
            return [SimpleNamespace(id=uuid.uuid4()) for _ in document_creates]

        with (
            patch("geoparser.project.project.get_session"),
            patch(
                "geoparser.project.project.DocumentRepository.create_many",
                side_effect=_create_many,
            ),
        ):
            # Act
            project.create_documents(["first", "second"])

        # Assert
        assert [d.text for d in created] == ["first", "second"]
        assert [d.project_id for d in created] == [project.id, project.id]

    def test_returns_the_new_ids_in_input_order(self):
        """The returned IDs line up with the texts that were passed in."""
        # Arrange
        project = self._project()
        ids = [uuid.uuid4(), uuid.uuid4()]

        with (
            patch("geoparser.project.project.get_session"),
            patch(
                "geoparser.project.project.DocumentRepository.create_many",
                return_value=list(ids),
            ),
        ):
            # Act
            returned = project.create_documents(["first", "second"])

        # Assert
        assert returned == ids


@pytest.mark.unit
class TestEnsureProjectRecord:
    """Looking up or creating the project row."""

    def test_reuses_an_existing_project_of_the_same_name(self):
        """An existing project is looked up by name and its id reused."""
        # Arrange
        project = Project.__new__(Project)
        existing_id = uuid.uuid4()
        get_by_name = Mock(return_value=SimpleNamespace(id=existing_id))

        with (
            patch("geoparser.project.project.get_session"),
            patch(
                "geoparser.project.project.ProjectRepository.get_by_name", get_by_name
            ),
            patch("geoparser.project.project.ProjectRepository.create") as create,
        ):
            # Act
            found = project._ensure_project_record("demo")

        # Assert
        assert found == existing_id
        assert get_by_name.call_args.args[1] == "demo"
        create.assert_not_called()

    def test_creates_a_project_when_the_name_is_new(self):
        """A missing project is created with the requested name."""
        # Arrange
        project = Project.__new__(Project)
        new_id = uuid.uuid4()

        with (
            patch("geoparser.project.project.get_session"),
            patch(
                "geoparser.project.project.ProjectRepository.get_by_name",
                return_value=None,
            ),
            patch(
                "geoparser.project.project.ProjectRepository.create",
                return_value=SimpleNamespace(id=new_id),
            ) as create,
        ):
            # Act
            created = project._ensure_project_record("fresh")

        # Assert
        assert created == new_id
        assert create.call_args.args[1].name == "fresh"


@pytest.mark.unit
class TestLoadAnnotations:
    """Importing an annotator export into the project."""

    @staticmethod
    def _export(tmp_path, documents, gazetteer="geonames"):
        """Write an annotator-format JSON file and return its path."""
        path = tmp_path / "annotations.json"
        path.write_text(
            json.dumps({"gazetteer": gazetteer, "documents": documents}),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _load(project, path, **kwargs):
        """Run load_annotations, capturing the two registration calls."""
        with (
            patch.object(project, "create_documents") as create_documents,
            patch.object(project, "create_references") as create_references,
            patch.object(project, "create_referents") as create_referents,
        ):
            project.load_annotations(str(path), "annotator_a", **kwargs)
        return create_documents, create_references, create_referents

    def test_reads_the_export_as_utf8(self, tmp_path, monkeypatch):
        """Exports are UTF-8 regardless of the platform's default encoding."""
        import builtins

        project = Project.__new__(Project)
        path = self._export(tmp_path, [{"text": "Zürich", "toponyms": []}])
        real_open = builtins.open
        encodings = []

        def recording_open(file, *args, **kwargs):
            if str(file) == str(path):
                encodings.append(kwargs.get("encoding"))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", recording_open)

        self._load(project, path)

        assert encodings == ["utf-8"]

    def test_registers_every_toponym_as_a_reference(self, tmp_path):
        """Spans come through per document, in file order."""
        # Arrange
        project = Project.__new__(Project)
        path = self._export(
            tmp_path,
            [
                {
                    "text": "Paris and Berlin",
                    "toponyms": [
                        {"start": 0, "end": 5, "loc_id": "1"},
                        {"start": 10, "end": 16, "loc_id": "2"},
                    ],
                },
                {"text": "Rome", "toponyms": [{"start": 0, "end": 4, "loc_id": "3"}]},
            ],
        )

        # Act
        _, create_references, _ = self._load(project, path)

        # Assert
        texts, references, tag = create_references.call_args.args
        assert texts == ["Paris and Berlin", "Rome"]
        assert references == [[(0, 5), (10, 16)], [(0, 4)]]
        assert tag == "annotator_a"

    def test_pairs_geocoded_toponyms_with_the_files_gazetteer(self, tmp_path):
        """Referents name the gazetteer the export declares."""
        # Arrange
        project = Project.__new__(Project)
        path = self._export(
            tmp_path,
            [{"text": "Paris", "toponyms": [{"start": 0, "end": 5, "loc_id": "7"}]}],
            gazetteer="swissnames3d",
        )

        # Act
        _, _, create_referents = self._load(project, path)

        # Assert
        texts, references, referents, tag = create_referents.call_args.args
        assert texts == ["Paris"]
        assert references == [[(0, 5)]]
        assert referents == [[("swissnames3d", "7")]]
        assert tag == "annotator_a"

    def test_keeps_ungeocoded_toponyms_as_references_without_referents(self, tmp_path):
        """
        A toponym left ungeocoded still counts as a reference.

        The two lists stay aligned by carrying None in the referent slot, so
        the resolver skips it rather than the reference disappearing.
        """
        # Arrange
        project = Project.__new__(Project)
        path = self._export(
            tmp_path,
            [
                {
                    "text": "Paris and Nowhere",
                    "toponyms": [
                        {"start": 0, "end": 5, "loc_id": "1"},
                        {"start": 10, "end": 17, "loc_id": ""},
                    ],
                }
            ],
        )

        # Act
        _, create_references, create_referents = self._load(project, path)

        # Assert
        assert create_references.call_args.args[1] == [[(0, 5), (10, 17)]]
        assert create_referents.call_args.args[2] == [[("geonames", "1"), None]]

    def test_does_not_create_documents_by_default(self, tmp_path):
        """Annotations attach to documents that already exist."""
        # Arrange
        project = Project.__new__(Project)
        path = self._export(
            tmp_path,
            [{"text": "Paris", "toponyms": [{"start": 0, "end": 5, "loc_id": "1"}]}],
        )

        # Act
        create_documents, _, _ = self._load(project, path)

        # Assert
        create_documents.assert_not_called()

    def test_creates_documents_from_the_export_when_asked(self, tmp_path):
        """With create_documents=True the texts are inserted first."""
        # Arrange
        project = Project.__new__(Project)
        path = self._export(
            tmp_path,
            [{"text": "Paris", "toponyms": [{"start": 0, "end": 5, "loc_id": "1"}]}],
        )

        # Act
        create_documents, _, _ = self._load(project, path, create_documents=True)

        # Assert
        create_documents.assert_called_once_with(["Paris"])

    def test_reads_utf8_annotations_when_default_encoding_is_cp1252(
        self, tmp_path, monkeypatch
    ):
        """An export remains intact on systems whose text default is not UTF-8."""
        # Arrange
        path = tmp_path / "annotations.json"
        path.write_bytes(
            json.dumps(
                {
                    "gazetteer": "geonames",
                    "documents": [{"text": "Zürich", "toponyms": []}],
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )
        real_open = builtins.open

        def cp1252_default_open(file, *args, **kwargs):
            if Path(file).resolve() == path.resolve() and "encoding" not in kwargs:
                kwargs["encoding"] = "cp1252"
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", cp1252_default_open)
        project = Project.__new__(Project)

        # Act
        create_documents, _, _ = self._load(project, path, create_documents=True)

        # Assert
        create_documents.assert_called_once_with(["Zürich"])


@pytest.mark.unit
class TestRunModules:
    """Running a module and recording it against a tag."""

    @staticmethod
    def _project() -> Project:
        """A Project with the database untouched."""
        project = Project.__new__(Project)
        project.id = uuid.uuid4()
        project.context = Mock()
        return project

    def test_records_the_recognizer_against_the_default_tag(self):
        """Omitting the tag files the run under "latest"."""
        # Arrange
        project = self._project()
        recognizer = Mock(id="rec-1")

        with (
            patch.object(project, "get_documents", return_value=[]),
            patch("geoparser.project.project.RecognitionService"),
        ):
            # Act
            project.run_recognizer(recognizer)

        # Assert
        cast(Mock, project.context).update_recognizer_context.assert_called_once_with(
            "latest", "rec-1"
        )

    def test_records_the_recognizer_against_an_explicit_tag(self):
        """A caller-supplied tag is used verbatim."""
        # Arrange
        project = self._project()
        recognizer = Mock(id="rec-1")

        with (
            patch.object(project, "get_documents", return_value=[]),
            patch("geoparser.project.project.RecognitionService"),
        ):
            # Act
            project.run_recognizer(recognizer, tag="experiment")

        # Assert
        cast(Mock, project.context).update_recognizer_context.assert_called_once_with(
            "experiment", "rec-1"
        )

    def test_runs_the_recognizer_over_the_projects_documents(self):
        """The service is handed the documents this project holds."""
        # Arrange
        project = self._project()
        documents = [Mock(), Mock()]

        with (
            patch.object(project, "get_documents", return_value=documents),
            patch("geoparser.project.project.RecognitionService") as service,
        ):
            # Act
            project.run_recognizer(Mock(id="rec-1"))

        # Assert
        service.return_value.predict.assert_called_once_with(documents)

    def test_records_the_resolver_against_the_default_tag(self):
        """The resolver path files under "latest" too."""
        # Arrange
        project = self._project()
        resolver = Mock(id="res-1")

        with (
            patch.object(project, "get_documents", return_value=[]),
            patch("geoparser.project.project.ResolutionService"),
        ):
            # Act
            project.run_resolver(resolver)

        # Assert
        cast(Mock, project.context).update_resolver_context.assert_called_once_with(
            "latest", "res-1"
        )

    def test_runs_the_resolver_over_the_projects_documents(self):
        """The resolution service sees the same documents."""
        # Arrange
        project = self._project()
        documents = [Mock()]

        with (
            patch.object(project, "get_documents", return_value=documents),
            patch("geoparser.project.project.ResolutionService") as service,
        ):
            # Act
            project.run_resolver(Mock(id="res-1"), tag="experiment")

        # Assert
        service.return_value.predict.assert_called_once_with(documents)
        cast(Mock, project.context).update_resolver_context.assert_called_once_with(
            "experiment", "res-1"
        )


@pytest.mark.unit
class TestGetDocumentsTag:
    """Which tag's results a retrieval is scoped to."""

    def test_defaults_to_the_latest_tag(self):
        """Both contexts are looked up for "latest" unless told otherwise."""
        # Arrange
        project = Project.__new__(Project)
        project.id = uuid.uuid4()
        project.context = Mock()
        project.context.get_recognizer_context.return_value = None
        project.context.get_resolver_context.return_value = None

        with (
            patch("geoparser.project.project.get_session"),
            patch(
                "geoparser.project.project.DocumentRepository.get_by_project",
                return_value=[],
            ),
        ):
            # Act
            project.get_documents()

        # Assert
        project.context.get_recognizer_context.assert_called_once_with("latest")
        project.context.get_resolver_context.assert_called_once_with("latest")

    def test_uses_an_explicit_tag_for_both_contexts(self):
        """A named tag scopes the recognizer and resolver together."""
        # Arrange
        project = Project.__new__(Project)
        project.id = uuid.uuid4()
        project.context = Mock()
        project.context.get_recognizer_context.return_value = None
        project.context.get_resolver_context.return_value = None

        with (
            patch("geoparser.project.project.get_session"),
            patch(
                "geoparser.project.project.DocumentRepository.get_by_project",
                return_value=[],
            ),
        ):
            # Act
            project.get_documents(tag="experiment")

        # Assert
        project.context.get_recognizer_context.assert_called_once_with("experiment")
        project.context.get_resolver_context.assert_called_once_with("experiment")


@pytest.fixture
def project() -> Project:
    """A project backed by the in-memory test database."""
    return Project("persistence")


@pytest.mark.unit
class TestProjectRecord:
    """Creating and reusing the project's own row."""

    def test_creates_a_record_and_reuses_it_by_name(self, project):
        """Opening the same name twice yields the same project id."""
        # Act
        reopened = Project("persistence")

        # Assert
        assert reopened.id == project.id

    def test_distinct_names_get_distinct_records(self, project):
        """A different name is a different project."""
        # Act
        other = Project("something-else")

        # Assert
        assert other.id != project.id

    def test_the_context_is_scoped_to_this_project(self, project):
        """The context reads and writes rows for this project only."""
        # Assert
        assert project.context.project_id == project.id


@pytest.mark.unit
class TestDocumentRoundTrip:
    """Writing documents and reading them back."""

    def test_created_documents_are_readable_in_order(self, project):
        """Texts go in, documents come back in the same order."""
        # Act
        ids = project.create_documents(["first", "second", "third"])
        documents = project.get_documents()

        # Assert
        assert len(ids) == 3
        assert [document.text for document in documents] == [
            "first",
            "second",
            "third",
        ]

    def test_documents_belong_to_the_project_that_created_them(self, project):
        """Another project does not see these documents."""
        # Arrange
        project.create_documents(["mine"])
        other = Project("other")

        # Act & Assert
        assert other.get_documents() == []

    def test_requested_ids_come_back_in_the_order_asked_for(self, project):
        """Retrieval order follows the request, not the insertion order."""
        # Arrange
        first, second, third = project.create_documents(["a", "b", "c"])

        # Act
        documents = project.get_documents([third, first, second])

        # Assert
        assert [document.text for document in documents] == ["c", "a", "b"]

    def test_an_unknown_id_is_reported_by_value(self, project):
        """The error names the id that was not found."""
        # Arrange
        project.create_documents(["a"])
        unknown = uuid.uuid4()

        # Act & Assert
        with pytest.raises(ValueError, match=str(unknown)):
            project.get_documents([unknown])

    def test_several_unknown_ids_are_listed_comma_separated(self, project):
        """Every missing id is listed, so a caller can fix them in one go."""
        # Arrange
        first, second = uuid.uuid4(), uuid.uuid4()

        # Act & Assert
        with pytest.raises(ValueError, match=f"{first}, {second}"):
            project.get_documents([first, second])


@pytest.mark.unit
class TestDelete:
    """Removing a project."""

    def test_delete_removes_the_project_and_its_documents(self, project):
        """After deleting, reopening the name starts from an empty project."""
        # Arrange
        project.create_documents(["a", "b"])

        # Act
        project.delete()
        reopened = Project("persistence")

        # Assert
        assert reopened.id != project.id
        assert reopened.get_documents() == []


@pytest.mark.unit
class TestManualAnnotationTags:
    """The tag manual annotations are filed under."""

    def test_create_references_files_them_under_the_given_tag(self, project):
        """The tag reaches run_recognizer rather than falling back to latest."""
        # Arrange & Act
        with patch.object(project, "run_recognizer") as run:
            project.create_references(["Paris"], [[(0, 5)]], tag="gold")

        # Assert
        assert run.call_args.kwargs["tag"] == "gold"

    def test_create_referents_files_them_under_the_given_tag(self, project):
        """The tag reaches run_resolver rather than falling back to latest."""
        # Arrange & Act
        with patch.object(project, "run_resolver") as run:
            project.create_referents(
                ["Paris"], [[(0, 5)]], [[("geonames", "1")]], tag="gold"
            )

        # Assert
        assert run.call_args.kwargs["tag"] == "gold"
