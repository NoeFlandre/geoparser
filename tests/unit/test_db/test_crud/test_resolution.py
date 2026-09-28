"""
Unit tests for geoparser/db/crud/resolution.py

Tests the ResolutionRepository class with custom query methods.
"""

import pytest
from sqlmodel import Session

from geoparser.db.crud import ResolutionRepository
from geoparser.db.crud.reference import ReferenceRepository
from geoparser.db.models import (
    Document,
    DocumentCreate,
    Project,
    ProjectCreate,
    ReferenceCreate,
    ResolutionCreate,
)


@pytest.mark.unit
class TestResolutionRepositoryGetByReference:
    """Test the get_by_reference method of ResolutionRepository."""

    def test_returns_resolutions_for_reference(
        self,
        test_session: Session,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_by_reference returns all resolutions for a reference."""
        # Arrange
        reference = reference_factory()
        resolver_factory(id="res1")
        resolver_factory(id="res2")

        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=reference.id, resolver_id="res1"),
        )
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=reference.id, resolver_id="res2"),
        )

        # Act
        resolutions = ResolutionRepository.get_by_reference(test_session, reference.id)

        # Assert
        assert len(resolutions) == 2
        resolver_ids = [r.resolver_id for r in resolutions]
        assert "res1" in resolver_ids
        assert "res2" in resolver_ids

    def test_returns_empty_list_for_reference_without_resolutions(
        self, test_session: Session, reference_factory
    ):
        """Test that get_by_reference returns empty list for reference without resolutions."""
        # Arrange
        reference = reference_factory()

        # Act
        resolutions = ResolutionRepository.get_by_reference(test_session, reference.id)

        # Assert
        assert resolutions == []


@pytest.mark.unit
class TestResolutionRepositoryGetByResolver:
    """Test the get_by_resolver method of ResolutionRepository."""

    def test_returns_resolutions_for_resolver(
        self,
        test_session: Session,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_by_resolver returns all resolutions for a resolver."""
        # Arrange
        resolver_factory(id="test_res")
        ref1 = reference_factory()
        ref2 = reference_factory()

        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=ref1.id, resolver_id="test_res"),
        )
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=ref2.id, resolver_id="test_res"),
        )

        # Act
        resolutions = ResolutionRepository.get_by_resolver(test_session, "test_res")

        # Assert
        assert len(resolutions) == 2
        reference_ids = [r.reference_id for r in resolutions]
        assert ref1.id in reference_ids
        assert ref2.id in reference_ids

    def test_returns_empty_list_for_resolver_without_resolutions(
        self, test_session: Session, resolver_factory
    ):
        """Test that get_by_resolver returns empty list for resolver without resolutions."""
        # Arrange
        resolver_factory(id="test_res")

        # Act
        resolutions = ResolutionRepository.get_by_resolver(test_session, "test_res")

        # Assert
        assert resolutions == []


@pytest.mark.unit
class TestResolutionRepositoryGetByReferenceAndResolver:
    """Test the get_by_reference_and_resolver method of ResolutionRepository."""

    def test_returns_resolution_for_matching_pair(
        self,
        test_session: Session,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_by_reference_and_resolver returns resolution for matching pair."""
        # Arrange
        reference = reference_factory()
        resolver_factory(id="test_res")

        created_resolution = ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=reference.id, resolver_id="test_res"),
        )

        # Act
        resolution = ResolutionRepository.get_by_reference_and_resolver(
            test_session, reference.id, "test_res"
        )

        # Assert
        assert resolution is not None
        assert resolution.id == created_resolution.id
        assert resolution.reference_id == reference.id
        assert resolution.resolver_id == "test_res"

    def test_returns_none_for_non_matching_pair(
        self,
        test_session: Session,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_by_reference_and_resolver returns None for non-matching pair."""
        # Arrange
        reference = reference_factory()
        resolver_factory(id="test_res")

        # Act
        resolution = ResolutionRepository.get_by_reference_and_resolver(
            test_session, reference.id, "test_res"
        )

        # Assert
        assert resolution is None


@pytest.mark.unit
class TestResolutionRepositoryGetUnprocessedReferences:
    """Test the get_unprocessed_references method of ResolutionRepository."""

    def test_returns_references_not_processed_by_resolver(
        self,
        test_session: Session,
        project_factory,
        document_factory,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_unprocessed_references returns references not yet processed."""
        # Arrange
        project = project_factory()
        resolver_factory(id="test_res")
        document = document_factory(project_id=project.id)

        # Create three references
        ref1 = reference_factory(document_id=document.id)
        ref2 = reference_factory(document_id=document.id)
        ref3 = reference_factory(document_id=document.id)

        # Mark ref1 as processed
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=ref1.id, resolver_id="test_res"),
        )

        # Act
        unprocessed = ResolutionRepository.get_unprocessed_references(
            test_session, project.id, "test_res"
        )

        # Assert
        assert len(unprocessed) == 2
        unprocessed_ids = [r.id for r in unprocessed]
        assert ref1.id not in unprocessed_ids
        assert ref2.id in unprocessed_ids
        assert ref3.id in unprocessed_ids

    def test_returns_all_references_when_none_processed(
        self,
        test_session: Session,
        project_factory,
        document_factory,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_unprocessed_references returns all references when none processed."""
        # Arrange
        project = project_factory()
        resolver_factory(id="test_res")
        document = document_factory(project_id=project.id)

        reference_factory(document_id=document.id)
        reference_factory(document_id=document.id)

        # Act
        unprocessed = ResolutionRepository.get_unprocessed_references(
            test_session, project.id, "test_res"
        )

        # Assert
        assert len(unprocessed) == 2

    def test_returns_empty_list_when_all_processed(
        self,
        test_session: Session,
        project_factory,
        document_factory,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_unprocessed_references returns empty list when all processed."""
        # Arrange
        project = project_factory()
        resolver_factory(id="test_res")
        document = document_factory(project_id=project.id)

        ref1 = reference_factory(document_id=document.id)
        ref2 = reference_factory(document_id=document.id)

        # Mark both as processed
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=ref1.id, resolver_id="test_res"),
        )
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=ref2.id, resolver_id="test_res"),
        )

        # Act
        unprocessed = ResolutionRepository.get_unprocessed_references(
            test_session, project.id, "test_res"
        )

        # Assert
        assert unprocessed == []

    def test_filters_by_project(
        self,
        test_session: Session,
        project_factory,
        document_factory,
        reference_factory,
        resolver_factory,
    ):
        """Test that get_unprocessed_references only returns references from specified project."""
        # Arrange
        project1 = project_factory()
        project2 = project_factory()
        resolver_factory(id="test_res")

        # References in project1
        doc1_proj1 = document_factory(project_id=project1.id)
        ref1_proj1 = reference_factory(document_id=doc1_proj1.id)

        # References in project2
        doc1_proj2 = document_factory(project_id=project2.id)
        reference_factory(document_id=doc1_proj2.id)

        # Act - Get unprocessed from project1
        unprocessed = ResolutionRepository.get_unprocessed_references(
            test_session, project1.id, "test_res"
        )

        # Assert - Should only contain ref from project1
        assert len(unprocessed) == 1
        assert unprocessed[0].id == ref1_proj1.id


@pytest.mark.unit
class TestResolutionRepositoryProcessedReferenceIds:
    """Test the set-based processed-reference lookup."""

    def test_returns_processed_ids_for_requested_references(
        self,
        test_session: Session,
        reference_factory,
        resolver_factory,
    ):
        """The lookup scopes one resolver to the requested reference IDs."""
        # Arrange
        resolver_factory(id="test_res")
        processed = reference_factory()
        unprocessed = reference_factory()
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=processed.id, resolver_id="test_res"),
        )

        # Act
        ids = ResolutionRepository.get_processed_reference_ids(
            test_session, [unprocessed.id, processed.id], "test_res"
        )

        # Assert
        assert ids == {processed.id}

    def test_returns_empty_set_for_no_requested_references(self, test_session: Session):
        """An empty batch does not issue an invalid empty IN query."""
        assert (
            ResolutionRepository.get_processed_reference_ids(
                test_session, [], "test_res"
            )
            == set()
        )


@pytest.mark.unit
class TestGetProcessedReferenceIdsScope:
    """Both filters of the processed-ID lookup matter."""

    def test_ignores_another_resolvers_work(
        self, test_session: Session, reference_factory, resolver_factory
    ):
        """A reference another resolver processed is still to do."""
        resolver_factory(id="mine")
        resolver_factory(id="other")
        reference = reference_factory()
        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=reference.id, resolver_id="other"),
        )

        assert (
            ResolutionRepository.get_processed_reference_ids(
                test_session, [reference.id], "mine"
            )
            == set()
        )

    def test_ignores_references_that_were_not_asked_about(
        self, test_session: Session, reference_factory, resolver_factory
    ):
        """Only the requested IDs are reported, even if others were processed."""
        resolver_factory(id="mine")
        asked, other = reference_factory(), reference_factory()
        for reference in (asked, other):
            ResolutionRepository.create(
                test_session,
                ResolutionCreate(reference_id=reference.id, resolver_id="mine"),
            )

        assert ResolutionRepository.get_processed_reference_ids(
            test_session, [asked.id], "mine"
        ) == {asked.id}


@pytest.mark.unit
class TestResolutionByReferenceAndResolver:
    """Which reference a resolver has already processed."""

    @pytest.fixture(autouse=True)
    def resolutions(self, session: Session, world):
        """Every combination of the two references and two resolvers."""
        for reference in world["references"]:
            for resolver in world["resolvers"]:
                ResolutionRepository.create(
                    session,
                    ResolutionCreate(
                        reference_id=reference.id, resolver_id=resolver.id
                    ),
                )

    def test_matches_on_both_the_reference_and_the_resolver(
        self, session: Session, world
    ):
        """Exactly the one row for that pair comes back."""
        found = ResolutionRepository.get_by_reference_and_resolver(
            session, world["references"][0].id, world["resolvers"][1].id
        )

        assert found is not None
        assert (found.reference_id, found.resolver_id) == (
            world["references"][0].id,
            world["resolvers"][1].id,
        )

    def test_ignores_the_same_resolver_on_another_reference(
        self, session: Session, world
    ):
        """The other reference's row for the same resolver is excluded."""
        found = ResolutionRepository.get_by_reference_and_resolver(
            session, world["references"][1].id, world["resolvers"][0].id
        )

        assert found is not None
        assert found.reference_id == world["references"][1].id


@pytest.mark.unit
class TestUnprocessedReferences:
    """Which references a resolver has still to see."""

    def test_excludes_references_this_resolver_already_resolved(
        self, session: Session, world
    ):
        ResolutionRepository.create(
            session,
            ResolutionCreate(
                reference_id=world["references"][0].id,
                resolver_id=world["resolvers"][0].id,
            ),
        )
        project_id = world["documents"][0].project_id

        pending = ResolutionRepository.get_unprocessed_references(
            session, project_id, world["resolvers"][0].id
        )

        assert [row.id for row in pending] == [world["references"][1].id]

    def test_another_resolvers_work_does_not_count(self, session: Session, world):
        ResolutionRepository.create(
            session,
            ResolutionCreate(
                reference_id=world["references"][0].id,
                resolver_id=world["resolvers"][1].id,
            ),
        )
        project_id = world["documents"][0].project_id

        pending = ResolutionRepository.get_unprocessed_references(
            session, project_id, world["resolvers"][0].id
        )

        assert {row.id for row in pending} == {
            world["references"][0].id,
            world["references"][1].id,
        }

    def test_references_in_another_project_are_not_returned(
        self, session: Session, world, add_row
    ):
        other_project = add_row(Project(**ProjectCreate(name="other").model_dump()))
        other_document = add_row(
            Document(
                **DocumentCreate(
                    text="Vienna", project_id=other_project.id
                ).model_dump()
            )
        )
        ReferenceRepository.create(
            session,
            ReferenceCreate(
                start=0,
                end=6,
                document_id=other_document.id,
                recognizer_id=world["recognizers"][0].id,
            ),
        )

        pending = ResolutionRepository.get_unprocessed_references(
            session, other_project.id, world["resolvers"][0].id
        )

        assert [row.document_id for row in pending] == [other_document.id]
