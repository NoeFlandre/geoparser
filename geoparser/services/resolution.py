import typing as t
import uuid

from sqlalchemy import insert
from sqlmodel import Session

from geoparser.db.crud import (
    ReferentRepository,  # noqa: F401 - retained as a patch/extension seam
    ResolutionRepository,
    ResolverRepository,
)
from geoparser.db.db import get_session
from geoparser.db.models import (
    Referent,
    ReferentCreate,
    Resolution,
    ResolutionCreate,
    ResolverCreate,
)
from geoparser.gazetteer.gazetteer import Gazetteer

if t.TYPE_CHECKING:
    from geoparser.db.models import Document, Reference
    from geoparser.modules.resolvers.base import Resolver


class ResolutionService:
    """
    Service layer that handles all database operations for reference resolution.

    This service acts as a bridge between resolver modules (which are DB-agnostic)
    and the database layer.
    """

    def __init__(self, resolver: "Resolver"):
        """
        Initialize the resolution service.

        Args:
            resolver: The resolver module to use for predictions
        """
        self.resolver = resolver
        self._gazetteers: dict[str, Gazetteer] = {}

    def _ensure_resolver_record(self, resolver: "Resolver") -> str:
        """
        Ensure a resolver record exists in the database.

        Creates a new resolver record if it doesn't already exist.

        Args:
            resolver: The resolver module to ensure exists in the database

        Returns:
            The resolver ID from the database
        """
        with get_session() as session:
            resolver_record = ResolverRepository.get(session, id=resolver.id)
            if resolver_record is None:
                resolver_create = ResolverCreate(
                    id=resolver.id,
                    name=resolver.name,
                    config=resolver.config,
                )
                resolver_record = ResolverRepository.create(session, resolver_create)
            return resolver_record.id

    def predict(self, documents: list["Document"]) -> None:
        """
        Run the resolver on all references from the provided documents and store results in the database.

        Args:
            documents: List of Document objects containing references to process
        """
        # Ensure resolver record exists in database and get the ID
        resolver_id = self._ensure_resolver_record(self.resolver)

        if not documents:
            return

        with get_session() as session:
            try:
                texts, reference_boundaries, reference_objects = (
                    self._collect_unprocessed(session, documents, resolver_id)
                )

                # Only call predict if there are documents with unprocessed references
                if not texts:
                    return

                predicted_referents = self.resolver.predict(texts, reference_boundaries)

                # Validate and stage every row before one atomic commit.
                self._record_all_referent_predictions(
                    session, reference_objects, predicted_referents, resolver_id
                )
                session.commit()
            except Exception:
                session.rollback()
                raise

    def _collect_unprocessed(
        self,
        session: Session,
        documents: list["Document"],
        resolver_id: str,
    ) -> tuple[list[str], list[list[tuple[int, int]]], list[list["Reference"]]]:
        """
        Gather the documents that still have references this resolver has not seen.

        Args:
            session: Database session
            documents: Documents to inspect
            resolver_id: Resolver whose prior work should be skipped

        Returns:
            Parallel lists of document texts, reference spans, and the
            reference objects those spans came from
        """
        texts: list[str] = []
        boundaries: list[list[tuple[int, int]]] = []
        objects: list[list[Reference]] = []
        processed_ids = self._processed_reference_ids(session, documents, resolver_id)

        for doc in documents:
            unprocessed = self._unprocessed(doc, processed_ids)
            if unprocessed:
                texts.append(doc.text)
                boundaries.append([(ref.start, ref.end) for ref in unprocessed])
                objects.append(unprocessed)

        return texts, boundaries, objects

    @staticmethod
    def _unprocessed(doc: "Document", processed_ids: set[uuid.UUID]) -> list:
        """Return the document's references not in ``processed_ids``."""
        return [
            reference
            for reference in doc.references
            if reference.id not in processed_ids
        ]

    @staticmethod
    def _processed_reference_ids(
        session: Session, documents: list["Document"], resolver_id: str
    ) -> set[uuid.UUID]:
        """Return the IDs of the documents' references this resolver has seen."""
        return ResolutionRepository.get_processed_reference_ids(
            session,
            [reference.id for doc in documents for reference in doc.references],
            resolver_id,
        )

    def fit(self, documents: list["Document"], **kwargs) -> None:
        """
        Train the resolver using the provided documents.

        This method prepares training data from documents that have reference and referent
        annotations and calls the resolver's fit method if it exists.

        Args:
            documents: List of Document objects with referent annotations for training
            **kwargs: Additional training parameters (e.g., output_path, epochs, batch_size)

        Raises:
            ValueError: If the resolver does not implement a fit method
        """
        # Resolvers are not required to be trainable, so `fit` is looked up
        # rather than declared on the base class.
        fit: t.Callable[..., None] | None = getattr(self.resolver, "fit", None)
        if fit is None:
            raise ValueError(
                f"Resolver '{self.resolver.name}' does not implement a fit method"
            )

        # Extract texts, references, and referents from documents
        texts = []
        references = []
        referents = []

        for doc in documents:
            doc_references, doc_referents = self._annotated_pairs(doc)

            # Only include documents that have referent annotations
            if doc_references:
                texts.append(doc.text)
                references.append(doc_references)
                referents.append(doc_referents)

        # Call the resolver's fit method with the prepared data
        fit(texts, references, referents, **kwargs)

    @staticmethod
    def _annotated_pairs(
        doc: "Document",
    ) -> tuple[list[tuple[int, int]], list[tuple[str, str]]]:
        """
        One document's resolved toponyms, as parallel spans and referents.

        ``doc.toponyms`` is already filtered by the recognizer context, and
        ``ref.location`` by the resolver context, so this keeps only the
        references that carry a referent from the resolver being trained.

        Args:
            doc: The document to read annotations from

        Returns:
            The reference spans and their (gazetteer, identifier) referents
        """
        annotated = [(ref, ref.location) for ref in doc.toponyms if ref.location]
        spans = [(ref.start, ref.end) for ref, _ in annotated]
        pairs = [
            (location.gazetteer_name, location.identifier) for _, location in annotated
        ]
        return spans, pairs

    def _record_all_referent_predictions(
        self,
        session: Session,
        reference_groups: list[list["Reference"]],
        predicted_groups: list[list[tuple[str, str] | None]],
        resolver_id: str,
    ) -> None:
        """Stage all document resolution rows with one database write."""
        self._record_referent_prediction_groups(
            session, reference_groups, predicted_groups, resolver_id
        )

    def _record_referent_prediction_groups(
        self,
        session: Session,
        reference_groups: list[list["Reference"]],
        predicted_groups: list[list[tuple[str, str] | None]],
        resolver_id: str,
    ) -> None:
        """Validate and stage grouped referent/resolution records."""
        referent_rows: list[dict[str, t.Any]] = []
        resolution_rows: list[dict[str, t.Any]] = []
        # Process each reference with its predicted referent; see above on
        # why a short prediction list is tolerated rather than rejected.
        # pragma: no mutate start - strict=False is the default, so a mutant
        # that drops it or passes another falsy value pairs them identically.
        group_pairs = zip(reference_groups, predicted_groups, strict=False)
        # pragma: no mutate end
        for references, predictions in group_pairs:
            # pragma: no mutate start - as above, strict=False is the default.
            pairs = zip(references, predictions, strict=False)
            # pragma: no mutate end
            for reference, referent in pairs:
                # Skip references where predictions are not available
                # (None indicates the resolver couldn't process this reference)
                if referent is not None:
                    referent_row, resolution_row = self._reference_records(
                        reference, referent, resolver_id
                    )
                    if referent_row is not None:
                        referent_rows.append(referent_row)
                    if resolution_row is not None:
                        resolution_rows.append(resolution_row)

        if referent_rows:
            table = Referent.__table__  # ty: ignore[unresolved-attribute]
            session.execute(insert(table), referent_rows)  # ty: ignore[deprecated]
        if resolution_rows:
            table = Resolution.__table__  # ty: ignore[unresolved-attribute]
            session.execute(insert(table), resolution_rows)  # ty: ignore[deprecated]

    def _reference_records(
        self,
        reference: "Reference",
        referent: tuple[str, str],
        resolver_id: str,
    ) -> tuple[dict[str, t.Any] | None, dict[str, t.Any] | None]:
        """
        Build one reference's referent record and its processed marker.

        Args:
            reference: The reference that was resolved
            referent: The (gazetteer name, identifier) it was resolved to
            resolver_id: ID of the resolver that made the prediction

        Returns:
            The referent and resolution mappings
        """
        gazetteer_name, identifier = referent
        referent_record = self._create_referent_record(
            reference.id, gazetteer_name, identifier, resolver_id
        )
        resolution_record = self._create_resolution_record(reference.id, resolver_id)
        return referent_record, resolution_record

    def _create_referent_record(
        self,
        reference_id: uuid.UUID,
        gazetteer_name: str,
        identifier: str,
        resolver_id: str,
    ) -> dict[str, t.Any]:
        """
        Create a referent record with the resolver ID.

        Args:
            reference_id: ID of the reference
            gazetteer_name: Name of the gazetteer
            identifier: Identifier value in the gazetteer
            resolver_id: ID of the resolver

        Raises:
            ValueError: If the feature does not exist in the gazetteer
        """
        # Validate that the feature exists in the installed gazetteer
        if gazetteer_name not in self._gazetteers:
            self._gazetteers[gazetteer_name] = Gazetteer(gazetteer_name)
        feature = self._gazetteers[gazetteer_name].find(identifier)
        if feature is None:
            raise ValueError(
                f"Feature '{identifier}' does not exist in gazetteer '{gazetteer_name}'"
            )

        row = ReferentCreate(
            reference_id=reference_id,
            gazetteer_name=gazetteer_name,
            feature_identifier=feature.identifier,
            resolver_id=resolver_id,
        ).model_dump()
        row["id"] = uuid.uuid4()
        return row

    def _create_resolution_record(
        self, reference_id: uuid.UUID, resolver_id: str
    ) -> dict[str, t.Any]:
        """
        Create a resolution record for a reference processed by a specific resolver.

        Args:
            reference_id: ID of the reference that was processed
            resolver_id: ID of the resolver that processed it
        """
        row = ResolutionCreate(
            reference_id=reference_id,
            resolver_id=resolver_id,
        ).model_dump()
        row["id"] = uuid.uuid4()
        return row
