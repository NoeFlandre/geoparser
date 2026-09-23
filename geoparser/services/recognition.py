import typing as t
import uuid

from sqlmodel import Session

from geoparser.db.crud import (
    RecognitionRepository,
    RecognizerRepository,
)
from geoparser.db.db import get_session
from geoparser.db.models import (
    Document,
    Recognition,
    RecognizerCreate,
    Reference,
)

if t.TYPE_CHECKING:
    from geoparser.modules.recognizers.base import Recognizer


class RecognitionService:
    """
    Service layer that handles all database operations for reference recognition.

    This service acts as a bridge between recognizer modules (which are DB-agnostic)
    and the database layer.
    """

    def __init__(self, recognizer: "Recognizer"):
        """
        Initialize the recognition service.

        Args:
            recognizer: The recognizer module to use for predictions
        """
        self.recognizer = recognizer

    def _ensure_recognizer_record(self, recognizer: "Recognizer") -> str:
        """
        Ensure a recognizer record exists in the database.

        Creates a new recognizer record if it doesn't already exist.

        Args:
            recognizer: The recognizer module to ensure exists in the database

        Returns:
            The recognizer ID from the database
        """
        with get_session() as session:
            recognizer_record = RecognizerRepository.get(session, id=recognizer.id)
            if recognizer_record is None:
                recognizer_create = RecognizerCreate(
                    id=recognizer.id,
                    name=recognizer.name,
                    config=recognizer.config,
                )
                recognizer_record = RecognizerRepository.create(
                    session, recognizer_create
                )
            return recognizer_record.id

    def predict(self, documents: list["Document"]) -> None:
        """
        Run the recognizer on the provided documents and store results in the database.

        Args:
            documents: List of Document objects to process
        """
        # Ensure recognizer record exists in database and get the ID
        recognizer_id = self._ensure_recognizer_record(self.recognizer)

        if not documents:
            return

        with get_session() as session:
            try:
                # Filter out documents that have already been processed by this recognizer
                unprocessed_documents = self._filter_unprocessed_documents(
                    session, documents, recognizer_id
                )
                if unprocessed_documents:
                    self._predict_and_record(
                        session, unprocessed_documents, recognizer_id
                    )
                    session.commit()
            except Exception:
                session.rollback()
                raise

    def _predict_and_record(
        self,
        session: Session,
        documents: list["Document"],
        recognizer_id: str,
    ) -> None:
        """
        Predict references for documents not yet seen and stage the records.

        Args:
            session: Database session the records are staged in
            documents: Documents this recognizer has not processed
            recognizer_id: ID of the recognizer making the predictions
        """
        predicted_references = self.recognizer.predict(
            [document.text for document in documents]
        )
        self._record_reference_predictions(
            session, documents, predicted_references, recognizer_id
        )

    def fit(self, documents: list["Document"], **kwargs) -> None:
        """
        Train the recognizer using the provided documents.

        This method prepares training data from documents that have reference annotations
        and calls the recognizer's fit method if it exists.

        Args:
            documents: List of Document objects with reference annotations for training
            **kwargs: Additional training parameters (e.g., output_path, epochs, batch_size)

        Raises:
            ValueError: If the recognizer does not implement a fit method
        """
        # Recognizers are not required to be trainable, so `fit` is looked up
        # rather than declared on the base class.
        fit: t.Callable[..., None] | None = getattr(self.recognizer, "fit", None)
        if fit is None:
            raise ValueError(
                f"Recognizer '{self.recognizer.name}' does not implement a fit method"
            )

        # Extract texts and references from documents
        texts = []
        references = []

        for doc in documents:
            # Only include documents that have reference annotations
            # doc.toponyms returns references filtered by the recognizer context
            if doc.toponyms:
                texts.append(doc.text)
                references.append([(ref.start, ref.end) for ref in doc.toponyms])

        # Call the recognizer's fit method with the prepared data
        fit(texts, references, **kwargs)

    def _record_reference_predictions(
        self,
        session: Session,
        documents: list["Document"],
        predicted_references: list[list[tuple[int, int]] | None],
        recognizer_id: str,
    ) -> None:
        """
        Process reference predictions and update the database.

        Args:
            session: Database session
            documents: List of document objects
            predicted_references: List where each element is either a list of predicted references
                                 or None for documents where predictions are not available
            recognizer_id: ID of the recognizer that made the predictions
        """
        # Process each document with its predicted references. Recognizers
        # are pluggable, so the prediction count is not enforced here;
        # a short list leaves the trailing documents unprocessed rather
        # than failing the whole batch.
        # pragma: no mutate start - strict=False is the default, so a mutant
        # that drops it or passes another falsy value pairs them identically.
        pairs = zip(documents, predicted_references, strict=False)
        # pragma: no mutate end
        pending: list[Reference | Recognition] = []
        for document, references in pairs:
            # Skip documents where predictions are not available
            # (None indicates the recognizer couldn't process this document)
            if references is not None:
                pending += self._document_records(document, references, recognizer_id)

        if pending:
            session.add_all(pending)

    def _document_records(
        self,
        document: "Document",
        references: list[tuple[int, int]],
        recognizer_id: str,
    ) -> list[Reference | Recognition]:
        """
        Build one document's reference records and its processed marker.

        Args:
            document: The document the references were predicted for
            references: Predicted (start, end) spans
            recognizer_id: ID of the recognizer that made the predictions

        Returns:
            The records to stage, references first
        """
        records: list[Reference | Recognition | None] = [
            self._create_reference_record(document, start, end, recognizer_id)
            for start, end in references
        ]
        # Mark document as processed
        records.append(self._create_recognition_record(document.id, recognizer_id))
        return [record for record in records if record is not None]

    def _create_reference_record(
        self,
        document: "Document",
        start: int,
        end: int,
        recognizer_id: str,
    ) -> Reference:
        """
        Create a reference record with the recognizer ID.

        The span's text is cut from the document in hand, so recording a
        batch needs no query per predicted span.

        Args:
            document: The document containing the reference
            start: Start position of the reference
            end: End position of the reference
            recognizer_id: ID of the recognizer
        """
        text = getattr(document, "text", None)
        return Reference(
            start=start,
            end=end,
            text=text[start:end] if isinstance(text, str) else None,
            document_id=document.id,
            recognizer_id=recognizer_id,
        )

    def _create_recognition_record(
        self, document_id: uuid.UUID, recognizer_id: str
    ) -> Recognition:
        """
        Create a recognition record for a document processed by a specific recognizer.

        Args:
            document_id: ID of the document that was processed
            recognizer_id: ID of the recognizer that processed it
        """
        return Recognition(
            document_id=document_id,
            recognizer_id=recognizer_id,
        )

    def _filter_unprocessed_documents(
        self, session: Session, documents: list["Document"], recognizer_id: str
    ) -> list["Document"]:
        """
        Filter out documents that have already been processed by this recognizer.

        Args:
            session: Database session
            documents: List of all documents to check
            recognizer_id: ID of the recognizer to check for

        Returns:
            List of documents that haven't been processed by this recognizer
        """
        document_ids = [doc.id for doc in documents]
        processed_ids = RecognitionRepository.get_processed_document_ids(
            session, document_ids, recognizer_id
        )
        return [doc for doc in documents if doc.id not in processed_ids]
