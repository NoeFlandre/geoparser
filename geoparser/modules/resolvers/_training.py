"""Fine-tuning for SentenceTransformerResolver.

This is the only resolver code that needs ``datasets`` and the
sentence-transformers training classes, so their imports live here.
"""

import typing as t
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import ContrastiveLoss
from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
)

from geoparser._logging import get_logger

if t.TYPE_CHECKING:
    from geoparser.gazetteer.feature import Feature
    from geoparser.gazetteer.gazetteer import Gazetteer

logger = get_logger(__name__)


class TrainingMixin:
    """Fine-tune the resolver's model on annotated references."""

    transformer: SentenceTransformer
    gazetteer: "Gazetteer"

    # Supplied by the resolver; declared so the training code type-checks.
    if t.TYPE_CHECKING:

        def _extract_context(self, text: str, start: int, end: int) -> str: ...

        def _generate_description(self, candidate: "Feature") -> str: ...

    def fit(  # noqa: PLR0913, PLR0917 - public API; make keyword-only in the next major release
        self,
        texts: list[str],
        references: list[list[tuple[int, int]]],
        referents: list[list[tuple[str, str]]],
        output_path: str | Path,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        save_strategy: str = "epoch",
    ) -> None:
        """
        Fine-tune the SentenceTransformer model using references and their resolved referents as training data.

        This method gathers all references that have been resolved (i.e., have referents), extracts
        their contexts and all candidate descriptions, and uses them to create positive and negative
        training examples for fine-tuning the underlying SentenceTransformer model using ContrastiveLoss.

        Args:
            texts: List of document text strings
            references: List of lists of (start, end) position tuples
            referents: List of lists of (gazetteer_name, identifier) tuples
            output_path: Directory path to save the fine-tuned model
            epochs: Number of training epochs (default: 1)
            batch_size: Training batch size (default: 8)
            learning_rate: Learning rate for training (default: 2e-5)
            warmup_ratio: Warmup ratio for learning rate scheduler (default: 0.1)
            save_strategy: When to save the model during training (default: "epoch")

        Raises:
            ValueError: If no training examples can be created from the provided documents
        """
        logger.info("Preparing training data from referent annotations...")

        # Step 1: Gather training data from resolved references
        training_data = self._prepare_training_data(texts, references, referents)

        if not training_data["sentence1"] or len(training_data["sentence1"]) == 0:
            msg = "No training examples found. Ensure documents contain references with referent annotations."
            raise ValueError(msg)

        logger.info(f"Created {len(training_data['sentence1'])} training examples")

        # Step 2: Create training dataset
        train_dataset = Dataset.from_dict(training_data)

        # Step 3: Setup training loss
        train_loss = ContrastiveLoss(self.transformer)

        # Step 4: Configure training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            save_strategy=save_strategy,
            logging_strategy="steps",
            logging_steps=max(1, len(training_data["sentence1"]) // (batch_size * 10)),
            eval_strategy="no",  # No evaluation for now
            save_total_limit=2,  # Keep only 2 checkpoints
            load_best_model_at_end=False,
        )

        # Step 5: Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.transformer,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
        )

        logger.info("Starting model fine-tuning...")

        # Step 6: Train the model
        trainer.train()

        # Step 7: Save the final model
        self.transformer.save_pretrained(str(output_path))

        logger.info(f"Model fine-tuning completed and saved to: {output_path}")

    def _prepare_training_data(
        self,
        texts: list[str],
        references: list[list[tuple[int, int]]],
        referents: list[list[tuple[str, str]]],
    ) -> dict[str, list]:
        """
        Prepare training data from documents with resolved references.

        This method extracts all references that have been resolved (have referents),
        gets their contexts and all candidate descriptions to create both positive
        and negative training examples for ContrastiveLoss.

        Args:
            texts: List of document text strings
            references: List of lists of (start, end) position tuples
            referents: List of lists of (gazetteer_name, identifier) tuples

        Returns:
            Dictionary with 'sentence1', 'sentence2', and 'label' lists for training
        """
        sentence1_texts = []  # contexts
        sentence2_texts = []  # candidate descriptions
        labels = []  # 1 for positive, 0 for negative

        for text, doc_references, doc_referents in zip(
            texts, references, referents, strict=True
        ):
            for (start, end), (_gazetteer_name, identifier) in zip(
                doc_references, doc_referents, strict=True
            ):
                # Extract context for this reference
                context = self._extract_context(text, start, end)

                # Get all candidates for this reference text to create negative examples
                reference_text = text[start:end]
                candidates = self.gazetteer.search(reference_text)

                for candidate in candidates:
                    # Generate description for this candidate
                    description = self._generate_description(candidate)

                    # Determine if this is a positive or negative example
                    label = 1 if candidate.identifier == identifier else 0

                    # Add as training example
                    sentence1_texts.append(context)
                    sentence2_texts.append(description)
                    labels.append(label)

        return {
            "sentence1": sentence1_texts,
            "sentence2": sentence2_texts,
            "label": labels,
        }
