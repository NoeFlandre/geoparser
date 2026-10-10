"""Typed encoding adapters for the pinned embedding models.

:class:`EmbeddingAdapter` owns the parts of encoding that every model shares:
the prompt for each role, fixed-size batches, the output shape, finiteness,
and L2 normalization. An :class:`Encoder` only maps one batch of strings to one
row per string. Keeping the checks here means a model that returns the wrong
shape or a zero vector fails the same way whichever backend produced it.
"""

from __future__ import annotations

import typing as t
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.embedding_resolution.models import EmbeddingModel, Role

if t.TYPE_CHECKING:
    from collections.abc import Iterable


class Encoder(Protocol):
    """Anything that encodes one batch of strings, one row per string."""

    def encode(self, texts: list[str], *, prompt: str, batch_size: int) -> np.ndarray:
        """Return an array of shape ``(len(texts), dimension)``.

        Args:
            texts: One batch of strings, in order
            prompt: Literal text to prefix to every string; empty for none
            batch_size: The configured batch size, which the backend must use for
                any internal batching so the measured batch size is the one planned
        """
        ...


class EmbeddingAdapter:
    """Encode queries and documents for one pinned model, batch by batch."""

    def __init__(
        self, model: EmbeddingModel, encoder: Encoder, *, batch_size: int
    ) -> None:
        """
        Bind a model to an encoder.

        Args:
            model: The pinned model whose prompts and dimension apply
            encoder: The backend that encodes each batch
            batch_size: Strings per call to the encoder

        Raises:
            ValueError: If ``batch_size`` is less than one
        """
        if batch_size < 1:
            msg = f"batch_size must be at least 1, got {batch_size}."
            raise ValueError(msg)
        self.model = model
        self.encoder = encoder
        self.batch_size = batch_size

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed reference contexts under the model's query prompt.

        Args:
            texts: Context strings

        Returns:
            Unit-length rows, one per context, in order
        """
        return self._encode(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed candidate descriptions under the model's document prompt.

        Args:
            texts: Candidate description strings

        Returns:
            Unit-length rows, one per description, in order
        """
        return self._encode(texts, "document")

    def _encode(self, texts: Sequence[str], role: Role) -> np.ndarray:
        """Encode in batches and stack the checked rows."""
        items = _as_strings(texts)
        if not items:
            return np.empty((0, self.model.dimension), dtype=np.float64)
        prompt = self.model.prompt(role)
        rows = [
            self._checked(
                self.encoder.encode(batch, prompt=prompt, batch_size=self.batch_size),
                len(batch),
            )
            for batch in _batches(items, self.batch_size)
        ]
        return np.vstack(rows)

    def _checked(self, output: t.Any, expected_rows: int) -> np.ndarray:
        """Validate one batch's output and normalize it when the model does."""
        array = float_array(output)
        expected = (expected_rows, self.model.dimension)
        if array.shape != expected:
            msg = (
                f"{self.model.key}: encoder returned shape {array.shape}, "
                f"expected {expected}."
            )
            raise ValueError(msg)
        if not np.isfinite(array).all():
            msg = f"{self.model.key}: encoder returned NaN or infinite values."
            raise ValueError(msg)
        norms = np.linalg.norm(array, axis=1)
        if (norms == 0).any():
            msg = f"{self.model.key}: encoder returned a zero-length embedding."
            raise ValueError(msg)
        if self.model.normalize:
            return array / norms[:, None]
        return array


def _as_strings(texts: Iterable[str]) -> list[str]:
    """Copy the input and refuse anything that is not a string."""
    items = list(texts)
    for item in items:
        if not isinstance(item, str):
            msg = f"Embedding inputs must be strings, got {type(item).__name__}."
            raise TypeError(msg)
    return items


def float_array(output: t.Any) -> np.ndarray:
    """
    Keep the backend's floating dtype, so checks and normalization add no wider copy.

    A float32 backend stays float32 through the checks and the normalization, which
    keeps each batch at the size the backend produced. Other dtypes become float64.

    Args:
        output: The raw output of one encoder call

    Returns:
        The output as an array of its own floating dtype, or float64 for other dtypes
    """
    array = np.asarray(output)
    if array.dtype.kind == "f":
        return array
    return array.astype(np.float64)


def _batches(items: list[str], size: int) -> list[list[str]]:
    """Split the items into consecutive batches of at most ``size``."""
    return [items[start : start + size] for start in range(0, len(items), size)]


def _pinned_snapshot(model: EmbeddingModel, path: str) -> Path:
    """
    Refuse a local snapshot unless it is a directory named by the pinned commit.

    The Hub cache names each snapshot directory by its commit, so the name is the
    offline evidence of the revision the files were fetched at. Digests of the
    weights are a separate input and are not checked here.

    Args:
        model: The pinned model the snapshot must belong to
        path: The local snapshot path returned by the download

    Returns:
        The snapshot path, as a ``Path``

    Raises:
        ValueError: If the path is not a directory or is named by another commit
    """
    snapshot = Path(path)
    if not snapshot.is_dir() or snapshot.name != model.revision:
        msg = (
            f"{model.key}: snapshot {snapshot} is not a directory of the pinned "
            f"commit {model.revision}."
        )
        raise ValueError(msg)
    return snapshot


class SentenceTransformerEncoder:
    """The one place that loads a checkpoint, and only when it is constructed."""

    def __init__(self, model: EmbeddingModel, *, device: str = "cpu") -> None:
        """
        Load the pinned checkpoint from a local snapshot of its commit.

        Custom model code can reach nested parts of a checkpoint by hub id, and
        those reads take no revision, so they resolve the mutable default branch.
        Loading from the local snapshot directory keeps every nested read on the
        pinned files.

        Args:
            model: The pinned model to load
            device: Torch device string, ``"cpu"`` for the comparison's CPU runs

        Raises:
            ValueError: If the snapshot is not a directory named by the pinned commit
        """
        # Deferred: importing torch is the expensive part of this module.
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        snapshot = _pinned_snapshot(
            model,
            snapshot_download(repo_id=model.repository, revision=model.revision),
        )
        self.model = model
        self._transformer = SentenceTransformer(
            str(snapshot),
            device=device,
            trust_remote_code=model.trust_remote_code,
        )
        self._transformer.max_seq_length = model.max_seq_length

    def encode(self, texts: list[str], *, prompt: str, batch_size: int) -> np.ndarray:
        """
        Encode one batch with the model's documented prompt and task.

        Args:
            texts: One batch of strings
            prompt: Literal prompt text; empty for none
            batch_size: The configured batch size, passed through so Sentence
                Transformers does not re-batch at its own default

        Returns:
            The raw embeddings, one row per string
        """
        options: dict[str, t.Any] = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": False,
            "normalize_embeddings": False,
        }
        if prompt:
            options["prompt"] = prompt
        if self.model.task is not None:
            options["task"] = self.model.task
        return np.asarray(self._transformer.encode(texts, **options))
