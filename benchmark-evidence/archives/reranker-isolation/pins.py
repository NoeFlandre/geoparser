"""Pinned reranker checkpoints and the review gate for their custom code.

The commits, licences and the digest of the custom modelling file were read
from the Hugging Face metadata on 2026-10-09. The digest was recomputed from the
pinned ``modeling.py``. Nothing here downloads a checkpoint or runs code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


class UnreviewedCodeError(ValueError):
    """Custom model code was not fetched, or its bytes differ from the pin."""


@dataclass(frozen=True)
class CustomCode:
    """One file of custom modelling code and the digest of its reviewed bytes."""

    path: str
    sha256: str


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint pinned to a full commit, with its licence and scoring style."""

    repository: str
    revision: str
    licence: str
    scoring: Literal["pointwise", "listwise"]
    custom_code: CustomCode | None = None


QWEN3_RERANKER_0_6B = Checkpoint(
    repository="Qwen/Qwen3-Reranker-0.6B",
    revision="e61197ed45024b0ed8a2d74b80b4d909f1255473",
    licence="apache-2.0",
    scoring="pointwise",
)

JINA_RERANKER_V3_5 = Checkpoint(
    repository="jinaai/jina-reranker-v3.5",
    revision="e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    licence="cc-by-nc-4.0",
    scoring="listwise",
    custom_code=CustomCode(
        path="modeling.py",
        sha256="ec6612461b4307eb3bab089e6916c00881b7e6b2e8edfbd56a9ace4560244837",
    ),
)

CHECKPOINTS = (QWEN3_RERANKER_0_6B, JINA_RERANKER_V3_5)


def sha256_hex(content: bytes) -> str:
    """Return the SHA-256 digest of the exact bytes, as lowercase hex."""
    return hashlib.sha256(content).hexdigest()


def remote_code_permitted(checkpoint: Checkpoint, content: bytes | None) -> bool:
    """Return whether the checkpoint's custom code may run.

    A checkpoint without custom code never needs remote code, so the answer is
    False. Otherwise the caller must pass the exact bytes it fetched, and they
    must match the reviewed digest.

    Raises:
        UnreviewedCodeError: If the bytes were not supplied, or they differ from
            the reviewed digest.
    """
    if checkpoint.custom_code is None:
        return False
    if content is None:
        msg = (
            f"{checkpoint.repository} custom code must be fetched and verified "
            "before remote code may run"
        )
        raise UnreviewedCodeError(msg)
    if sha256_hex(content) != checkpoint.custom_code.sha256:
        msg = f"{checkpoint.custom_code.path} does not match the reviewed pin"
        raise UnreviewedCodeError(msg)
    return True
