"""Pinned embedding models for the gold-span resolution comparison (issue #163).

Each revision is the full commit hash that the Hugging Face model API reported
on 2026-10-09. Prompts, pooling, normalization and dimensions come from the
sentence-transformers configuration files at that commit. Declaring a pin does
not download or load any weight, and it does not check the bytes of a cached
copy: the digest of the exact weights is a separate, later input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

VERIFIED_ON = "2026-10-09"

Pooling = Literal["mean", "cls", "last_token"]
Role = Literal["query", "document"]
Status = Literal["retained", "added"]

_COMMIT = re.compile(r"[a-f0-9]{40}")

# Copied from Qwen/Qwen3-Embedding-0.6B config_sentence_transformers.json at its
# pinned commit. The document side has no prompt.
_QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)


@dataclass(frozen=True)
class EmbeddingModel:
    """One pinned checkpoint and the encoding recipe its documentation prescribes."""

    key: str
    repository: str
    revision: str
    status: Status
    dimension: int
    max_seq_length: int
    max_seq_length_source: str
    pooling: Pooling
    normalize: bool
    query_prompt: str
    document_prompt: str
    task: str | None
    trust_remote_code: bool
    license: str

    def __post_init__(self) -> None:
        """Refuse a pin that is not a full commit hash, so branches cannot slip in."""
        if not _COMMIT.fullmatch(self.revision):
            msg = f"{self.key}: revision must be a full 40-character commit hash"
            raise ValueError(msg)

    def prompt(self, role: Role) -> str:
        """Return the literal text prefixed to each string of a role; empty for none."""
        return self.query_prompt if role == "query" else self.document_prompt


# Retained from the existing resolver stack.
GEO_MINILM = EmbeddingModel(
    key="geo-minilm",
    repository="dguzh/geo-all-MiniLM-L6-v2",
    revision="d678769f195c194ffdc6f3735a9360118a855b43",
    status="retained",
    dimension=384,
    max_seq_length=256,
    max_seq_length_source="sentence_bert_config.json",
    pooling="mean",
    normalize=True,
    query_prompt="",
    document_prompt="",
    task=None,
    trust_remote_code=False,
    license="not declared in model metadata",
)

JINA_V5_TEXT_SMALL = EmbeddingModel(
    key="jina-v5-text-small",
    repository="jinaai/jina-embeddings-v5-text-small",
    revision="dd76d535f5447ca3897a9c893fb1e612ead98192",
    status="retained",
    dimension=1024,
    max_seq_length=32768,
    max_seq_length_source="config.json max_position_embeddings",
    pooling="last_token",
    normalize=True,
    query_prompt="Query: ",
    document_prompt="Document: ",
    task="retrieval",
    trust_remote_code=True,
    license="cc-by-nc-4.0",
)

# Added by #163.
QWEN3_EMBEDDING_0_6B = EmbeddingModel(
    key="qwen3-embedding-0.6b",
    repository="Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    status="added",
    dimension=1024,
    max_seq_length=32768,
    max_seq_length_source="config.json max_position_embeddings",
    pooling="last_token",
    normalize=True,
    query_prompt=_QWEN_QUERY_PROMPT,
    document_prompt="",
    task=None,
    trust_remote_code=False,
    license="apache-2.0",
)

QWEN3_EMBEDDING_4B = EmbeddingModel(
    key="qwen3-embedding-4b",
    repository="Qwen/Qwen3-Embedding-4B",
    revision="5cf2132abc99cad020ac570b19d031efec650f2b",
    status="added",
    dimension=2560,
    max_seq_length=40960,
    max_seq_length_source="config.json max_position_embeddings",
    pooling="last_token",
    normalize=True,
    query_prompt=_QWEN_QUERY_PROMPT,
    document_prompt="",
    task=None,
    trust_remote_code=False,
    license="apache-2.0",
)

BGE_M3 = EmbeddingModel(
    key="bge-m3",
    repository="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    status="added",
    dimension=1024,
    max_seq_length=8192,
    max_seq_length_source="sentence_bert_config.json",
    pooling="cls",
    normalize=True,
    query_prompt="",
    document_prompt="",
    task=None,
    trust_remote_code=False,
    license="mit",
)

MODELS: tuple[EmbeddingModel, ...] = (
    GEO_MINILM,
    JINA_V5_TEXT_SMALL,
    QWEN3_EMBEDDING_0_6B,
    QWEN3_EMBEDDING_4B,
    BGE_M3,
)

_BY_KEY = {model.key: model for model in MODELS}


def get_model(key: str) -> EmbeddingModel:
    """Return the pinned model registered under ``key``.

    Args:
        key: A registry key such as ``"geo-minilm"``

    Returns:
        The pinned model

    Raises:
        KeyError: If no model is registered under the key
    """
    if key not in _BY_KEY:
        msg = f"Unknown embedding model {key!r}; expected one of {sorted(_BY_KEY)}."
        raise KeyError(msg)
    return _BY_KEY[key]


@dataclass(frozen=True)
class HistoricalSetting:
    """A threshold used before #163, labelled with where it came from.

    Historical values were chosen on other models and other score scales, so
    each setting is only ever recorded against the model and the embedding policy
    that used it.
    """

    model: str
    policy: Literal["similarity", "population"]
    min_similarity: float
    origin: str


HISTORICAL_SETTINGS: tuple[HistoricalSetting, ...] = (
    HistoricalSetting(
        model="geo-minilm",
        policy="similarity",
        min_similarity=0.6,
        origin=(
            "SentenceTransformerResolver constructor default (min_similarity=0.6 "
            "in geoparser/modules/resolvers/sentencetransformer.py); no population "
            "prior; never calibrated"
        ),
    ),
    HistoricalSetting(
        model="geo-minilm",
        policy="similarity",
        min_similarity=0.0,
        origin=(
            "benchmark CLI default; the 2026-09-23 hybrid evidence runs, which use "
            "the similarity ranking with no population prior, used 0.0 with the "
            "MiniLM encoder"
        ),
    ),
    HistoricalSetting(
        model="geo-minilm",
        policy="population",
        min_similarity=0.0,
        origin=(
            "benchmark CLI default; the 2026-09-23 prior evidence runs, which use "
            "population weight 0.3, used 0.0 with the MiniLM encoder"
        ),
    ),
)


def historical_setting(
    model: str, policy: str, min_similarity: float
) -> HistoricalSetting | None:
    """Return the registered historical setting for a model, policy and value, if any.

    Args:
        model: A registry key
        policy: The embedding policy, ``similarity`` or ``population``
        min_similarity: The threshold to look up

    Returns:
        The matching setting, or None when the value was never used for that model
        under that policy
    """
    for setting in HISTORICAL_SETTINGS:
        if (
            setting.model == model
            and setting.policy == policy
            and setting.min_similarity == min_similarity
        ):
            return setting
    return None
