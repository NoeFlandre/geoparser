"""Scoring semantics of the two rerankers, and validation of their raw outputs.

Qwen3-Reranker scores each (query, document) prompt on its own. The score is the
probability of "yes" in a two-way softmax over the "no" and "yes" logits at the
last position, as the model card documents. Jina reranker v3.5 scores the whole
shortlist in one listwise prompt and returns cosine similarities. Those scores
are comparable only within one call, so a shortlist must never be split.

The functions that read model output return validated numbers or raise
:class:`MalformedOutputError`. A malformed output is never converted into a score.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")

# Taken from the Qwen3-Reranker-0.6B model card at revision
# e61197ed45024b0ed8a2d74b80b4d909f1255473.
QWEN_MAX_LENGTH = 8192
QWEN_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Taken from modeling.py and the tokenizer config of jina-reranker-v3.5 at
# revision e8a93f33f0b22108f8c2364f8484ce3422552fbc. ``rerank`` truncates the
# query to 1024 tokens and each document to 8192 tokens, and it groups documents
# into blocks of at most 125.
JINA_MAX_QUERY_TOKENS = 1024
JINA_MAX_DOCUMENT_TOKENS = 8192
JINA_BLOCK_SIZE = 125


class MalformedOutputError(ValueError):
    """A reranker returned something that cannot be turned into a score."""


def _finite(value: object) -> float:
    """Return the value as a float, rejecting bools, non-numbers and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"{value!r} is not a real number"
        raise MalformedOutputError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{value!r} is not a finite number"
        raise MalformedOutputError(msg)
    return number


def _position(value: object, count: int) -> int:
    """Return a shortlist position, rejecting bools, floats and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        msg = f"index {value!r} is not an integer"
        raise MalformedOutputError(msg)
    index = int(value)
    if not 0 <= index < count:
        msg = f"index {index} is outside the shortlist of {count}"
        raise MalformedOutputError(msg)
    return index


# --- Qwen3-Reranker ------------------------------------------------------------------


def qwen_pair_text(
    query: str, document: str, instruction: str = QWEN_INSTRUCTION
) -> str:
    """Format one (query, document) pair with the documented template."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def qwen_yes_probability(no_logit: float, yes_logit: float) -> float:
    """Return P(yes) from a softmax over the "no" and "yes" logits.

    The two forms are algebraically the same, and each one avoids overflow in
    one direction. The result is in [0, 1]. Higher is more relevant.

    Raises:
        MalformedOutputError: If either logit is not a finite real number.
    """
    difference = _finite(no_logit) - _finite(yes_logit)
    if difference <= 0:
        return 1.0 / (1.0 + math.exp(difference))
    weight = math.exp(-difference)
    return weight / (1.0 + weight)


def qwen_token_budget(
    *, prefix_tokens: int, suffix_tokens: int, max_length: int = QWEN_MAX_LENGTH
) -> int:
    """Return how many pair tokens fit once the prefix and suffix are added.

    Raises:
        ValueError: If no token is left for the pair.
    """
    budget = max_length - prefix_tokens - suffix_tokens
    if budget < 1:
        msg = f"no room for the document: the budget is {budget} tokens"
        raise ValueError(msg)
    return budget


def truncate_tokens(token_ids: Sequence[int], budget: int) -> tuple[list[int], bool]:
    """Keep the first ``budget`` tokens and report whether anything was cut."""
    if budget < 0:
        msg = f"truncation budget must not be negative, got {budget}"
        raise ValueError(msg)
    kept = list(token_ids[:budget])
    return kept, len(kept) < len(token_ids)


def assemble_qwen_ids(
    *,
    prefix: Sequence[int],
    pair: Sequence[int],
    suffix: Sequence[int],
    max_length: int = QWEN_MAX_LENGTH,
) -> tuple[list[int], bool]:
    """Join prefix, truncated pair and suffix; report whether the pair was cut.

    The pair is cut from its end, which matches the tokenizer truncation in the
    model card. The prefix and suffix are never cut.
    """
    budget = qwen_token_budget(
        prefix_tokens=len(prefix), suffix_tokens=len(suffix), max_length=max_length
    )
    kept, truncated = truncate_tokens(pair, budget)
    return [*prefix, *kept, *suffix], truncated


def _yes_probability_of(pair: object) -> float:
    if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
        msg = f"expected a (no, yes) logit pair, got {pair!r}"
        raise MalformedOutputError(msg)
    no_logit, yes_logit = pair
    return qwen_yes_probability(no_logit, yes_logit)


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive batches in order; the last batch may be shorter."""
    if size < 1:
        msg = f"batch size must be at least 1, got {size}"
        raise ValueError(msg)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def score_pointwise(
    logit_batch: Callable[[list[str]], Sequence[object]],
    prompts: Sequence[str],
    batch_size: int,
) -> list[float]:
    """Score independent prompts in batches and return scores in prompt order.

    ``logit_batch`` returns one (no, yes) logit pair per prompt in its batch.

    Raises:
        MalformedOutputError: If a batch returns the wrong number of pairs, or a pair
            that is not two finite numbers.
    """
    scores: list[float] = []
    for batch in batched(prompts, batch_size):
        pairs = logit_batch(batch)
        if len(pairs) != len(batch):
            msg = f"expected {len(batch)} logit pairs, got {len(pairs)}"
            raise MalformedOutputError(msg)
        scores.extend(_yes_probability_of(pair) for pair in pairs)
    return scores


# --- Jina reranker (listwise) ----------------------------------------------------------


def checked_scores(values: Sequence[object], count: int) -> list[float]:
    """Return ``count`` finite scores, or raise :class:`MalformedOutputError`."""
    if len(values) != count:
        msg = f"expected {count} scores, got {len(values)}"
        raise MalformedOutputError(msg)
    return [_finite(value) for value in values]


def listwise_scores(results: Sequence[object], count: int) -> list[float]:
    """Align a listwise reranker's per-document results to shortlist positions.

    Each result is a mapping with an ``index`` (its position in the documents
    that were passed in) and a ``relevance_score``. Results may arrive in any
    order. Exactly one result must exist for each position.

    Raises:
        MalformedOutputError: If the count is wrong, an index is missing, invalid or
            repeated, or a score is missing or not finite.
    """
    if len(results) != count:
        msg = f"expected {count} results, got {len(results)}"
        raise MalformedOutputError(msg)
    scores = [0.0] * count
    seen: set[int] = set()
    for result in results:
        if not isinstance(result, Mapping):
            msg = f"{result!r} is not a mapping"
            raise MalformedOutputError(msg)
        position = _position(result.get("index"), count)
        if position in seen:
            msg = f"index {position} is repeated"
            raise MalformedOutputError(msg)
        seen.add(position)
        scores[position] = _finite(result.get("relevance_score"))
    return scores


def score_listwise(
    rerank: Callable[[list[str]], Sequence[object]], documents: Sequence[str]
) -> list[float]:
    """Score a whole shortlist with one listwise call.

    The reranker is called once with every document, so the scores share one
    prompt context.

    Raises:
        ValueError: If there are no documents. Callers must abstain instead.
        MalformedOutputError: If the reranker's results cannot be aligned.
    """
    if not documents:
        msg = "cannot rerank no documents: the shortlist is empty"
        raise ValueError(msg)
    return listwise_scores(rerank(list(documents)), len(documents))


def jina_blocks(
    document_lengths: Sequence[int],
    *,
    query_length: int,
    model_max_length: int,
    block_size: int = JINA_BLOCK_SIZE,
) -> list[list[int]]:
    """Predict how ``rerank`` splits one call into forward passes.

    The rule mirrors the pinned code: a block is closed after a document once it
    holds ``block_size`` documents, or once the remaining capacity drops to the
    document limit. A result with more than one block means the scores were not
    produced in one shared context, and the report must flag it.
    """
    blocks: list[list[int]] = []
    current: list[int] = []
    capacity = model_max_length - 2 * query_length
    for position, length in enumerate(document_lengths):
        current.append(position)
        capacity -= length
        if len(current) >= block_size or capacity <= JINA_MAX_DOCUMENT_TOKENS:
            blocks.append(current)
            current = []
            capacity = model_max_length - 2 * query_length
    if current:
        blocks.append(current)
    return blocks
