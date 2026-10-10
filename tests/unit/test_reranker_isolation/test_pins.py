"""Pins and the review gate for the two reranker checkpoints.

The commits and the custom-code digest were read from the Hugging Face metadata
on 2026-10-09. The digest of ``modeling.py`` was recomputed from the pinned
file. The published SHA-256 vector for ``abc`` is an independent oracle for the
hashing helper.
"""

import re

import pytest

from scripts.reranker_isolation.pins import (
    CHECKPOINTS,
    JINA_RERANKER_V3_5,
    QWEN3_RERANKER_0_6B,
    Checkpoint,
    CustomCode,
    UnreviewedCodeError,
    remote_code_permitted,
    sha256_hex,
)

COMMIT = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def test_checkpoints_pin_full_commit_revisions() -> None:
    assert QWEN3_RERANKER_0_6B.revision == "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    assert JINA_RERANKER_V3_5.revision == "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
    assert all(COMMIT.match(checkpoint.revision) for checkpoint in CHECKPOINTS)


def test_qwen_is_pointwise_and_runs_no_custom_code() -> None:
    assert QWEN3_RERANKER_0_6B.repository == "Qwen/Qwen3-Reranker-0.6B"
    assert QWEN3_RERANKER_0_6B.scoring == "pointwise"
    assert QWEN3_RERANKER_0_6B.custom_code is None
    assert QWEN3_RERANKER_0_6B.licence == "apache-2.0"


def test_jina_is_listwise_and_non_commercially_licensed() -> None:
    assert JINA_RERANKER_V3_5.repository == "jinaai/jina-reranker-v3.5"
    assert JINA_RERANKER_V3_5.scoring == "listwise"
    assert JINA_RERANKER_V3_5.licence == "cc-by-nc-4.0"


def test_jina_pins_its_reviewed_modeling_file() -> None:
    code = JINA_RERANKER_V3_5.custom_code

    assert code == CustomCode(
        path="modeling.py",
        sha256="ec6612461b4307eb3bab089e6916c00881b7e6b2e8edfbd56a9ace4560244837",
    )
    assert DIGEST.match(code.sha256)


def test_sha256_matches_the_published_vector_for_abc() -> None:
    assert (
        sha256_hex(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_pointwise_checkpoint_never_permits_remote_code() -> None:
    assert remote_code_permitted(QWEN3_RERANKER_0_6B, None) is False
    assert remote_code_permitted(QWEN3_RERANKER_0_6B, b"unrelated bytes") is False


def test_matching_custom_code_is_permitted() -> None:
    reviewed = b"class JinaForRanking: ...\n"
    checkpoint = Checkpoint(
        repository="example/listwise",
        revision="a" * 40,
        licence="mit",
        scoring="listwise",
        custom_code=CustomCode(path="modeling.py", sha256=sha256_hex(reviewed)),
    )

    assert remote_code_permitted(checkpoint, reviewed) is True


def test_altered_custom_code_is_refused() -> None:
    checkpoint = Checkpoint(
        repository="example/listwise",
        revision="a" * 40,
        licence="mit",
        scoring="listwise",
        custom_code=CustomCode(path="modeling.py", sha256=sha256_hex(b"reviewed")),
    )

    with pytest.raises(UnreviewedCodeError, match="does not match the reviewed pin"):
        remote_code_permitted(checkpoint, b"reviewed!")


def test_listwise_checkpoint_refuses_code_that_was_never_fetched() -> None:
    with pytest.raises(UnreviewedCodeError, match="fetched and verified"):
        remote_code_permitted(JINA_RERANKER_V3_5, None)


def test_unreviewed_code_error_is_a_value_error() -> None:
    assert issubclass(UnreviewedCodeError, ValueError)
