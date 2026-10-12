"""Wheel verification and report rendering for the native baseline runner."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from scripts.spacy_native_baselines import run
from scripts.spacy_native_baselines.roster import NativePipeline


def _pipeline(payload: bytes) -> NativePipeline:
    return NativePipeline(
        language="xx",
        spacy_language="xx",
        package="xx_core_news_sm",
        version="3.8.0",
        wheel="xx_core_news_sm-3.8.0-py3-none-any.whl",
        wheel_url="https://example.invalid/xx_core_news_sm-3.8.0-py3-none-any.whl",
        wheel_bytes=len(payload),
        license="mit",
        ner_labels=("LOC",),
        place_label_map={"LOC": "LOC"},
        label_meaning_basis="ontonotes_names",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_verify_wheel_accepts_the_pinned_bytes(tmp_path: Path) -> None:
    payload = b"wheel bytes"
    wheel = tmp_path / "xx.whl"
    wheel.write_bytes(payload)

    digest = run.verify_wheel(wheel, _pipeline(payload))

    assert digest == hashlib.sha256(payload).hexdigest()


def test_verify_wheel_refuses_a_size_change(tmp_path: Path) -> None:
    wheel = tmp_path / "xx.whl"
    wheel.write_bytes(b"longer than the pinned wheel")

    with pytest.raises(run.WheelMismatchError, match="bytes, roster pins"):
        run.verify_wheel(wheel, _pipeline(b"wheel bytes"))


def test_verify_wheel_refuses_a_digest_change_of_the_same_size(tmp_path: Path) -> None:
    wheel = tmp_path / "xx.whl"
    wheel.write_bytes(b"WHEEL BYTES")

    with pytest.raises(run.WheelMismatchError, match="sha256"):
        run.verify_wheel(wheel, _pipeline(b"wheel bytes"))


def test_verify_wheel_refuses_a_pipeline_without_a_pinned_digest(
    tmp_path: Path,
) -> None:
    payload = b"wheel bytes"
    wheel = tmp_path / "xx.whl"
    wheel.write_bytes(payload)
    unpinned = dataclasses.replace(_pipeline(payload), sha256=None)

    with pytest.raises(run.WheelMismatchError, match="no pinned sha256"):
        run.verify_wheel(wheel, unpinned)


def test_fetch_wheel_reuses_a_verified_copy_without_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"wheel bytes"
    pipeline = _pipeline(payload)
    (tmp_path / pipeline.wheel).write_bytes(payload)

    def no_network(*_args: object, **_kwargs: object) -> None:
        message = "network must not be used for a verified cached wheel"
        raise AssertionError(message)

    monkeypatch.setattr(run.urllib.request, "urlopen", no_network)

    assert run.fetch_wheel(pipeline, tmp_path) == tmp_path / pipeline.wheel


def test_fetch_wheel_replaces_a_cached_copy_that_does_not_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"wheel bytes"
    pipeline = _pipeline(payload)
    (tmp_path / pipeline.wheel).write_bytes(b"corrupted")

    class _Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            body, self._body = self._body, b""
            return body

    monkeypatch.setattr(
        run.urllib.request, "urlopen", lambda *_a, **_k: _Response(payload)
    )

    assert run.fetch_wheel(pipeline, tmp_path).read_bytes() == payload


def test_fetch_wheel_discards_a_download_that_does_not_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(b"wheel bytes")

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b""

    monkeypatch.setattr(run.urllib.request, "urlopen", lambda *_a, **_k: _Response())

    with pytest.raises(run.WheelMismatchError):
        run.fetch_wheel(pipeline, tmp_path)
    assert not (tmp_path / (pipeline.wheel + ".part")).exists()
    assert not (tmp_path / pipeline.wheel).exists()


def test_markdown_keeps_matched_and_transfer_in_separate_tables() -> None:
    result = {
        "matched": {"xx": {"sentences": 2, "precision": 0.5, "recall": 1.0, "f1": 0.667}},
        "transfer": {"yy": {"sentences": 3, "precision": 0.1, "recall": 0.2, "f1": 0.133}},
        "unsupported": {"zz": {"sentences": 4, "reason": "no pipeline"}},
    }

    text = run._markdown({**result, "dataset": {}})

    assert "## Matched" in text
    assert "## Transfer" in text
    assert "| xx | 2 | 0.500 | 1.000 | 0.667 |" in text
    assert "| yy | 3 | 0.133 |" in text
    assert "Unsupported languages with no native pipeline: 1." in text
