"""
Check benchmark corpora against the bytes and totals recorded on 2026-09-23.

The corpus loaders fetch from mutable URLs (``master`` and ``main`` branches,
and a figshare file id), so every run re-checks what it is about to score.
For each corpus the source file is downloaded or reused from the cache, its
SHA-256 is recorded, and only then is it parsed. The first mismatch stops the
check: nothing is silently accepted when the upstream bytes or the parsed
totals change.

NewsLi comes from the UniTopRank release zip, which must match the recorded
size and MD5 before anything else is done with it.

    python scripts/g5k/verify_corpora.py --cache DIR --output FILE [--corpus NAME ...]

Exit status: 0 when every checked corpus matches, 3 on the first mismatch,
2 on a usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from scripts.benchmark import corpora, corpus

# Digest is the first 16 hex characters of the loader's corpus digest, as
# written to benchmark-report.json on 2026-09-23. Documents and gold spans are
# the recorded totals.
RECORDED: dict[str, tuple[str, int, int]] = {
    "geovirus": ("2f8a928b9af4a442", 229, 2167),
    "hipe2020-de": ("4d90055cddb45a7f", 48, 558),
    "hipe2020-fr": ("7e756f77b368ec9a", 43, 800),
    "hipe2020-en": ("745da3aa1fce5f24", 38, 158),
    "topres19th-en": ("87911185a99ddd56", 110, 880),
    "newseye-de": ("6be58effb319164f", 8, 628),
    "newseye-fr": ("9f7e02ad02a29b0f", 33, 665),
    "newseye-fi": ("25b3956789d0a741", 18, 214),
    "newseye-sv": ("1a63659c45bb2bc1", 18, 265),
    "newsli-ar": ("bce24ad1279bc71c", 500, 3319),
    "newsli-de": ("f892793167f8e8a0", 500, 1019),
    "newsli-es": ("1fd75cdc4954aa6d", 500, 1270),
    "newsli-fa": ("7e4fe252e65da859", 71, 326),
    "newsli-ja": ("fba4707aa24003ab", 500, 1577),
    "newsli-pl": ("139fd34031005560", 186, 196),
    "newsli-ro": ("39857cb6b9094388", 226, 385),
    "newsli-sr": ("adf8496feae09afa", 500, 985),
    "newsli-ta": ("4a227b75125b5d73", 500, 1578),
    "newsli-tr": ("8afdca7d96ab679e", 500, 1578),
    "newsli-uk": ("5757f00bb2c9be45", 164, 258),
}

NEWSLI_RELEASE_BYTES = 63356728
NEWSLI_RELEASE_MD5 = "8e6f82a0931cb909e6f18217e80ce1f3"


class Mismatch(Exception):
    """A corpus no longer matches what was recorded."""


def _hash_file(path: Path, algorithm: str) -> str:
    """Return the hex digest of a file, read in chunks."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_newsli_release(path: Path) -> None:
    """Refuse the NewsLi zip unless its size and MD5 are the recorded ones."""
    size = path.stat().st_size
    if size != NEWSLI_RELEASE_BYTES:
        msg = f"{path.name}: {size} bytes, expected {NEWSLI_RELEASE_BYTES}"
        raise Mismatch(msg)
    md5 = _hash_file(path, "md5")
    if md5 != NEWSLI_RELEASE_MD5:
        msg = f"{path.name}: md5 {md5}, expected {NEWSLI_RELEASE_MD5}"
        raise Mismatch(msg)


def verify_one(name: str, cache: Path) -> dict[str, object]:
    """
    Download (or reuse), hash and parse one corpus, then compare it.

    Args:
        name: A key of corpora.CORPORA
        cache: Directory holding one folder per corpus

    Returns:
        A JSON-safe record of the file, its hash and the parsed totals

    Raises:
        Mismatch: When the bytes or the parsed totals differ from RECORDED
    """
    spec = corpora.CORPORA[name]
    folder = cache / name
    if spec.kind == corpora.NEWSLI:
        source = corpus.download_corpus(
            cache / corpora.NEWSLI_RELEASE, url=spec.url
        )
        check_newsli_release(source)
    else:
        source = corpus.download_corpus(folder / spec.filename, url=spec.url)
    raw_sha256 = _hash_file(source, "sha256")

    loaded = corpora.load(name, folder)
    documents = len(loaded.documents)
    gold = corpus.gold_toponym_count(loaded.documents)
    digest, expected_documents, expected_gold = RECORDED[name]
    record: dict[str, object] = {
        "corpus": name,
        "url": spec.url,
        "file": source.name,
        "file_sha256": raw_sha256,
        "digest": loaded.digest,
        "recorded_digest": digest,
        "documents": documents,
        "recorded_documents": expected_documents,
        "gold_toponyms": gold,
        "recorded_gold_toponyms": expected_gold,
    }
    problems = []
    if loaded.digest != digest:
        problems.append(f"digest {loaded.digest} != {digest}")
    if documents != expected_documents:
        problems.append(f"documents {documents} != {expected_documents}")
    if gold != expected_gold:
        problems.append(f"gold {gold} != {expected_gold}")
    record["status"] = "match" if not problems else "mismatch"
    if problems:
        record["problems"] = problems
        msg = f"{name}: " + "; ".join(problems)
        raise Mismatch(msg)
    return record


def main(argv: list[str] | None = None) -> int:
    """Verify the requested corpora, writing one JSON record per corpus."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        action="append",
        choices=sorted(corpora.CORPORA),
        help="Corpus to verify; repeat for several. Defaults to every corpus.",
    )
    arguments = parser.parse_args(argv)
    names = arguments.corpus or list(corpora.CORPORA)
    cache = arguments.cache.expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    status = 0
    try:
        for name in dict.fromkeys(names):
            print(f"verifying {name}", flush=True)
            records.append(verify_one(name, cache))
            print(f"  {records[-1]['status']}: {records[-1]['documents']} docs, "
                  f"{records[-1]['gold_toponyms']} gold, "
                  f"sha256 {records[-1]['file_sha256']}", flush=True)
    except Mismatch as error:
        status = 3
        print(f"MISMATCH {error}", file=sys.stderr)
        records.append({"status": "mismatch", "problems": [str(error)]})
    finally:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps({"records": records, "exit": status}, indent=2) + "\n",
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
