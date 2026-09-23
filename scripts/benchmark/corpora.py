"""
The corpora the benchmark can score, and one way to load any of them.

Every corpus loads to the same ``LoadedCorpus``, so the runner and the report
never need to know which one they are scoring. GeoVirus carries its own
coordinates; the HIPE-2022 corpora link toponyms to Wikidata, placed through a
committed coordinate cache (see ``wikidata``).

Only the unmasked test splits are registered: the models under test may have
seen training data, and the masked files have their gold removed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark import corpus
from scripts.benchmark.corpus import Document
from scripts.benchmark.hipe import hipe_qids, parse_hipe
from scripts.benchmark.wikidata import load_coordinates

HIPE_BASE_URL = (
    "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1"
)
COORDINATE_CACHE = Path(__file__).with_name("data") / "wikidata-coordinates.json"

GEOVIRUS = "geovirus"
HIPE = "hipe"


@dataclass(frozen=True)
class CorpusSpec:
    """Where a corpus comes from and how to read it."""

    name: str
    language: str
    url: str
    kind: str

    @property
    def filename(self) -> str:
        """The name the downloaded file is cached under."""
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class LoadedCorpus:
    """A corpus ready to score, with a digest over everything its gold rests on."""

    name: str
    language: str
    documents: list[Document]
    digest: str


def _hipe(dataset: str, language: str) -> CorpusSpec:
    """Return the spec of one HIPE-2022 test split."""
    return CorpusSpec(
        name=f"{dataset}-{language}",
        language=language,
        url=(
            f"{HIPE_BASE_URL}/{dataset}/{language}/"
            f"HIPE-2022-v2.1-{dataset}-test-{language}.tsv"
        ),
        kind=HIPE,
    )


CORPORA: dict[str, CorpusSpec] = {
    spec.name: spec
    for spec in (
        CorpusSpec(GEOVIRUS, "en", corpus.CORPUS_URL, GEOVIRUS),
        _hipe("hipe2020", "de"),
        _hipe("hipe2020", "fr"),
        _hipe("hipe2020", "en"),
        _hipe("newseye", "de"),
        _hipe("newseye", "fr"),
        _hipe("newseye", "fi"),
        _hipe("newseye", "sv"),
        _hipe("topres19th", "en"),
    )
}
DEFAULT = (GEOVIRUS,)


def load(
    name: str,
    cache_dir: Path,
    *,
    limit: int | None = None,
    coordinate_cache: Path = COORDINATE_CACHE,
) -> LoadedCorpus:
    """
    Download (once) and parse one registered corpus.

    Args:
        name: A key of CORPORA
        cache_dir: Where downloaded corpus files are kept between runs
        limit: Keep only the first this many documents
        coordinate_cache: The Wikidata coordinate cache for HIPE corpora

    Returns:
        The corpus, parsed

    Raises:
        KeyError: When the name is not registered
    """
    spec = CORPORA[name]
    path = corpus.download_corpus(cache_dir / spec.filename, url=spec.url)
    if spec.kind == GEOVIRUS:
        return LoadedCorpus(
            spec.name,
            spec.language,
            corpus.parse_corpus(path, limit=limit),
            corpus.corpus_digest(path),
        )
    coordinates = load_coordinates(hipe_qids(path), coordinate_cache)
    digest = hashlib.sha256(path.read_bytes())
    digest.update(json.dumps(sorted(coordinates.items())).encode())
    return LoadedCorpus(
        spec.name,
        spec.language,
        parse_hipe(path, coordinates, limit=limit),
        digest.hexdigest()[:16],
    )
