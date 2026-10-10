"""Recognition with a caller-loaded native-language spaCy pipeline.

The caller loads a pinned, already-installed pipeline and passes it in. This
adapter never downloads a package and never routes a language to another
pipeline. Labels are harmonized explicitly: only labels present in
``place_label_map`` produce spans, and every mapped value is the WikiANN
location class ``LOC``.
"""

import typing as t
from collections.abc import Mapping

from geoparser.modules.recognizers.base import Recognizer

LOCATION_CLASS = "LOC"


def _positive_integer(value: t.Any, name: str) -> int:
    if type(value) is not int or value < 1:
        message = f"{name} must be a positive integer."
        raise ValueError(message)
    return value


def _nonblank_string(value: t.Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        message = f"{name} must be a nonblank string."
        raise ValueError(message)
    return value


def _label_map(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        message = "place_label_map must be a nonempty mapping."
        raise ValueError(message)
    labels: dict[str, str] = {}
    for label, target in value.items():
        _nonblank_string(label, "place_label_map label")
        if target != LOCATION_CLASS:
            message = f"place_label_map values must be {LOCATION_CLASS!r}."
            raise ValueError(message)
        labels[label] = target
    return dict(sorted(labels.items()))


def _check_documents(texts: list[str]) -> None:
    if not isinstance(texts, list):
        message = "texts must be a list of strings."
        raise TypeError(message)
    for text in texts:
        if not isinstance(text, str):
            message = "Every document in texts must be a string."
            raise TypeError(message)


class NativeSpacyRecognizer(Recognizer):
    """Emit character spans for harmonized place labels of one spaCy pipeline.

    Args:
        nlp: A loaded spaCy pipeline. Its identity is recorded by the caller
            through ``package`` and ``version``; this adapter does not verify
            the installed distribution.
        language: The target language code this pipeline is routed to.
        package: The pinned pipeline package name.
        version: The pinned package version.
        place_label_map: Native entity label to WikiANN class, for example
            ``{"GPE": "LOC", "LOC": "LOC"}``. Other labels are dropped.
        batch_size: Documents per call to the pipeline's batched ``pipe``.
    """

    NAME = "NativeSpacyRecognizer"

    def __init__(
        self,
        nlp: t.Any,
        *,
        language: str,
        package: str,
        version: str,
        place_label_map: Mapping[str, str],
        batch_size: int = 8,
    ):
        self.nlp = nlp
        self.place_label_map = _label_map(place_label_map)
        self.batch_size = _positive_integer(batch_size, "batch_size")
        super().__init__(
            language=_nonblank_string(language, "language"),
            package=_nonblank_string(package, "package"),
            version=_nonblank_string(version, "version"),
            place_label_map=self.place_label_map,
            batch_size=self.batch_size,
        )

    def predict(self, texts: list[str]) -> list[list[tuple[int, int]] | None]:
        """Return sorted, distinct place spans for each document."""
        return [sorted(spans) for spans in self.predict_batch(texts)]

    def predict_batch(self, texts: list[str]) -> list[set[tuple[int, int]]]:
        """Return the same place spans as sets, aligned with the input order."""
        _check_documents(texts)
        return [
            {
                (entity.start_char, entity.end_char)
                for entity in document.ents
                if entity.label_ in self.place_label_map
            }
            for document in self.nlp.pipe(texts, batch_size=self.batch_size)
        ]
