"""Characterise the pilot's timing boundaries, model order and reported values.

``run_pilot`` is driven through its public entry point with fake models, a fake
project, a fake gazetteer builder and a fake clock. Nothing is downloaded, no
model runs, and no real time is read: every ``perf_counter`` call returns a
value from the fake clock, which advances only by a fixed 1 ms step per read
and by the fixed cost of each fake model call.
"""

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scripts import pilot

RUNTIME_ENVIRONMENT = (
    "DATABASE_URL",
    "GEOPARSER_GAZETTEERS_DIR",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

CASES = (
    pilot.PilotCase("first", "Alpha Beta.", (pilot.PilotSpan(0, 5, "100"),)),
    pilot.PilotCase("second", "Gamma here.", (pilot.PilotSpan(0, 5, "200"),)),
    pilot.PilotCase("third", "Nothing.", ()),
)

# Fixed model costs, in fake milliseconds.
RECOGNIZER_LOAD_MS = 100
RESOLVER_LOAD_MS = 200
RECOGNITION_MS = {"Alpha Beta.": 4, "Gamma here.": 6, "Nothing.": 2}
RESOLUTION_MS = {"Alpha Beta.": 3, "Gamma here.": 5, "Nothing.": 1}
RECOGNITION_SPANS = {
    "Alpha Beta.": [(0, 5)],
    "Gamma here.": [(0, 5), (6, 10)],
    "Nothing.": [],
}
RESOLUTIONS = {
    ("Alpha Beta.", (0, 5)): "100",
    ("Gamma here.", (0, 5)): "201",
}


class FakeWorld:
    """Shared event log and the fake clock that every fake model advances."""

    def __init__(self, artifact_path: Path) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.ms = 0
        self.artifact_path = artifact_path

    def read_clock(self) -> float:
        """A ``perf_counter`` read: log the tick, then advance by one 1 ms step."""
        tick = self.ms
        self.events.append(("clock", tick))
        self.ms += 1
        return tick / 1000

    def advance(self, label: str, ms: int) -> None:
        """Model work: log it and move the clock forward by its fixed cost."""
        self.events.append(("advance", label, ms))
        self.ms += ms


def _module(name: str, **attributes: object) -> ModuleType:
    """Build a module object whose attributes are the given fakes."""
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _fake_modules(world: FakeWorld) -> dict[str, ModuleType]:
    """Build the fake ``torch`` and ``geoparser`` modules that ``run_pilot`` imports."""

    class FakeTransformer:
        def __init__(self) -> None:
            self._max_seq_length = 512

        @property
        def max_seq_length(self) -> int:
            return self._max_seq_length

        @max_seq_length.setter
        def max_seq_length(self, value: int) -> None:
            world.events.append(("max_seq_length", value))
            self._max_seq_length = value

    class FakeReranker:
        def __init__(self) -> None:
            self.dtype = "torch.bfloat16"

        def float(self) -> "FakeReranker":
            world.events.append(("reranker.float",))
            self.dtype = "torch.float32"
            return self

        def parameters(self) -> Any:
            return iter([SimpleNamespace(dtype=self.dtype)])

    class GLiNER2Recognizer:
        def __init__(self) -> None:
            world.events.append(("load", "GLiNER2Recognizer"))
            world.advance("load GLiNER2", RECOGNIZER_LOAD_MS)
            self.model_name = "fake-gliner"

        def predict(self, texts: list[str]) -> list[list[tuple[int, int]]]:
            return [self._document_references(text) for text in texts]

        def _document_references(self, text: str) -> list[tuple[int, int]]:
            world.events.append(("recognize", text))
            world.advance(f"recognize {text}", RECOGNITION_MS[text])
            return list(RECOGNITION_SPANS[text])

        def __del__(self) -> None:
            world.events.append(("release", "GLiNER2Recognizer"))

    class JinaResolver:
        def __init__(
            self,
            *,
            gazetteer_name: str,
            min_similarity: float,
            max_tiers: int,
            attribute_map: dict[str, str],
        ) -> None:
            world.events.append(("load", "JinaResolver"))
            world.advance("load JinaResolver", RESOLVER_LOAD_MS)
            self.gazetteer_name = gazetteer_name
            self.min_similarity = min_similarity
            self.max_tiers = max_tiers
            self.attribute_map = attribute_map
            self.model_name = "fake-embedding"
            self.reranker_name = "fake-reranker"
            self.rerank_top_k = 20
            self.transformer = FakeTransformer()
            self.reranker = FakeReranker()

        def predict(
            self, texts: list[str], references: list[list[tuple[int, int]]]
        ) -> list[list[tuple[str, str] | None]]:
            results: list[list[tuple[str, str] | None]] = [
                [None] * len(refs) for refs in references
            ]
            for tier in range(1, self.max_tiers + 1):
                world.events.append(("tier", tier))
                self._evaluate_candidates(
                    texts, references, results, self.min_similarity
                )
            return results

        def _evaluate_candidates(
            self,
            texts: list[str],
            references: list[list[tuple[int, int]]],
            results: list[list[tuple[str, str] | None]],
            min_similarity: float,
        ) -> None:
            for index, text in enumerate(texts):
                self._evaluate_document(
                    text, references[index], results[index], min_similarity
                )

        def _evaluate_document(
            self,
            text: str,
            doc_references: list[tuple[int, int]],
            doc_results: list[tuple[str, str] | None],
            min_similarity: float,
        ) -> None:
            world.events.append(("evaluate_document", text))
            world.advance(f"resolve {text}", RESOLUTION_MS[text])
            for index, span in enumerate(doc_references):
                key = (text, span)
                if doc_results[index] is None and key in RESOLUTIONS:
                    doc_results[index] = ("andorranames", RESOLUTIONS[key])

    class GazetteerBuilder:
        def build(self, config_path: Path) -> Path:
            world.events.append(("gazetteer.build", config_path.name))
            return world.artifact_path

    class Project:
        def __init__(self, name: str) -> None:
            world.events.append(("project.open",))
            self._texts: dict[str, str] = {}
            self._toponyms: dict[str, list[tuple[int, int, str | None]]] = {}

        def create_documents(self, texts: list[str]) -> list[str]:
            world.events.append(("project.create_documents", tuple(texts)))
            ids = [f"doc{index}" for index in range(len(texts))]
            for doc_id, text in zip(ids, texts, strict=True):
                self._texts[doc_id] = text
                self._toponyms[doc_id] = []
            return ids

        def run_recognizer(self, recognizer: Any) -> None:
            world.events.append(("run_recognizer", "start"))
            # The database reads documents back in reverse creation order, so
            # the recognizer must be matched to cases by text, not position.
            order = list(reversed(list(self._texts)))
            spans = recognizer.predict([self._texts[doc_id] for doc_id in order])
            for doc_id, doc_spans in zip(order, spans, strict=True):
                self._toponyms[doc_id] = [(s, e, None) for s, e in doc_spans]
            world.events.append(("run_recognizer", "end"))

        def run_resolver(self, resolver: Any) -> None:
            world.events.append(("run_resolver", "start"))
            ids = list(self._texts)
            texts = [self._texts[doc_id] for doc_id in ids]
            references = [
                [(s, e) for s, e, _ in self._toponyms[doc_id]] for doc_id in ids
            ]
            results = resolver.predict(texts, references)
            for doc_id, doc_results in zip(ids, results, strict=True):
                self._toponyms[doc_id] = [
                    (s, e, result[1] if result is not None else None)
                    for (s, e, _), result in zip(
                        self._toponyms[doc_id], doc_results, strict=True
                    )
                ]
            world.events.append(("run_resolver", "end"))

        def get_documents(self, doc_ids: list[str]) -> list[SimpleNamespace]:
            world.events.append(("get_documents", tuple(doc_ids)))
            return [
                SimpleNamespace(
                    toponyms=[
                        SimpleNamespace(
                            start=s,
                            end=e,
                            location=None
                            if identifier is None
                            else SimpleNamespace(identifier=identifier),
                        )
                        for s, e, identifier in self._toponyms[doc_id]
                    ]
                )
                for doc_id in doc_ids
            ]

        def delete(self) -> None:
            world.events.append(("project.delete",))

    return {
        "torch": _module(
            "torch",
            set_num_threads=lambda count: world.events.append(
                ("torch.set_num_threads", count)
            ),
        ),
        "geoparser.gazetteer.build": _module(
            "geoparser.gazetteer.build", GazetteerBuilder=GazetteerBuilder
        ),
        "geoparser.modules": _module(
            "geoparser.modules",
            GLiNER2Recognizer=GLiNER2Recognizer,
            JinaResolver=JinaResolver,
        ),
        "geoparser.project": _module("geoparser.project", Project=Project),
    }


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeWorld:
    """Install the fakes, the fake clock and a fixed three-case corpus."""
    fake = FakeWorld(tmp_path / "gazetteers" / "andorranames.sqlite")
    for name, module in _fake_modules(fake).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(pilot, "time", SimpleNamespace(perf_counter=fake.read_clock))
    monkeypatch.setattr(
        pilot,
        "gc",
        SimpleNamespace(collect=lambda: fake.events.append(("gc.collect",))),
    )
    monkeypatch.setattr(pilot, "PILOT_CASES", CASES)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    for name in RUNTIME_ENVIRONMENT:
        # Setting then deleting makes monkeypatch restore the variable's
        # original state, even when the pilot writes it during the test.
        monkeypatch.setenv(name, "unset-by-test")
        monkeypatch.delenv(name)
    return fake


def _run(tmp_path: Path) -> tuple[Path, Path]:
    return pilot.run_pilot(
        config_path=tmp_path / "andorra.yaml",
        output_dir=tmp_path / "out",
        hf_home=None,
        offline=False,
    )


EXPECTED_EVENTS: list[tuple[Any, ...]] = [
    ("torch.set_num_threads", 4),
    ("gazetteer.build", "andorra.yaml"),
    ("project.open",),
    ("project.create_documents", ("Alpha Beta.", "Gamma here.", "Nothing.")),
    # Recognizer is loaded before the recognition timer starts.
    ("load", "GLiNER2Recognizer"),
    ("advance", "load GLiNER2", 100),
    ("clock", 100),
    ("run_recognizer", "start"),
    # Reverse read-back order: Nothing, Gamma, Alpha. Each timer is one read
    # before the model call and one read after it.
    ("clock", 101),
    ("recognize", "Nothing."),
    ("advance", "recognize Nothing.", 2),
    ("clock", 104),
    ("clock", 105),
    ("recognize", "Gamma here."),
    ("advance", "recognize Gamma here.", 6),
    ("clock", 112),
    ("clock", 113),
    ("recognize", "Alpha Beta."),
    ("advance", "recognize Alpha Beta.", 4),
    ("clock", 118),
    ("run_recognizer", "end"),
    ("clock", 119),
    # The recognizer is released before the resolver is loaded.
    ("release", "GLiNER2Recognizer"),
    ("gc.collect",),
    # Resolver is loaded and configured before the resolution timer starts.
    ("load", "JinaResolver"),
    ("advance", "load JinaResolver", 200),
    ("max_seq_length", 128),
    ("reranker.float",),
    ("clock", 320),
    ("run_resolver", "start"),
    # Creation order. Each tier re-evaluates every document, and each
    # document's timer wraps only its own evaluation.
    ("tier", 1),
    ("clock", 321),
    ("evaluate_document", "Alpha Beta."),
    ("advance", "resolve Alpha Beta.", 3),
    ("clock", 325),
    ("clock", 326),
    ("evaluate_document", "Gamma here."),
    ("advance", "resolve Gamma here.", 5),
    ("clock", 332),
    ("clock", 333),
    ("evaluate_document", "Nothing."),
    ("advance", "resolve Nothing.", 1),
    ("clock", 335),
    ("tier", 2),
    ("clock", 336),
    ("evaluate_document", "Alpha Beta."),
    ("advance", "resolve Alpha Beta.", 3),
    ("clock", 340),
    ("clock", 341),
    ("evaluate_document", "Gamma here."),
    ("advance", "resolve Gamma here.", 5),
    ("clock", 347),
    ("clock", 348),
    ("evaluate_document", "Nothing."),
    ("advance", "resolve Nothing.", 1),
    ("clock", 350),
    ("run_resolver", "end"),
    ("clock", 351),
    ("get_documents", ("doc0", "doc1", "doc2")),
    ("project.delete",),
]


def test_timer_boundaries_and_model_order_are_characterised(
    world: FakeWorld, tmp_path: Path
) -> None:
    _run(tmp_path)

    assert world.events == EXPECTED_EVENTS


def test_reported_values_and_documents_are_characterised(
    world: FakeWorld, tmp_path: Path
) -> None:
    json_path, markdown_path = _run(tmp_path)

    assert json_path == tmp_path / "out" / "pilot-report.json"
    assert markdown_path == tmp_path / "out" / "pilot-report.md"
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == 1
    assert report["models"] == {
        "recognizer": "fake-gliner",
        "resolver_embedding": "fake-embedding",
        "resolver_reranker": "fake-reranker",
    }
    assert report["configuration"] == {
        "gazetteer": "andorranames",
        "gazetteer_artifact": "andorranames.sqlite",
        "min_similarity": 0.5,
        "max_tiers": 2,
        "rerank_top_k": 20,
        "max_sequence_length": 128,
        "reranker_dtype": "torch.float32",
        "torch_threads": 4,
        "recognition_batch_elapsed_ms": 19.0,
        "resolution_batch_elapsed_ms": 31.0,
        "batch_elapsed_ms": 50.0,
        "execution": (
            "Project.run_recognizer then Project.run_resolver with model release "
            "between phases"
        ),
        "timing_scope": (
            "per-document recognition plus resolution decisions; shared "
            "batch embedding time is reported separately"
        ),
        "gold_cases": 3,
    }
    assert report["documents"] == [
        {
            "id": "first",
            "text": "Alpha Beta.",
            "gold": [
                {"start": 0, "end": 5, "text": "Alpha", "identifier": "100"},
            ],
            "predicted": [
                {"start": 0, "end": 5, "text": "Alpha", "identifier": "100"},
            ],
            "metrics": {
                "recognition": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "resolution": {"accuracy": 1.0},
            },
            "elapsed_ms": 13.0,
        },
        {
            "id": "second",
            "text": "Gamma here.",
            "gold": [
                {"start": 0, "end": 5, "text": "Gamma", "identifier": "200"},
            ],
            "predicted": [
                {"start": 0, "end": 5, "text": "Gamma", "identifier": "201"},
                {"start": 6, "end": 10, "text": "here", "identifier": None},
            ],
            "metrics": {
                "recognition": {
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 0.6666666666666666,
                },
                "resolution": {"accuracy": 0.0},
            },
            "elapsed_ms": 19.0,
        },
        {
            "id": "third",
            "text": "Nothing.",
            "gold": [],
            "predicted": [],
            "metrics": {
                "recognition": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "resolution": {"accuracy": 1.0},
            },
            "elapsed_ms": 7.0,
        },
    ]
    aggregate = report["aggregate"]
    assert aggregate["document_count"] == 3
    assert aggregate["gold_annotation_count"] == 2
    assert aggregate["predicted_annotation_count"] == 3
    assert aggregate["recognition"]["precision"] == pytest.approx(2 / 3)
    assert aggregate["recognition"]["recall"] == 1.0
    assert aggregate["recognition"]["f1"] == pytest.approx(0.8)
    assert aggregate["resolution"]["accuracy"] == 0.5
