"""Synthetic fixtures for benchmarks; no installed datasets or model downloads."""

import json
import sqlite3
from pathlib import Path

import pytest

BENCHMARK_SIZE = 5_000


@pytest.fixture(scope="session")
def synthetic_gazetteers_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small, deterministic artifact using the production SQLite schema."""
    from geoparser.gazetteer.artifact import (
        BASE_SCHEMA,
        SEARCH_SCHEMA,
        register_functions,
    )

    directory = tmp_path_factory.mktemp("synthetic-gazetteers")
    artifact_path = directory / "benchmark.db"
    connection = sqlite3.connect(artifact_path)
    try:
        register_functions(connection)
        for statement in BASE_SCHEMA:
            connection.execute(statement)
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("name", "benchmark"),
                ("crs", "EPSG:4326"),
            ],
        )
        features = []
        names = []
        for index in range(BENCHMARK_SIZE):
            identifier = f"ID{index:05d}"
            name = f"Place {index:05d}"
            data = {
                "name": name,
                "feature_name": "Benchmark feature",
                "country_name": "Benchmark country",
                "admin1_name": "Benchmark region",
                "admin2_name": "Benchmark district",
                "population": index,
            }
            features.append(
                (index + 1, identifier, "synthetic", json.dumps(data), None)
            )
            names.append((index + 1, index + 1, name))
        connection.executemany(
            "INSERT INTO feature (id, identifier, source, data, geometry) "
            "VALUES (?, ?, ?, ?, ?)",
            features,
        )
        connection.executemany(
            "INSERT INTO name (id, feature_id, text) VALUES (?, ?, ?)", names
        )
        for statement in SEARCH_SCHEMA:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return directory


@pytest.fixture
def synthetic_gazetteer(
    synthetic_gazetteers_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Expose the generated artifact through the public Gazetteer interface."""
    monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(synthetic_gazetteers_dir))
    from geoparser.gazetteer.gazetteer import Gazetteer

    return Gazetteer("benchmark")


@pytest.fixture
def benchmark_database(request: pytest.FixtureRequest, tmp_path: Path):
    """Use production-like connection isolation in a temporary SQLite file."""
    from unittest.mock import patch

    from sqlalchemy.pool import NullPool
    from sqlmodel import SQLModel, create_engine

    import geoparser.db.models  # noqa: F401 - register mapped tables
    from geoparser.db import db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'benchmark.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    engine_patch = patch.object(db, "get_engine", return_value=engine)
    engine_patch.start()
    request.addfinalizer(engine.dispose)
    request.addfinalizer(engine_patch.stop)
    return engine


@pytest.fixture
def benchmark_project(benchmark_database):
    """Return an isolated project in the benchmark database."""
    from uuid import uuid4

    from geoparser.project.project import Project

    return Project(f"benchmark-{uuid4().hex}")


@pytest.fixture
def benchmark_project_with_documents(benchmark_project):
    """A project with 5,000 distinct, short documents and stable spans."""
    texts = [f"Place {index:05d}" for index in range(BENCHMARK_SIZE)]
    document_ids = benchmark_project.create_documents(texts)
    references = [[(0, 5)] for _ in texts]
    return benchmark_project, texts, references, document_ids
