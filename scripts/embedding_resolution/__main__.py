"""Print the pinned registry, or validate and freeze a comparison plan offline.

Neither command loads a model, opens a dataset or reads a gazetteer. ``freeze``
writes only the planned inventory, and only when a complete output path is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from geoparser.modules.resolvers.ranking import PRIOR_SCALE
from scripts._io import write_json_atomic
from scripts.benchmark_protocol.summary import inventory_summary
from scripts.embedding_resolution.models import (
    HISTORICAL_SETTINGS,
    MODELS,
    VERIFIED_ON,
)
from scripts.embedding_resolution.protocol import FreezePlan, build_experiment
from scripts.embedding_resolution.resolution import (
    CALIBRATION_GRID,
    DISTANCE_THRESHOLDS_KM,
    POPULATION_WEIGHT,
)


def registry_document() -> dict[str, object]:
    """Return every pin, prompt, historical setting and scoring constant as JSON."""
    return {
        "verified_on": VERIFIED_ON,
        "models": [dataclasses.asdict(model) for model in MODELS],
        "historical_settings": [
            dataclasses.asdict(setting) for setting in HISTORICAL_SETTINGS
        ],
        "population_weight": POPULATION_WEIGHT,
        "population_prior_scale": PRIOR_SCALE,
        "distance_thresholds_km": list(DISTANCE_THRESHOLDS_KM),
        "calibration_grid": {
            "minimum": CALIBRATION_GRID[0],
            "maximum": CALIBRATION_GRID[-1],
            "points": len(CALIBRATION_GRID),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    Dispatch the registry and freeze commands.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``

    Returns:
        0 on success, 2 when the plan is invalid or cannot be read
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("registry", help="print the pinned registry as JSON")
    freeze = commands.add_parser("freeze", help="validate a freeze plan offline")
    freeze.add_argument("plan", type=Path, metavar="PLAN.json")
    freeze.add_argument("--output", type=Path, default=None, metavar="EXPERIMENT.json")
    arguments = parser.parse_args(argv)
    if arguments.command == "registry":
        print(json.dumps(registry_document(), indent=2, sort_keys=True))
        return 0
    return _freeze(arguments.plan, arguments.output)


def _freeze(plan_path: Path, output: Path | None) -> int:
    """Validate the plan, print its inventory summary, and optionally write it."""
    try:
        plan = FreezePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        experiment = build_experiment(plan)
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        print(f"Invalid freeze plan: {error}", file=sys.stderr)
        return 2
    if output is not None:
        write_json_atomic(output, experiment.model_dump(mode="json"), pretty=True)
    print(json.dumps(inventory_summary(experiment), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
