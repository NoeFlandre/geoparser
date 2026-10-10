"""The CI jobs that run the UniTopRank ranker test fetch the commit that pins.py verifies."""

from scripts.unitoprank_benchmark.pins import COMMIT, REPOSITORY
from tests.unit.test_quality.project_contract_support import job_steps, named_step

FETCH_STEP = "Fetch the pinned UniTopRank checkout"
FETCH_ENV = {"UNITORANK_REPOSITORY": REPOSITORY, "UNITORANK_COMMIT": COMMIT}
ENVIRONMENT_STEP = "Create the reviewed UniTopRank ranker environment"
GAUNTLET_STEP = "Run the complete deterministic quality gauntlet"
LINUX_3_12 = "matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'"


def test_the_test_cell_fetches_the_pinned_commit_before_pytest() -> None:
    steps = job_steps("test.yml", "pytest")
    names = [step.get("name") for step in steps]

    fetch = named_step(steps, FETCH_STEP)

    assert names.index(FETCH_STEP) < names.index("Run pytest")
    assert fetch["env"] == FETCH_ENV
    assert fetch["if"] == (
        "matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'"
    )


def test_only_the_linux_3_12_cell_points_pytest_at_the_checkout() -> None:
    pytest_step = named_step(job_steps("test.yml", "pytest"), "Run pytest")

    assert pytest_step["env"]["UNITORANK_CHECKOUT"] == (
        "${{ matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12' "
        "&& format('{0}/UniTopRank', runner.temp) || '' }}"
    )


def test_the_test_cell_builds_the_reviewed_environment_before_pytest() -> None:
    steps = job_steps("test.yml", "pytest")
    names = [step.get("name") for step in steps]
    setup = named_step(steps, ENVIRONMENT_STEP)

    assert (
        names.index(FETCH_STEP)
        < names.index(ENVIRONMENT_STEP)
        < names.index("Run pytest")
    )
    assert setup["if"] == LINUX_3_12
    assert "ranker-requirements.txt" in setup["run"]


def test_only_the_linux_3_12_cell_points_pytest_at_the_reviewed_python() -> None:
    pytest_step = named_step(job_steps("test.yml", "pytest"), "Run pytest")

    assert pytest_step["env"]["UNITORANK_PYTHON"] == (
        "${{ matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12' "
        "&& format('{0}/unitorank-ranker/bin/python', runner.temp) || '' }}"
    )


def test_the_quality_gauntlet_fetches_the_pinned_checkout_too() -> None:
    steps = job_steps("quality.yml", "quality")
    names = [step.get("name") for step in steps]

    fetch = named_step(steps, FETCH_STEP)

    assert names.index(FETCH_STEP) < names.index(GAUNTLET_STEP)
    assert fetch["env"] == FETCH_ENV
    assert named_step(steps, GAUNTLET_STEP)["env"]["UNITORANK_CHECKOUT"] == (
        "${{ runner.temp }}/UniTopRank"
    )


def test_the_quality_gauntlet_builds_the_reviewed_environment_first() -> None:
    steps = job_steps("quality.yml", "quality")
    names = [step.get("name") for step in steps]

    assert (
        names.index(FETCH_STEP)
        < names.index(ENVIRONMENT_STEP)
        < names.index(GAUNTLET_STEP)
    )
    assert "ranker-requirements.txt" in named_step(steps, ENVIRONMENT_STEP)["run"]
    assert named_step(steps, GAUNTLET_STEP)["env"]["UNITORANK_PYTHON"] == (
        "${{ runner.temp }}/unitorank-ranker/bin/python"
    )
