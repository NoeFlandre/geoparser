import subprocess

import pytest

from scripts.g5k import wait_for_node


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_build_parser_exposes_the_readiness_wait_options() -> None:
    args = wait_for_node.build_parser().parse_args(["hydra-1"])

    assert (args.host, args.timeout, args.interval) == (
        "hydra-1",
        wait_for_node.DEFAULT_TIMEOUT_SECONDS,
        wait_for_node.DEFAULT_POLL_INTERVAL_SECONDS,
    )


def test_wait_until_ready_retries_refused_connection_until_node_is_ready() -> None:
    clock = FakeClock()
    probe_timeouts: list[float] = []
    responses = iter(
        [
            (False, ""),
            (False, "Connection refused"),
            (True, ""),
        ]
    )

    def probe(remaining: float) -> tuple[bool, str]:
        probe_timeouts.append(remaining)
        return next(responses)

    ready, attempts, error = wait_for_node.wait_until_ready(
        probe,
        timeout_seconds=30,
        poll_interval_seconds=10,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert (
        ready,
        attempts,
        error,
        clock.sleeps,
        probe_timeouts,
    ) == (True, 3, None, [10, 10], [30, 20, 10])


def test_wait_until_ready_stops_at_deadline_and_keeps_last_error() -> None:
    clock = FakeClock()
    probe_timeouts: list[float] = []

    def probe(remaining: float) -> tuple[bool, str]:
        probe_timeouts.append(remaining)
        return False, "Connection refused"

    ready, attempts, error = wait_for_node.wait_until_ready(
        probe,
        timeout_seconds=25,
        poll_interval_seconds=10,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert (
        ready,
        attempts,
        error,
        clock.sleeps,
        probe_timeouts,
    ) == (False, 3, "Connection refused", [10, 10, 5], [25, 15, 5])


def test_wait_until_ready_rejects_success_returned_at_deadline() -> None:
    clock = FakeClock()

    def probe(remaining: float) -> tuple[bool, str]:
        clock.now += remaining
        return True, ""

    result = wait_for_node.wait_until_ready(
        probe,
        timeout_seconds=5,
        poll_interval_seconds=2,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result == (False, 1, None)
    assert clock.sleeps == []


def test_wait_until_ready_does_not_sleep_after_probe_uses_remaining_time() -> None:
    clock = FakeClock()

    def probe(remaining: float) -> tuple[bool, str]:
        clock.now += remaining
        return False, "probe timed out"

    ready, attempts, error = wait_for_node.wait_until_ready(
        probe,
        timeout_seconds=5,
        poll_interval_seconds=2,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert not ready
    assert attempts == 1
    assert error == "probe timed out"
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("timeout_seconds", "poll_interval_seconds", "message"),
    [
        (0, 1, "timeout_seconds"),
        (1, 0, "poll_interval_seconds"),
        (float("inf"), 1, "timeout_seconds"),
        (1, float("nan"), "poll_interval_seconds"),
    ],
)
def test_wait_until_ready_rejects_nonpositive_durations(
    timeout_seconds: float,
    poll_interval_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        wait_for_node.wait_until_ready(
            lambda _: (True, ""),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (0, "", (True, "")),
        (255, "Connection refused", (False, "Connection refused")),
        (255, "", (False, "oarsh exited with status 255")),
    ],
)
def test_oarsh_probe_reports_command_result(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: str,
    expected: tuple[bool, str],
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    monkeypatch.setattr(wait_for_node.subprocess, "run", run)

    assert wait_for_node._oarsh_probe("hydra-1", 20) == expected
    assert calls == [
        (
            ["oarsh", "hydra-1", "true"],
            {"capture_output": True, "check": False, "text": True, "timeout": 10},
        )
    ]


def test_oarsh_probe_turns_subprocess_timeout_into_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("oarsh", 3)

    monkeypatch.setattr(wait_for_node.subprocess, "run", run)

    assert wait_for_node._oarsh_probe("hydra-1", 3) == (
        False,
        "oarsh probe timed out after 3s",
    )


def test_oarsh_probe_turns_launch_error_into_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("missing oarsh")

    monkeypatch.setattr(wait_for_node.subprocess, "run", run)

    assert wait_for_node._oarsh_probe("hydra-1", 3) == (False, "missing oarsh")


def test_main_returns_success_after_node_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(wait_for_node.shutil, "which", lambda _: "/usr/bin/oarsh")
    monkeypatch.setattr(
        wait_for_node, "_oarsh_probe", lambda _host, _timeout: (True, "")
    )

    assert wait_for_node.main(["hydra-1"]) == 0
    assert "hydra-1 is ready after 1 probe(s)." in capsys.readouterr().out


def test_main_reports_timeout_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = FakeClock()
    original_wait = wait_for_node.wait_until_ready

    def fake_clock_wait(
        probe: wait_for_node.ReadinessProbe,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> tuple[bool, int, str | None]:
        return original_wait(
            probe,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    monkeypatch.setattr(wait_for_node.shutil, "which", lambda _: "/usr/bin/oarsh")
    monkeypatch.setattr(
        wait_for_node,
        "_oarsh_probe",
        lambda _host, _timeout: (False, "Connection refused"),
    )
    monkeypatch.setattr(wait_for_node, "wait_until_ready", fake_clock_wait)

    assert wait_for_node.main(["hydra-1", "--timeout", "25", "--interval", "10"]) == 1
    captured = capsys.readouterr()
    assert "Timed out waiting for hydra-1 after 3 probe(s)." in captured.err
    assert "Last error: Connection refused" in captured.err
    assert clock.sleeps == [10, 10, 5]


def test_main_fails_fast_when_oarsh_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(wait_for_node.shutil, "which", lambda _: None)

    assert wait_for_node.main(["hydra-1"]) == 2
    assert "oarsh is not available on this frontend" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["hydra-1", "--timeout", "0"],
        ["hydra-1", "--interval", "0"],
        ["hydra-1", "--timeout", "inf"],
        ["hydra-1", "--interval", "nan"],
        ["hydra-1", "--timeout", "invalid"],
    ],
)
def test_main_rejects_nonpositive_wait_durations(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        wait_for_node.main(arguments)

    assert error.value.code == 2
