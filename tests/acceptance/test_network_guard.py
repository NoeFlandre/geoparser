import socket

import pytest

from tests.fixtures.network import external_network_disabled


@pytest.fixture(scope="session")
def session_network_guard():
    """Confirm the guard is active in session-scoped setup."""
    with (
        external_network_disabled(),
        pytest.raises(RuntimeError, match="network access is disabled"),
    ):
        socket.getaddrinfo("203.0.113.1", 443)
    return True


def test_network_guard_is_active_for_session_setup(session_network_guard):
    """Session fixtures cannot make external network requests."""
    assert session_network_guard


def test_acceptance_tests_reject_external_address_resolution() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.getaddrinfo("203.0.113.1", 443)


def test_acceptance_tests_reject_external_socket_connections() -> None:
    connection = socket.socket.__new__(socket.socket)

    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.socket.connect(connection, ("203.0.113.1", 443))
