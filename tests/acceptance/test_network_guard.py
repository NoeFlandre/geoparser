import socket

import pytest


def test_acceptance_tests_reject_external_address_resolution() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.getaddrinfo("203.0.113.1", 443)


def test_acceptance_tests_reject_external_socket_connections() -> None:
    connection = socket.socket.__new__(socket.socket)

    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.socket.connect(connection, ("203.0.113.1", 443))
