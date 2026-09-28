import socket
from ipaddress import ip_address

import pytest


def _is_loopback_host(host: str | bytes) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    host = host.casefold().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_address(address: object) -> None:
    if isinstance(address, (str, bytes)):
        return
    if isinstance(address, tuple) and address and _is_loopback_host(address[0]):
        return
    raise RuntimeError("network access is disabled in acceptance tests")


@pytest.fixture(autouse=True)
def disable_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_loopback_host(host):
            raise RuntimeError("network access is disabled in acceptance tests")
        return original_getaddrinfo(host, *args, **kwargs)

    def connect(sock, address):
        _require_loopback_address(address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        _require_loopback_address(address)
        return original_connect_ex(sock, address)

    def sendto(sock, data, *args):
        _require_loopback_address(args[-1])
        return original_sendto(sock, data, *args)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
