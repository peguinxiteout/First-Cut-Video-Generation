from __future__ import annotations

import ipaddress
import socket

import pytest

from src.data.schemas import PropertyContact, PropertyInput, PropertyLocation


def _is_local(host) -> bool:
    if host in (None, "", "localhost"):
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def network_attempts(monkeypatch):
    """No test reaches a remote host, whatever keys are in .env.

    A model or provider call that is not faked fails here instead of
    spending quota. Broad `except Exception` handlers in the agents would
    swallow the error and the test would still pass, so every blocked
    attempt is also recorded and fails the test at teardown. A test that
    blocks on purpose clears the list. Loopback and Unix sockets are
    allowed (the event loop and the test client use them). Subprocesses
    are not covered: the Chromium worker only opens local files."""
    attempts: list[str] = []
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def blocked(target):
        attempts.append(str(target))
        raise OSError(f"network access is blocked in tests: {target}")

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            blocked(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def checked(real):
        def wrapper(self, address):
            if self.family != getattr(socket, "AF_UNIX", None) and isinstance(address, tuple) \
                    and not _is_local(address[0]):
                blocked(address[0])
            return real(self, address)
        return wrapper

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", checked(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", checked(real_connect_ex))
    yield attempts
    assert not attempts, f"a test tried to reach the network: {attempts}"


@pytest.fixture
def sample_property_input() -> PropertyInput:
    return PropertyInput(
        property_id="PROP-TEST",
        name="Test Residences",
        developer="Test Developer",
        location=PropertyLocation(city="Ahmedabad", area="SG Highway"),
        configuration="3 BHK",
        area="1450 sq ft",
        price="₹1.2 Cr",
        amenities=["Swimming Pool", "Gym", "Clubhouse"],
        description="A test property used only in unit tests.",
        contact=PropertyContact(phone="+91-00000-00000", email="sales@example.com"),
    )


@pytest.fixture(autouse=True)
def _offline_image_generation(tmp_path, monkeypatch):
    """Every test runs image generation OFFLINE, into a temp directory.

    Without this, a developer with real credentials in .env would have the
    integration tests call a live image API — slow, flaky, and spending a
    real allowance — and every run would leave generated files in
    data/image_cache. Tests that exercise a provider inject a fake
    transport directly (see tests/unit/test_image_gen.py), so nothing here
    needs the network.
    """
    from src.core.config import settings

    monkeypatch.setattr(settings, "image_gen_provider", "mock", raising=False)
    monkeypatch.setattr(settings, "gemini_api_key", "", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "", raising=False)
    monkeypatch.setattr(settings, "hf_token", "", raising=False)
    monkeypatch.setattr(settings, "cloudflare_account_id", "", raising=False)
    monkeypatch.setattr(settings, "cloudflare_api_token", "", raising=False)
    # And no research tool reaches the network either. Same reasoning: a
    # test that quietly called OpenStreetMap or Google News would be slow,
    # flaky, dependent on somebody else's uptime, and — for the APIs that
    # ask you not to hammer them — rude. Tests that exercise a tool inject
    # a fake transport (see tests/unit/test_tools.py).
    monkeypatch.setattr(settings, "tools_enabled", False, raising=False)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"), raising=False)
