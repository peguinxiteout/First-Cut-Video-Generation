"""The Windows ConnectionResetError suppressor.

These run on every platform: the point is to prove the patch is narrow
(only ConnectionResetError, only from that one teardown call) and inert
off-Windows, and CI is Linux.
"""
from __future__ import annotations

import sys

from src.core import win_asyncio


def test_install_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(win_asyncio, "_installed", False, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert win_asyncio.install() is False


def test_install_patches_on_windows_and_swallows_only_connection_reset(monkeypatch):
    """Simulates the Windows path: a fake transport class stands in for
    asyncio's private _ProactorBasePipeTransport, which doesn't exist on
    Linux."""
    calls: list = []

    class FakeTransport:
        def _call_connection_lost(self, exc):
            calls.append(exc)
            raise ConnectionResetError(10054, "An existing connection was forcibly closed")

    fake_module = type(sys)("asyncio.proactor_events")
    fake_module._ProactorBasePipeTransport = FakeTransport
    monkeypatch.setitem(sys.modules, "asyncio.proactor_events", fake_module)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(win_asyncio, "_installed", False, raising=False)

    assert win_asyncio.install() is True

    # The exact failure from the user's terminal is now swallowed...
    assert FakeTransport()._call_connection_lost(None) is None
    assert calls == [None]


def test_the_patch_does_not_swallow_other_exceptions(monkeypatch):
    """The whole risk of a patch like this is hiding a real failure. Only
    ConnectionResetError may be swallowed; anything else must propagate."""
    class FakeTransport:
        def _call_connection_lost(self, exc):
            raise ValueError("a real bug")

    fake_module = type(sys)("asyncio.proactor_events")
    fake_module._ProactorBasePipeTransport = FakeTransport
    monkeypatch.setitem(sys.modules, "asyncio.proactor_events", fake_module)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(win_asyncio, "_installed", False, raising=False)

    win_asyncio.install()

    import pytest
    with pytest.raises(ValueError, match="a real bug"):
        FakeTransport()._call_connection_lost(None)


def test_install_is_idempotent(monkeypatch):
    class FakeTransport:
        def _call_connection_lost(self, exc):
            return "original"

    fake_module = type(sys)("asyncio.proactor_events")
    fake_module._ProactorBasePipeTransport = FakeTransport
    monkeypatch.setitem(sys.modules, "asyncio.proactor_events", fake_module)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(win_asyncio, "_installed", False, raising=False)

    assert win_asyncio.install() is True
    first = FakeTransport._call_connection_lost
    assert win_asyncio.install() is False
    # Not double-wrapped — a second install must not re-patch the patch.
    assert FakeTransport._call_connection_lost is first
