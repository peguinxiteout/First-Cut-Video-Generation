"""
Silences a Windows-only asyncio traceback that is pure noise.

On Windows, every time a browser tab talking to the Streamlit server goes
away (a refresh, a closed tab, a navigation, the dev tools reconnecting),
the proactor event loop tries to shut down a socket that the peer has
already closed and Python prints a full, alarming traceback:

    Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
    handle: <Handle _ProactorBasePipeTransport._call_connection_lost(None)>
    Traceback (most recent call last):
      File "...asyncio\\events.py", line 88, in _run
        self._context.run(self._callback, *self._args)
      File "...asyncio\\proactor_events.py", line 116, in _call_connection_lost
        self._sock.shutdown(socket.SHUT_RDWR)
    ConnectionResetError: [WinError 10054] An existing connection was
    forcibly closed by the remote host

This is a long-standing CPython issue (bpo-39010 and friends), not a fault
in this application: the connection is *already* being torn down, and the
shutdown() that fails is the last step of tearing it down. Nothing is lost
and nothing is retried. But 57 of these scrolled past during a single
brochure generation in a user's terminal, which makes a healthy run look
broken and buries the messages that do matter.

The patch wraps `_call_connection_lost` and swallows exactly
ConnectionResetError from exactly that call. Everything else — every other
exception type, every other code path — propagates untouched, so this
cannot hide a real failure. It is a no-op on non-Windows platforms and on
any future Python where the private attribute has been renamed.

Call `install()` once, as early as possible, from any entrypoint that runs
an event loop on Windows (streamlit_app.py, src/api/main.py).
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_installed = False


def install() -> bool:
    """Returns True if the patch was applied, False if it wasn't needed.

    Safe to call more than once; only the first call does anything."""
    global _installed  # noqa: PLW0603
    if _installed or not sys.platform.startswith("win"):
        return False

    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport  # noqa: WPS433, PLC2701
    except Exception:  # noqa: BLE001 - not Windows, or the internals moved
        return False

    original = getattr(_ProactorBasePipeTransport, "_call_connection_lost", None)
    if original is None:  # pragma: no cover - private API gone in a future Python
        return False

    def _quiet_call_connection_lost(self, exc):  # noqa: ANN001, ANN202
        try:
            return original(self, exc)
        except ConnectionResetError:
            # The peer closed first; the socket is being discarded anyway.
            # Deliberately narrow: only this exception type, only from this
            # one teardown call.
            return None

    _quiet_call_connection_lost.__wrapped__ = original  # type: ignore[attr-defined]
    _ProactorBasePipeTransport._call_connection_lost = _quiet_call_connection_lost  # noqa: SLF001
    _installed = True
    logger.debug("Windows asyncio ConnectionResetError noise suppressed.")
    return True
