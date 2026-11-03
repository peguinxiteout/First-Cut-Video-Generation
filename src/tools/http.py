"""The one place this project talks to the open internet.

Every research tool goes through here, which buys three things that would
otherwise have to be remembered in nine different modules:

A KILL SWITCH. `TOOLS_ENABLED=false` (or no network at all) makes every
tool report itself unavailable instead of hanging. The test suite sets it,
so no test can reach the network by accident — a research tool that
quietly called a live API during tests would make the suite slow, flaky,
and dependent on somebody else's uptime.

A HONEST USER-AGENT. OpenStreetMap's Nominatim and Overpass are run on
donated hardware and their usage policy requires an identifying
User-Agent; sending a default `python-requests/2.x` is how a project gets
its IP blocked for everyone. Wikipedia asks for the same.

A DEADLINE ON EVERYTHING. Each call has a timeout and a small retry
budget for the transient statuses only. Nothing here retries a 404 or a
403 — those are answers, not failures.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

USER_AGENT = (
    "AgenticContentPlatform/1.0 (real-estate content research; "
    "+https://github.com/ -- contact via repository)"
)

# Worth trying again; anything else is the server's answer.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class HttpError(Exception):
    """A request that could not be completed. Callers turn this into a
    ToolResult.failed() — it never escapes the tools layer."""


class HttpClient:
    """A thin wrapper over `requests` with the policy above baked in."""

    def __init__(self, settings=None, timeout_s: float = 12.0, max_retries: int = 2,
                 get: Optional[Callable[..., Any]] = None,
                 post: Optional[Callable[..., Any]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        if settings is None:
            from src.core.config import settings as _settings
            settings = _settings
        self._settings = settings
        self._timeout_s = timeout_s
        self._max_retries = max(1, max_retries)
        self._get = get                 # injectable for tests
        self._post = post
        self._sleep = sleep

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings, "tools_enabled", True))

    def _session_get(self):
        if self._get is not None:
            return self._get
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise HttpError("requests is not installed") from exc
        return requests.get

    def get(self, url: str, *, params: Optional[dict] = None,
            headers: Optional[dict] = None, timeout_s: Optional[float] = None) -> Any:
        """A GET, or HttpError. Never returns a failed response object."""
        if not self.enabled:
            raise HttpError("external tools are turned off (TOOLS_ENABLED=false)")

        request_headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
        request_headers.update(headers or {})
        last: str = "no attempt made"

        for attempt in range(self._max_retries):
            try:
                response = self._session_get()(
                    url, params=params, headers=request_headers,
                    timeout=timeout_s or self._timeout_s)
            except Exception as exc:  # noqa: BLE001 - transport errors of any kind
                last = f"{type(exc).__name__}: {exc}"
            else:
                status = getattr(response, "status_code", 0)
                if 200 <= status < 300:
                    return response
                last = f"HTTP {status}"
                if status not in RETRY_STATUSES:
                    break
            if attempt < self._max_retries - 1:
                self._sleep(1.5 * (attempt + 1) + random.uniform(0, 0.5))

        raise HttpError(f"{url}: {last}")

    def get_json(self, url: str, **kwargs) -> Any:
        response = self.get(url, **kwargs)
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise HttpError(f"{url}: response was not JSON") from exc

    def get_text(self, url: str, **kwargs) -> str:
        return self.get(url, **kwargs).text

    def post_json(self, url: str, *, json: Optional[dict] = None,
                  headers: Optional[dict] = None, timeout_s: Optional[float] = None) -> Any:
        """A POST for the few tools that need one (Tavily). Deliberately
        not retried: a POST is not guaranteed safe to repeat, and none of
        the endpoints here promise idempotency."""
        if not self.enabled:
            raise HttpError("external tools are turned off (TOOLS_ENABLED=false)")
        request_headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
        request_headers.update(headers or {})
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise HttpError("requests is not installed") from exc
        post = self._post or requests.post
        try:
            response = post(url, json=json, headers=request_headers,
                            timeout=timeout_s or self._timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise HttpError(f"{url}: {type(exc).__name__}: {exc}") from exc
        status = getattr(response, "status_code", 0)
        if not 200 <= status < 300:
            raise HttpError(f"{url}: HTTP {status}")
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise HttpError(f"{url}: response was not JSON") from exc
