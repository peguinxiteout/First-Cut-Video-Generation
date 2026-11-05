"""Beatoven.ai music over its public API.

The API is asynchronous, as in Beatoven's own spec (github.com/Beatoven/public-api,
docs/api-spec.md, read 4 Nov 2026):

    POST /api/v1/tracks/compose    {"prompt": {"text": ...}, "format": "wav"}   -> {"status": "started", "task_id": ...}
    GET  /api/v1/tasks/<task_id>   composing | running | composed
    composed: {"meta": {"project_id", "track_id", "version", "track_url", "stems_url", "prompt": {"text"}}}

with the API token as a Bearer header. The API has no length field: the length is
written into the prompt ("a 16-second instrumental track"), and the track is cut to
the video's length afterwards (MusicService._fit). It returns no price and no licence
terms either, so the cost and the licence note come from music.yaml and the record says
so. Failure statuses and error bodies are not documented; any status other than the
three above, and any error response, is treated as a failed request.

Making a track costs the account something, so the compose call is made ONCE: a dropped
connection is not retried, because the track may have been started. Polling and the
download are retried.

    401/403                                   MusicAuthError
    402, 429, or a body naming credits/plan   MusicQuotaError       not retried
    5xx, timeout, dropped connection          MusicTransportError
"""
from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from src.agents.base import with_retries
from src.core.exceptions import RenderingError
from src.models.music.base import (
    MusicAuthError, MusicError, MusicProvider, MusicQuotaError, MusicRequest, MusicTrack, MusicTransportError,
)
from src.models.speech.base import wav_duration
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10.0
_NEEDS_PAYMENT = ("credit", "quota", "plan", "subscription", "limit", "balance", "upgrade", "payment")
_WAITING = ("composing", "running", "started", "queued", "pending")


class BeatovenProvider(MusicProvider):
    name = "beatoven"

    def __init__(self, api_key: str, base_url: str, model_id: str = "beatoven-compose", extra_s: float = 1.0,
                 cost_usd: float = 0.0, licence_note: str = "", poll_s: float = 5.0, poll_timeout_s: float = 300.0,
                 timeout_s: float = 60.0, max_retries: int = 3,
                 post: Optional[Callable[..., object]] = None, get: Optional[Callable[..., object]] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self._api_key = (api_key or "").strip()
        self._base = base_url.rstrip("/")
        self.model_id = model_id
        self._extra_s = extra_s
        self._cost_usd = cost_usd
        self._licence_note = licence_note
        self._poll_s = poll_s
        self._poll_timeout_s = poll_timeout_s
        self._timeout_s = timeout_s
        self._max_retries = max(1, max_retries)
        self._post, self._get = post, get          # injectable for tests
        self._sleep, self._clock = sleep, clock

    def is_available(self) -> bool:
        return bool(self._api_key and self._base)

    def unavailable_reason(self) -> str:
        if not self._api_key:
            return "BEATOVEN_API_KEY is not set"
        return "" if self._base else "no Beatoven url is configured"

    # -- what is asked ---------------------------------------------------------

    def length_s(self, duration_s: float) -> int:
        """The length asked for: the video's to the nearest second, with a little over so the cut is exact."""
        return int(round(duration_s) + math.ceil(self._extra_s))

    def prompt_text(self, request: MusicRequest) -> str:
        return f"A {self.length_s(request.duration_s)}-second instrumental track. {request.prompt}"

    def cache_key_parts(self, request: MusicRequest) -> dict:
        return {"model": self.model_id, "prompt": self.prompt_text(request)}

    # -- http -----------------------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _http(method: str, injected):
        if injected is not None:
            return injected
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise MusicError("requests is not installed; run `pip install -r requirements.txt`") from exc
        return getattr(requests, method)

    def _call(self, method: str, url: str, **kwargs):
        send = self._http(method, self._post if method == "post" else self._get)
        try:
            return send(url, timeout=(CONNECT_TIMEOUT, self._timeout_s), **kwargs)
        except MusicError:
            raise
        except Exception as exc:  # noqa: BLE001 - an unrecognised exception from an HTTP call is a failed call
            raise MusicTransportError(
                f"the music service could not be reached or did not answer in time ({type(exc).__name__})") from exc

    def _check(self, response) -> dict:
        """The JSON of a successful answer; the right error for any other."""
        status = getattr(response, "status_code", 0)
        if status == 200:
            try:
                return response.json()
            except Exception as exc:  # noqa: BLE001
                raise MusicError("the music service answered with something that is not JSON") from exc
        detail = (getattr(response, "text", "") or "")[:300]
        low = detail.lower()
        if status in (401, 403):
            raise MusicAuthError(f"Beatoven rejected the key ({status}). Check BEATOVEN_API_KEY.")
        if status in (402, 429) or (status in (400, 422) and any(word in low for word in _NEEDS_PAYMENT)):
            raise MusicQuotaError(f"Beatoven's allowance is spent, or the plan does not cover the API "
                                  f"(HTTP {status}): {detail[:160]}")
        if status in (500, 502, 503, 504):
            raise MusicTransportError(f"the music service is busy (HTTP {status})")
        raise MusicError(f"the music service refused the request (HTTP {status}): {detail[:160]}")

    # -- the three steps ------------------------------------------------------------

    def _compose(self, text: str) -> str:
        body = {"prompt": {"text": text}, "format": "wav"}
        data = self._check(self._call("post", f"{self._base}/api/v1/tracks/compose", headers=self._headers(), json=body))
        task_id = data.get("task_id")
        if not task_id:
            raise MusicError("the music service did not name the task it started")
        return str(task_id)

    def _poll_once(self, task_id: str) -> dict:
        return self._check(self._call("get", f"{self._base}/api/v1/tasks/{task_id}", headers=self._headers()))

    def _wait(self, task_id: str) -> dict:
        """The `meta` of the finished task."""
        started = self._clock()
        while True:
            data = with_retries(lambda: self._poll_once(task_id), max_retries=self._max_retries)
            status = str(data.get("status", "")).lower()
            if status == "composed":
                meta = data.get("meta") or {}
                if not meta.get("track_url"):
                    raise MusicError("the music service finished the task without a track to download")
                return meta
            if status not in _WAITING:
                raise MusicError(f"the music task ended with status '{status or 'unknown'}'")
            if self._clock() - started > self._poll_timeout_s:
                raise MusicTransportError(f"the music task was not done after {self._poll_timeout_s:.0f} s")
            self._sleep(self._poll_s)

    def _download(self, url: str) -> bytes:
        response = self._call("get", url)
        status = getattr(response, "status_code", 0)
        if status != 200:
            raise (MusicTransportError if status >= 500 else MusicError)(f"the track could not be downloaded (HTTP {status})")
        return response.content

    # -- the provider contract ------------------------------------------------------

    def generate(self, request: MusicRequest, dest: Path) -> MusicTrack:
        if not self.is_available():
            raise MusicAuthError(self.unavailable_reason())
        text = self.prompt_text(request)
        task_id = self._compose(text)
        meta = self._wait(task_id)
        data = with_retries(lambda: self._download(meta["track_url"]), max_retries=self._max_retries)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "beatoven_track"
            raw.write_bytes(data)
            try:
                ffmpeg.run(["-i", str(raw), "-vn", "-ar", "44100", "-ac", "2", str(dest)])
            except RenderingError as exc:
                raise MusicError(f"the music service's audio could not be read: {exc}") from exc
        notes = [f"Beatoven project {meta.get('project_id', '?')}, version {meta.get('version', '?')}, task {task_id}",
                 "Beatoven's API returns no licence terms and no price"
                 + (": the licence line is the project's note, not the API's" if self._licence_note else "")]
        return MusicTrack(path=dest, provider=self.name, model_id=self.model_id, prompt=text,
                          duration_s=wav_duration(dest), cost_usd=self._cost_usd,
                          track_id=str(meta.get("track_id", "")), licence=self._licence_note, notes=notes)
