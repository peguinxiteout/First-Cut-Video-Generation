"""Lyria music over the Gemini API.

Like the speech model, Lyria answers through the Interactions API
(`/v1beta/interactions`): one POST, a model id and a text prompt, and the track
comes back as base64 audio in the `model_output` step (MP3 by default; it is
turned into a WAV here). Request and response follow Google's music-generation
guide as it stood on 4 Nov 2026:

    lyria-3-clip-preview   30 s clip            $0.04 a track, no free tier
    lyria-3.5              full song, minutes   $0.08 a track, no free tier

so the model is chosen by the video's length (configs/video/music.yaml). A key
without billing gets a refusal, which is a quota error here: the service goes on
to the library, or to no music, and records why. Every track carries Google's
SynthID watermark.

    401/403, or 400 "API key not valid"      MusicAuthError
    429, or a refusal naming billing/tier    MusicQuotaError       not retried
    5xx, timeout, dropped connection         MusicTransportError   retried
"""
from __future__ import annotations

import base64
import logging
import tempfile
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
_NEEDS_PAYMENT = ("billing", "free tier", "free_tier", "paid", "not available", "quota", "exhausted")


class LyriaProvider(MusicProvider):
    name = "lyria"

    def __init__(self, api_key: str, url: str, models: list[dict], api_revision: str = "",
                 timeout_s: float = 240.0, max_retries: int = 3, post: Optional[Callable[..., object]] = None):
        self._api_key = (api_key or "").strip()
        self._url = url
        self._models = sorted(models, key=lambda m: m["max_video_s"])
        self._api_revision = api_revision
        self._timeout_s = timeout_s
        self._max_retries = max(1, max_retries)
        self._post = post          # injectable for tests
        self.model_id = self._models[0]["id"] if self._models else ""

    def is_available(self) -> bool:
        return bool(self._api_key and self._models and self._url)

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        return "GEMINI_API_KEY is not set" if not self._api_key else "no Lyria model is configured"

    def _model_for(self, duration_s: float) -> dict:
        return next((m for m in self._models if m["max_video_s"] >= duration_s), self._models[-1])

    def cache_key_parts(self, request: MusicRequest) -> dict:
        return {"model": self._model_for(request.duration_s)["id"], "prompt": request.prompt}

    # -- one call -------------------------------------------------------------

    def _request(self, body: dict):
        post = self._post
        if post is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise MusicError("requests is not installed; run `pip install -r requirements.txt`") from exc
            post = requests.post
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        if self._api_revision:
            headers["Api-Revision"] = self._api_revision
        try:
            return post(self._url, headers=headers, json=body, timeout=(CONNECT_TIMEOUT, self._timeout_s))
        except MusicError:
            raise
        except Exception as exc:  # noqa: BLE001 - an unrecognised exception from an HTTP call is a failed call
            raise MusicTransportError(
                f"the music service could not be reached or did not answer in time ({type(exc).__name__})") from exc

    @staticmethod
    def _audio_bytes(response) -> bytes:
        try:
            steps = response.json().get("steps", [])
        except Exception as exc:  # noqa: BLE001
            raise MusicError("the music service answered with something that is not JSON") from exc
        blocks = [c for s in steps if isinstance(s, dict) and s.get("type") == "model_output"
                  for c in s.get("content", []) if isinstance(c, dict) and c.get("type") == "audio" and c.get("data")]
        if not blocks:
            raise MusicError("the music service answered without any audio")
        return base64.b64decode(blocks[-1]["data"])

    def _once(self, model_id: str, prompt: str) -> bytes:
        response = self._request({"model": model_id, "input": prompt, "response_format": {"type": "audio"}})
        status = getattr(response, "status_code", 0)
        if status == 200:
            return self._audio_bytes(response)
        detail = (getattr(response, "text", "") or "")[:300]
        low = detail.lower()
        needs_payment = any(word in low for word in _NEEDS_PAYMENT)
        if status == 429 or (status in (400, 402, 403) and needs_payment):
            raise MusicQuotaError(f"Lyria is not available on this key's plan, or its allowance is spent "
                                  f"(HTTP {status}): {detail[:160]}")
        if status in (401, 403) or (status == 400 and "api key" in low):
            raise MusicAuthError(f"Google rejected the key ({status}). Check GEMINI_API_KEY.")
        if status in (500, 502, 503, 504):
            raise MusicTransportError(f"the music service is busy (HTTP {status})")
        raise MusicError(f"the music service refused the request (HTTP {status}): {detail[:160]}")

    # -- the provider contract --------------------------------------------------

    def generate(self, request: MusicRequest, dest: Path) -> MusicTrack:
        if not self.is_available():
            raise MusicAuthError(self.unavailable_reason())
        model = self._model_for(request.duration_s)
        data = with_retries(lambda: self._once(model["id"], request.prompt), max_retries=self._max_retries)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "lyria_track"
            raw.write_bytes(data)
            try:
                ffmpeg.run(["-i", str(raw), "-vn", "-ar", "44100", "-ac", "2", str(dest)])
            except RenderingError as exc:
                raise MusicError(f"the music service's audio could not be read: {exc}") from exc
        return MusicTrack(path=dest, provider=self.name, model_id=model["id"], prompt=request.prompt,
                          duration_s=wav_duration(dest), cost_usd=float(model.get("cost_usd", 0.0)))
