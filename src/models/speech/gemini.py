"""Gemini text-to-speech over plain HTTPS.

The model speaks through the Interactions API (`/v1beta/interactions`), not
generateContent, so this is one POST with an API key and no SDK. The request
shape is the one the D.1.1 spike used for the working narration.

Errors are named for what to do next, as in src/models/image_gen:

    401/403, or 400 "API key not valid"   SpeechAuthError       every line would fail alike
    429                                   SpeechQuotaError      not retried; the next provider speaks
    5xx, timeout, dropped connection      SpeechTransportError  retried (agents.base.with_retries)

Audio comes back as WAV. If a model version answers with raw 16-bit PCM
instead, it is wrapped as WAV here, so everything downstream sees one format.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from src.agents.base import with_retries
from src.models.speech.base import (
    Audio, SpeechAuthError, SpeechError, SpeechProvider, SpeechQuotaError, SpeechTransportError, is_wav,
    pcm_to_wav, wav_duration,
)

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10.0
_RATE = re.compile(r"rate=(\d+)")


class GeminiSpeechProvider(SpeechProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, default_voice: str, url: str, style: str = "",
                 sample_rate: int = 24000, timeout_s: float = 180.0, max_retries: int = 3,
                 post: Optional[Callable[..., object]] = None):
        self._api_key = (api_key or "").strip()
        self.model_id = (model or "").strip()
        self._default_voice = (default_voice or "").strip()
        self._url = url
        self._style = style
        self._sample_rate = sample_rate
        self._timeout_s = timeout_s
        self._max_retries = max(1, max_retries)
        self._post = post          # injectable for tests

    def is_available(self) -> bool:
        return bool(self._api_key and self.model_id)

    def unavailable_reason(self) -> str:
        if not self._api_key:
            return "GEMINI_API_KEY is not set"
        return "" if self.model_id else "GEMINI_TTS_MODEL is not set"

    # -- one call -----------------------------------------------------------

    def _body(self, text: str, voice: str) -> dict:
        content: dict = {"type": "text", "text": text}
        if self._style:
            content["annotations"] = [{"type": "speech_metadata", "style": self._style}]
        return {
            "model": self.model_id,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav", "sample_rate": self._sample_rate},
            "generation_config": {"speech_config": [{"voice": voice}]},
        }

    def _request(self, body: dict):
        post = self._post
        if post is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise SpeechError("requests is not installed; run `pip install -r requirements.txt`") from exc
            post = requests.post
        try:
            return post(self._url, headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                        json=body, timeout=(CONNECT_TIMEOUT, self._timeout_s))
        except SpeechError:
            raise
        except Exception as exc:  # noqa: BLE001 - an unrecognised exception from an HTTP call is a failed call
            raise SpeechTransportError(
                f"the speech service could not be reached or did not answer in time ({type(exc).__name__})"
            ) from exc

    def _audio_bytes(self, response) -> bytes:
        try:
            steps = response.json().get("steps", [])
        except Exception as exc:  # noqa: BLE001
            raise SpeechError("the speech service answered with something that is not JSON") from exc
        blocks = [c for s in steps if isinstance(s, dict) and s.get("type") == "model_output"
                  for c in s.get("content", []) if isinstance(c, dict) and c.get("type") == "audio" and c.get("data")]
        if not blocks:
            raise SpeechError("the speech service answered without any audio")
        raw = base64.b64decode(blocks[-1]["data"])
        if is_wav(raw):
            return raw
        rate = _RATE.search(str(blocks[-1].get("mime_type", "")))
        return pcm_to_wav(raw, int(rate.group(1)) if rate else self._sample_rate)

    def _once(self, text: str, voice: str) -> bytes:
        response = self._request(self._body(text, voice))
        status = getattr(response, "status_code", 0)
        if status == 200:
            return self._audio_bytes(response)
        detail = (getattr(response, "text", "") or "")[:200]
        if status in (401, 403) or (status == 400 and "api key" in detail.lower()):
            raise SpeechAuthError(f"Google rejected the key ({status}). Check GEMINI_API_KEY.")
        if status == 429:
            raise SpeechQuotaError("Gemini speech allowance or rate limit reached (429).")
        if status in (500, 502, 503, 504):
            raise SpeechTransportError(f"the speech service is busy (HTTP {status})")
        raise SpeechError(f"the speech service refused the request (HTTP {status}): {detail}")

    # -- the provider contract ------------------------------------------------

    def synthesize(self, text: str, voice: Optional[str], output_path: Path) -> Audio:
        if not self.is_available():
            raise SpeechAuthError(self.unavailable_reason())
        voice = voice or self._default_voice
        data = with_retries(lambda: self._once(text, voice), max_retries=self._max_retries)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        return Audio(path=output_path, duration_s=wav_duration(output_path), provider=self.name, voice=voice,
                     model_id=self.model_id)
