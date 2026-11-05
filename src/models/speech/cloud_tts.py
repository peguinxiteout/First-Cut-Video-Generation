"""Google Cloud Text-to-Speech (Chirp 3: HD) over plain HTTPS.

One POST to `text:synthesize` with an API key; no SDK. Request and response follow
Google's reference as it stood on 4 Nov 2026:

    POST https://texttospeech.googleapis.com/v1/text:synthesize
    {"input": {"text": ..., "customPronunciations": {"pronunciations": [
                  {"phrase": ..., "phoneticEncoding": "PHONETIC_ENCODING_IPA", "pronunciation": ...}]}},
     "voice": {"languageCode": "en-IN", "name": "en-IN-Chirp3-HD-Sulafat"},
     "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.0}}
    -> {"audioContent": <base64 of a WAV>}

A request holds at most 5000 bytes of text. The voice name is
`<locale>-Chirp3-HD-<persona>`; configs/video/speech.yaml says which.

PRONUNCIATION. A phrase found in the text that speech.yaml gives an IPA spelling for
(`cloud_tts.phonemes`) is sent as a custom pronunciation, which the voice follows
exactly. Google's reference says custom pronunciations started with en-US and the
Chirp 3 page lists en-IN as covered; if the service refuses them, the line is spoken
again with the plain-text respelling instead and the reason is kept on the audio.

    401/403, or 400 "API key not valid"   SpeechAuthError       every line would fail alike
    429                                   SpeechQuotaError      not retried; the next provider speaks
    5xx, timeout, dropped connection      SpeechTransportError  retried (agents.base.with_retries)
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
from src.models.speech.pronunciation import respell

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10.0
MAX_TEXT_BYTES = 5000
IPA = "PHONETIC_ENCODING_IPA"


def _error_text(response) -> str:
    """Google's own message from an error body, else the body."""
    text = getattr(response, "text", "") or ""
    try:
        return str(response.json()["error"]["message"])
    except Exception:  # noqa: BLE001 - an error body that is not Google's JSON is shown as it came
        return text


class _PronunciationRefused(SpeechError):
    """The service took the line but not its custom pronunciations."""


class CloudTTSProvider(SpeechProvider):
    name = "cloud_tts"

    def __init__(self, api_key: str, url: str, model: str = "Chirp3-HD", language: str = "en-IN",
                 default_voice: str = "Sulafat", speaking_rate: float = 1.0, sample_rate: int = 24000,
                 phonemes: Optional[dict[str, str]] = None, respellings: Optional[dict[str, str]] = None,
                 usd_per_million_chars: float = 0.0, timeout_s: float = 60.0, max_retries: int = 3,
                 post: Optional[Callable[..., object]] = None):
        self._api_key = (api_key or "").strip()
        self._url = url
        self.model_id = f"{language}-{model}"
        self._language = language
        self._model = model
        self._default_voice = default_voice
        self._speaking_rate = speaking_rate
        self._sample_rate = sample_rate
        self._phonemes = {w.strip(): p.strip() for w, p in (phonemes or {}).items() if w.strip() and str(p).strip()}
        self._respellings = respellings or {}          # the plain-text map, for when phonemes are refused
        self._usd_per_million = usd_per_million_chars
        self._timeout_s = timeout_s
        self._max_retries = max(1, max_retries)
        self._post = post          # injectable for tests
        self._phonemes_refused = ""

    def is_available(self) -> bool:
        return bool(self._api_key and self._url)

    def unavailable_reason(self) -> str:
        if not self._api_key:
            return "GOOGLE_CLOUD_TTS_API_KEY is not set"
        return "" if self._url else "no Cloud TTS url is configured"

    # -- pronunciation --------------------------------------------------------

    def phoneme_words(self) -> frozenset[str]:
        return frozenset(w.lower() for w in self._phonemes) if not self._phonemes_refused else frozenset()

    def _pronunciations(self, text: str) -> list[dict]:
        """The phrases of `text` that have an IPA spelling, as the API wants them. Whole words
        only, as in respell; each phrase once."""
        if self._phonemes_refused:
            return []
        found = []
        for word, ipa in sorted(self._phonemes.items()):
            if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text):
                found.append({"phrase": word, "phoneticEncoding": IPA, "pronunciation": ipa})
        return found

    def cache_salt(self, text: str) -> str:
        return "|".join(f"{p['phrase']}={p['pronunciation']}" for p in self._pronunciations(text))

    # -- one call -----------------------------------------------------------

    def _voice_name(self, voice: str) -> str:
        voice = voice or self._default_voice
        return voice if "-" in voice else f"{self._language}-{self._model}-{voice}"

    def _body(self, text: str, voice: str, pronunciations: list[dict]) -> dict:
        source: dict = {"text": text}
        if pronunciations:
            source["customPronunciations"] = {"pronunciations": pronunciations}
        return {"input": source,
                "voice": {"languageCode": self._language, "name": self._voice_name(voice)},
                "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": self._speaking_rate}}

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
            content = response.json().get("audioContent")
        except Exception as exc:  # noqa: BLE001
            raise SpeechError("the speech service answered with something that is not JSON") from exc
        if not content:
            raise SpeechError("the speech service answered without any audio")
        raw = base64.b64decode(content)
        return raw if is_wav(raw) else pcm_to_wav(raw, self._sample_rate)

    def _once(self, text: str, voice: str, pronunciations: list[dict]) -> bytes:
        response = self._request(self._body(text, voice, pronunciations))
        status = getattr(response, "status_code", 0)
        if status == 200:
            return self._audio_bytes(response)
        detail = _error_text(response)[:300]
        low = detail.lower()
        if status in (401, 403) or (status == 400 and ("api key" in low or "api_key" in low)):
            raise SpeechAuthError(f"Google rejected the key ({status}). Check GOOGLE_CLOUD_TTS_API_KEY and that "
                                  "the Cloud Text-to-Speech API is enabled for its project.")
        if status == 429:
            raise SpeechQuotaError("Cloud Text-to-Speech quota or rate limit reached (429).")
        if status in (500, 502, 503, 504):
            raise SpeechTransportError(f"the speech service is busy (HTTP {status})")
        if status == 400 and pronunciations and ("pronunciation" in low or "phonetic" in low):
            raise _PronunciationRefused(detail[:160])
        raise SpeechError(f"the speech service refused the request (HTTP {status}): {detail[:160]}")

    # -- the provider contract ------------------------------------------------

    def synthesize(self, text: str, voice: Optional[str], output_path: Path) -> Audio:
        if not self.is_available():
            raise SpeechAuthError(self.unavailable_reason())
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise SpeechError(f"the line is longer than Cloud Text-to-Speech takes ({MAX_TEXT_BYTES} bytes)")
        voice = voice or self._default_voice
        notes: list[str] = []
        pronunciations = self._pronunciations(text)
        try:
            data = with_retries(lambda: self._once(text, voice, pronunciations), max_retries=self._max_retries)
        except _PronunciationRefused as exc:
            self._phonemes_refused = str(exc)
            logger.warning("Cloud TTS refused custom pronunciations, using the plain-text respelling: %s", exc)
            notes.append(f"custom pronunciations were refused ({exc}); the names were respelled in plain text")
            phrases = {p["phrase"].lower() for p in pronunciations}
            text = respell(text, {w: s for w, s in self._respellings.items() if w.lower() in phrases})
            data = with_retries(lambda: self._once(text, voice, []), max_retries=self._max_retries)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        cost = round(len(text) * self._usd_per_million / 1_000_000, 6)
        return Audio(path=output_path, duration_s=wav_duration(output_path), provider=self.name,
                     voice=self._voice_name(voice), model_id=self.model_id, cost_usd=cost, notes=notes)
