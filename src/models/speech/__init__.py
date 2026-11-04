"""Speech service.

    from src.models.speech import get_speech_service
    audio = get_speech_service().synthesize("Welcome home.", "", dest_path)

The chain comes from configs/video/speech.yaml (gemini, then a voice on this
machine, then a silent track). get_speech_service() never raises for want of
a key or an engine: the silent track is always the last link, and the Audio
that comes back says which provider spoke, whether it was a voice at all
(mode), and why the providers ahead of it did not (notes).

One service is one RUN: a provider that answers with an auth or quota error
is skipped for the rest of it, so later lines do not each wait for the same
refusal. Other failures cost that line only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

from src.core.config import load_yaml_config
from src.core.exceptions import ConfigError
from src.models.speech.base import (
    Audio, SpeechAuthError, SpeechError, SpeechProvider, SpeechQuotaError, SpeechTransportError,
)
from src.models.speech.gemini import GeminiSpeechProvider
from src.models.speech.mock import MockSpeechProvider
from src.models.speech.offline import OfflineSpeechProvider
from src.models.speech.silent import SilentSpeechProvider

logger = logging.getLogger(__name__)

__all__ = [
    "Audio", "SpeechAuthError", "SpeechError", "SpeechProvider", "SpeechQuotaError", "SpeechService",
    "SpeechTransportError", "get_speech_service",
]

CONFIG_FILE = "video/speech.yaml"


class SpeechService:
    def __init__(self, providers: Sequence[SpeechProvider], cache_dir: Path, voices: Optional[dict] = None,
                 default_voice: str = "default", words_per_second: float = 2.1):
        chain = list(providers)
        if not any(isinstance(p, SilentSpeechProvider) for p in chain):
            chain.append(SilentSpeechProvider(words_per_second))
        self.providers = chain
        self.cache_dir = cache_dir
        self.voices = voices or {}
        self.default_voice = default_voice
        self._dropped: dict[str, str] = {}

    @property
    def dropped(self) -> dict[str, str]:
        return dict(self._dropped)

    # -- cache ----------------------------------------------------------------

    @staticmethod
    def _key(provider: SpeechProvider, text: str, voice: Optional[str]) -> str:
        payload = json.dumps({"provider": provider.name, "model": provider.model_id, "voice": voice or "",
                              "text": text}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _lookup(self, key: str) -> Optional[Audio]:
        wav, meta_path = self.cache_dir / f"{key}.wav", self.cache_dir / f"{key}.json"
        if not (wav.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return Audio(path=wav, duration_s=meta["duration_s"], provider=meta["provider"],
                         voice=meta.get("voice", ""), model_id=meta.get("model_id", ""), from_cache=True)
        except (OSError, ValueError, KeyError):
            logger.warning("Ignoring an unreadable speech cache entry: %s", key)
            return None

    def _store(self, key: str, audio: Audio) -> None:
        meta = {"provider": audio.provider, "voice": audio.voice, "model_id": audio.model_id,
                "duration_s": audio.duration_s}
        (self.cache_dir / f"{key}.json").write_text(json.dumps(meta), encoding="utf-8")

    # -- one line ---------------------------------------------------------------

    def _voice_entry(self, voice: str, notes: list[str]) -> dict:
        name = voice or self.default_voice
        if name in self.voices:
            return self.voices[name] or {}
        if voice:
            notes.append(f"unknown voice '{voice}'; using '{self.default_voice}'")
        return self.voices.get(self.default_voice) or {}

    def synthesize(self, text: str, voice: str, output_path: Path) -> Audio:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = (text or "").strip()
        notes: list[str] = []
        entry = self._voice_entry(voice, notes)
        for provider in self.providers:
            wanted = entry.get(provider.name)
            if provider.name in self._dropped:
                notes.append(f"{provider.name}: skipped for this run ({self._dropped[provider.name]})")
                continue
            key = self._key(provider, text, wanted) if provider.cacheable else ""
            if key:
                cached = self._lookup(key)
                if cached is not None:
                    shutil.copyfile(cached.path, output_path)
                    cached.path, cached.notes = output_path, notes
                    return cached
            if not provider.is_available():
                notes.append(f"{provider.name}: {provider.unavailable_reason()}")
                continue
            try:
                if key:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    audio = provider.synthesize(text, wanted, self.cache_dir / f"{key}.wav")
                    self._store(key, audio)
                    shutil.copyfile(audio.path, output_path)
                    audio.path = output_path
                else:
                    audio = provider.synthesize(text, wanted, output_path)
            except (SpeechAuthError, SpeechQuotaError) as exc:
                logger.warning("%s dropped for the rest of this run: %s", provider.name, exc)
                self._dropped[provider.name] = str(exc)
                notes.append(f"{provider.name}: {exc}")
                continue
            except SpeechError as exc:
                logger.warning("%s could not speak this line, trying the next provider: %s", provider.name, exc)
                notes.append(f"{provider.name}: {exc}")
                continue
            audio.notes = notes
            return audio
        raise SpeechError("no speech provider could make this line: " + "; ".join(notes))


def _build(name: str, settings, cfg: dict) -> SpeechProvider:
    if name == "gemini":
        g = cfg.get("gemini") or {}
        return GeminiSpeechProvider(
            api_key=settings.gemini_api_key, model=settings.gemini_tts_model, default_voice=settings.gemini_tts_voice,
            url=g.get("url", ""), style=g.get("style", ""), sample_rate=int(g.get("sample_rate", 24000)))
    if name == "offline":
        return OfflineSpeechProvider()
    if name == "silent":
        return SilentSpeechProvider(float(cfg.get("words_per_second", 2.1)))
    if name == "mock":
        return MockSpeechProvider()
    raise ConfigError(f"Unknown speech provider '{name}' (configs/{CONFIG_FILE} or SPEECH_PROVIDER). "
                      "Use gemini, offline, silent or mock.")


def get_speech_service(settings=None) -> SpeechService:
    """The service for SPEECH_PROVIDER: "auto" is the configured chain, a name
    is that provider and then silence. The cache folder is worked out now,
    not at import."""
    if settings is None:
        from src.core.config import settings as _settings

        settings = _settings
    cfg = load_yaml_config(CONFIG_FILE)
    choice = (settings.speech_provider or "auto").strip().lower()
    names = list(cfg.get("chain") or ["gemini", "offline", "silent"]) if choice == "auto" else [choice]
    providers = [_build(n, settings, cfg) for n in dict.fromkeys(names)]
    return SpeechService(providers, settings.data_dir_resolved / "speech_cache", voices=cfg.get("voices"),
                         default_voice=cfg.get("default_voice", "default"),
                         words_per_second=float(cfg.get("words_per_second", 2.1)))
