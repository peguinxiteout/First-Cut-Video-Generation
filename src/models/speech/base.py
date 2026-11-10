"""Provider-agnostic speech interface.

Architecture:  voice node -> SpeechService -> SpeechProvider -> a voice.
Nothing outside src/models/speech knows which engine spoke a line. Audio is
always a WAV file, so every provider's output goes through the same
loudness and mixing steps.
"""
from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.exceptions import ModelProviderError, PlatformError


class SpeechError(PlatformError):
    """A provider could not speak this line. The service moves to the next
    provider; the reason is kept on the Audio that finally answers."""


class SpeechAuthError(SpeechError):
    """Key missing or rejected. Every line would fail the same way."""


class SpeechQuotaError(SpeechError):
    """The allowance (or the rate limit) is spent. Not retried: waiting
    seconds does not bring a daily budget back."""


class SpeechTransportError(SpeechError, ModelProviderError):
    """No answer: a timeout, a dropped connection, a 5xx. Also a
    ModelProviderError so agents.base.with_retries retries it, and ONLY this
    kind: auth and quota are not worth a second attempt."""


@dataclass
class Audio:
    path: Path
    duration_s: float
    provider: str
    mode: str = "speech"                 # speech | silent
    from_cache: bool = False
    voice: str = ""
    model_id: str = ""
    cost_usd: float = 0.0                # what this line cost to make; 0 when it came from the cache
    spoken: str = ""                     # the text this provider was given, after respelling
    # Why a provider ahead of this one did not speak, in order.
    notes: list[str] = field(default_factory=list)
    # The providers that were tried and FAILED on this line, as (name, reason). A provider that
    # is merely unavailable (no key, no engine) is a note, not a failure.
    failed: list[tuple[str, str]] = field(default_factory=list)


class SpeechProvider(ABC):
    name: str = "base"
    model_id: str = ""
    # False for a provider whose output costs nothing to make again.
    cacheable: bool = True

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else f"{self.name} is not available"

    def phoneme_words(self) -> frozenset[str]:
        """Lower-case words this provider says from a phonetic spelling of its own, so the
        service leaves them as written instead of respelling them."""
        return frozenset()

    def cache_salt(self, text: str) -> str:
        """What else, besides the text and voice, changes the audio this provider makes."""
        return ""

    @abstractmethod
    def synthesize(self, text: str, voice: Optional[str], output_path: Path) -> Audio:
        """Speak `text` and write a WAV to output_path. `voice` is what this
        provider was asked for in configs/video/speech.yaml, or None for its
        own default."""


# ---------------------------------------------------------------------------
# WAV helpers (standard library only)
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len(text.split())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return round(handle.getnframes() / float(handle.getframerate() or 1), 3)


def pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Raw 16-bit little-endian PCM wrapped as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def is_wav(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"
