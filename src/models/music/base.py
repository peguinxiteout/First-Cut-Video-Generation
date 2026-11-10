"""Provider-agnostic music interface.

Architecture:  voice node -> MusicService -> MusicProvider -> a track.
Nothing outside src/models/music knows where a track came from. A track is
always a WAV file, so it goes through the same mix as the voice.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.exceptions import ModelProviderError, PlatformError


class MusicError(PlatformError):
    """A provider could not make a track. The service moves to the next
    provider; the reason is kept on the result."""


class MusicAuthError(MusicError):
    """Key missing or rejected. Every request would fail the same way."""


class MusicQuotaError(MusicError):
    """The allowance is spent, or the model needs a paid account. Not retried."""


class MusicTransportError(MusicError, ModelProviderError):
    """No answer: a timeout, a dropped connection, a 5xx. Retried, and only
    this kind (agents.base.with_retries retries ModelProviderError)."""


@dataclass
class MusicRequest:
    prompt: str                        # what a generator is asked; always instrumental
    mood: str
    duration_s: float                  # the video's length
    tone: str = ""
    objective: str = ""
    creative_direction: str = ""


@dataclass
class MusicTrack:
    path: Path
    provider: str
    duration_s: float
    model_id: str = ""
    prompt: str = ""
    cost_usd: float = 0.0
    from_cache: bool = False
    track_id: str = ""                 # the generator's own id for the track
    title: str = ""                    # a library track's name
    licence: str = ""                  # a library track's licence, first line; a generated track's known terms
    notes: list[str] = field(default_factory=list)


class MusicProvider(ABC):
    name: str = "base"
    model_id: str = ""
    # False when making the track again costs nothing (a library file).
    cacheable: bool = True

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else f"{self.name} is not available"

    def cache_key_parts(self, request: MusicRequest) -> dict:
        """What the cache keys on besides the provider: the prompt, and the
        model this request would be made with."""
        return {"model": self.model_id, "prompt": request.prompt}

    @abstractmethod
    def generate(self, request: MusicRequest, dest: Path) -> MusicTrack:
        """A track for the request, written to `dest` as a WAV."""

    def beat_times(self, track: Optional[Path]) -> list[float]:
        """Beat positions in seconds, for cutting on the beat. Nothing snaps to
        them yet."""
        return []
