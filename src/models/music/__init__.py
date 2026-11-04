"""Background music.

The video workflow has a place for a music track (it is mixed under the
voice, ducked while it speaks) and for beat times to cut on. Nothing supplies
either yet: there is no licensed library, and the question of which music may
be used is open. NullMusic is the provider until one exists, so the mix and
the cuts are written against the real interface now."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class MusicProvider(ABC):
    name: str = "base"

    @abstractmethod
    def track(self, mood: str, duration_s: float, dest: Path) -> Optional[Path]:
        """A music track of about `duration_s` for the mood, written to dest
        as a WAV, or None when there is none."""

    def beat_times(self, track: Optional[Path]) -> list[float]:
        """Beat positions in seconds, for cutting on the beat."""
        return []


class NullMusic(MusicProvider):
    name = "none"

    def track(self, mood: str, duration_s: float, dest: Path) -> Optional[Path]:
        return None


def get_music_provider() -> MusicProvider:
    return NullMusic()
