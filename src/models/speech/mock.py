"""A stand-in voice for tests and development. Not part of any configured
chain: tests put it there.

It makes a quiet tone as long as the line takes to say, so loudness
measurement and mixing have something real to work on, and it records every
call. Pass `fail` to make it raise, to test the chain's fallback."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Optional

from src.models.speech.base import Audio, SpeechProvider, count_words

SAMPLE_RATE = 24000


class MockSpeechProvider(SpeechProvider):
    name = "mock"
    model_id = "mock"

    def __init__(self, words_per_second: float = 2.1, fail: Optional[Exception] = None):
        self._wps = words_per_second
        self._fail = fail
        self.calls: list[tuple[str, Optional[str]]] = []

    def synthesize(self, text: str, voice: Optional[str], output_path: Path) -> Audio:
        self.calls.append((text, voice))
        if self._fail is not None:
            raise self._fail
        frames = round(max(0.6, count_words(text) / self._wps) * SAMPLE_RATE)
        samples = (int(3000 * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE)) for i in range(frames))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(b"".join(struct.pack("<h", s) for s in samples))
        return Audio(path=output_path, duration_s=round(frames / SAMPLE_RATE, 3), provider=self.name,
                     voice=voice or "", model_id=self.model_id)
