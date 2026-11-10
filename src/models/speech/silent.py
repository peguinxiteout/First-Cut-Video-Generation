"""The last link of every speech chain: no voice, a silent track.

It is as long as the line takes to say at a natural pace (words per second
from configs/video/speech.yaml), so shot timing, cuts and captions come out
as they would with a voice, and the captions carry the message. It cannot
fail for want of a key or an engine, which is what lets a video always be
made. The Audio is marked mode="silent" so the record and the quality gate
can say so rather than pass it off as narration."""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

from src.models.speech.base import Audio, SpeechProvider, count_words

MIN_DURATION_S = 0.6
SAMPLE_RATE = 24000


class SilentSpeechProvider(SpeechProvider):
    name = "silent"
    model_id = "silent"
    cacheable = False

    def __init__(self, words_per_second: float = 2.1):
        self._wps = max(0.5, words_per_second)

    def duration_for(self, text: str) -> float:
        return round(max(MIN_DURATION_S, count_words(text) / self._wps), 3)

    def synthesize(self, text: str, voice: Optional[str], output_path: Path) -> Audio:
        duration = self.duration_for(text)
        frames = round(duration * SAMPLE_RATE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(b"\x00\x00" * frames)
        return Audio(path=output_path, duration_s=round(frames / SAMPLE_RATE, 3), provider=self.name,
                     mode="silent", voice="", model_id=self.model_id)
