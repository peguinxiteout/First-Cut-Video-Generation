"""Fakes for the video and speech tests. Transports are injected, as in
tests/unit/test_image_gen.py; no library is mocked."""
from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from src.models.speech.base import pcm_to_wav
from src.models.video_gen.base import ImageToVideoRequest, ProviderSpec, VideoClip
from src.models.video_gen.mock import MockVideoProvider


def make_photo(path: Path, size=(1600, 900), seed: int = 0) -> Path:
    """A photograph-like JPEG: blobs on a flat ground, different per seed."""
    rng = random.Random(seed)
    colour = tuple(rng.randrange(60, 200) for _ in range(3))
    image = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(image)
    w, h = size
    for _ in range(30):
        x, y, r = rng.randrange(w), rng.randrange(h), rng.randrange(20, max(21, w // 5))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rng.randrange(0, 255) for _ in range(3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=85)
    return path


def small_request(image: Path, **overrides) -> ImageToVideoRequest:
    """A cheap request: 160x284 at 10 fps for one second."""
    fields = dict(image_path=image, camera="push_in", duration_s=1.0, width=160, height=284, fps=10)
    fields.update(overrides)
    return ImageToVideoRequest(**fields)


def paid_spec(provider_id: str = "paid", **overrides) -> ProviderSpec:
    fields = dict(id=provider_id, label="Paid", adapter="mock", cost_per_second=0.1, max_duration_s=10.0)
    fields.update(overrides)
    return ProviderSpec(**fields)


class ScriptedVideoProvider(MockVideoProvider):
    """A paid provider that does what `script` says, one entry per call:
    an exception instance is raised, anything else makes a clip. When the
    script runs out it makes clips."""

    def __init__(self, provider_id: str = "paid", script=(), spec: Optional[ProviderSpec] = None):
        super().__init__(spec or paid_spec(provider_id))
        self.id = provider_id
        self.script = list(script)

    def generate(self, request: ImageToVideoRequest, output_path: Path) -> VideoClip:
        outcome = self.script.pop(0) if self.script else None
        if isinstance(outcome, Exception):
            self.calls.append(request)
            raise outcome
        return super().generate(request, output_path)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakePost:
    """Stands in for requests.post. Outcomes are answered in order and the
    last one repeats; an exception instance is raised."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def silence(seconds: float, rate: int = 24000) -> bytes:
    return b"\x00\x00" * round(seconds * rate)


def audio_response(data: bytes, mime_type: str = "audio/wav") -> FakeResponse:
    block = {"type": "audio", "data": base64.b64encode(data).decode("ascii"), "mime_type": mime_type}
    return FakeResponse(200, {"steps": [{"type": "model_output", "content": [block]}]})


def wav_response(seconds: float = 0.5, rate: int = 24000) -> FakeResponse:
    return audio_response(pcm_to_wav(silence(seconds, rate), rate))


def tone_wav(path: Path, seconds: float, rate: int = 44100, amplitude: float = 0.4, hz: float = 220.0) -> Path:
    """A steady tone as a mono WAV: something with a level, for the music tests."""
    import math
    import struct
    import wave

    frames = b"".join(struct.pack("<h", round(32767 * amplitude * math.sin(2 * math.pi * hz * i / rate)))
                      for i in range(round(seconds * rate)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


class FakeMusic:
    """A music provider that writes a tone and counts its calls. Give it an
    exception instance to have it fail."""

    cacheable = True
    model_id = "fake-model"

    def __init__(self, name: str = "fake", seconds: float = 12.0, cost_usd: float = 0.04, error: Exception | None = None,
                 available: bool = True, cacheable: bool = True):
        self.name, self.seconds, self.cost_usd, self.error = name, seconds, cost_usd, error
        self._available, self.cacheable = available, cacheable
        self.requests: list = []

    def is_available(self) -> bool:
        return self._available

    def unavailable_reason(self) -> str:
        return "" if self._available else f"{self.name} has no key"

    def cache_key_parts(self, request) -> dict:
        return {"model": self.model_id, "prompt": request.prompt}

    def generate(self, request, dest: Path):
        from src.models.music.base import MusicTrack
        from src.models.speech.base import wav_duration

        self.requests.append(request)
        if self.error:
            raise self.error
        tone_wav(dest, self.seconds)
        return MusicTrack(path=dest, provider=self.name, model_id=self.model_id, prompt=request.prompt,
                          duration_s=wav_duration(dest), cost_usd=self.cost_usd)

    def beat_times(self, track):
        return []
