"""A stand-in video model for tests and development. Hidden from the
selector (providers.yaml: hidden).

It stands in for a PAID provider: it is cacheable, it can carry a price from
its spec, and it counts its calls, so the cache, the cost accounting and the
chain can be tested without a vendor. The clip is a flat colour taken from
the picture's bytes (different pictures give different clips) and is never
placed in a deliverable."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.exceptions import RenderingError
from src.models.video_gen.base import (
    ImageToVideoRequest, VideoClip, VideoGenerationError, VideoProvider, check_request,
)
from src.rendering import ffmpeg


class MockVideoProvider(VideoProvider):
    id = "mock"
    model_id = "mock"

    def __init__(self, spec=None, api_key: str = ""):
        super().__init__(spec, api_key)
        self.calls: list[ImageToVideoRequest] = []

    def generate(self, request: ImageToVideoRequest, output_path: Path) -> VideoClip:
        check_request(request)
        self.calls.append(request)
        duration = min(request.duration_s, self.max_duration_s)
        digest = hashlib.sha256(Path(request.image_path).read_bytes()).hexdigest()
        width, height = request.width - request.width % 2, request.height - request.height % 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ffmpeg.run(["-f", "lavfi", "-i", f"color=c=0x{digest[:6]}:s={width}x{height}:r={request.fps}:d={duration}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)])
        except RenderingError as exc:
            raise VideoGenerationError(f"mock encode failed: {exc}") from exc
        return VideoClip(path=output_path, provider=self.id, model_id=self.model_id, duration_s=duration,
                         width=width, height=height, fps=request.fps,
                         cost_usd=self.estimate_cost(duration) or 0.0, meta={"mock": True})
