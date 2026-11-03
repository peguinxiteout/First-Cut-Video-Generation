"""Offline stand-in used when no HF token is configured and in tests.

Produces a soft, deterministic gradient at the requested aspect ratio so
layouts can be built and checked end to end without network access or
credits. It is clearly a placeholder (labelled "preview"), never passed
off as a real render.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from src.models.image_gen.base import GeneratedImage, ImageProvider, ImageRequest


class MockImageProvider(ImageProvider):
    name = "mock"

    def generate(self, request: ImageRequest, output_path: Path) -> GeneratedImage:
        digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
        top = (180 + digest[0] % 50, 165 + digest[1] % 50, 145 + digest[2] % 50)
        bottom = (70 + digest[3] % 40, 60 + digest[4] % 40, 55 + digest[5] % 40)
        w, h = max(64, request.width // 2), max(64, request.height // 2)
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            t = y / max(1, h - 1)
            draw.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
        draw.text((12, h - 24), "AI image preview (offline mock)", fill=(245, 242, 236))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        return GeneratedImage(path=output_path, model_id="mock", prompt=request.prompt,
                              width=w, height=h, meta={"mock": True})
