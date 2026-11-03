"""
Shrinks images before they're sent to a vision-capable model API call —
used by both the Vision Agent (many uploaded photos in one batched call)
and the Critic (one rendered screenshot). Two independent, real failures
motivated this:

1. A batch of full-resolution uploaded photos (each several MB, a dozen or
   more in one call) can exceed a provider's payload size limit outright
   — OpenRouter rejected a real request with "Downloaded image content
   cannot exceed 30MB".
2. Even short of a hard rejection, full-resolution images send far more
   bytes than a vision model's classification/evaluation actually needs —
   most multimodal vision encoders operate on much smaller internal tiles
   (roughly 768-1568px) regardless of the input resolution, so anything
   larger than that is wasted bandwidth and latency, not extra accuracy.

Deliberately only touches a TEMPORARY copy used for the API call itself —
never the original file. The images used in the actual rendered PDF are
always the full-resolution originals, completely unaffected by this.

Two separate resize strategies, because "a photo" and "a full-page
brochure screenshot" are shaped completely differently:
  - A photo is roughly square-ish — capping its LONGEST edge is the right
    call (see `prepare_image_for_vision_api`).
  - The Critic's screenshot is a full multi-page brochure captured as one
    very tall image (Playwright's `full_page=True` stacks every page).
    Capping its longest edge would crush a 12-page screenshot down to a
    sliver a few dozen pixels wide — useless for judging visual design.
    It needs its WIDTH capped instead, with a generous height ceiling as
    a secondary safety net (see `prepare_screenshot_for_vision_api`).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image

# Long-edge cap for a photo sent to a vision API — comfortably above what
# multimodal vision encoders resolve internally, so there's no accuracy
# cost, while cutting a typical multi-megapixel photo down to a few
# hundred KB once re-encoded.
_MAX_DIMENSION = 1568
_JPEG_QUALITY = 85

# For the Critic's full-page screenshot: width matters far more than
# height (it's naturally tall, by design — every page stacked). Capped
# generously enough that each individual page within the stack still
# reads clearly; the height ceiling only kicks in for an unusually long
# brochure, as a hard safety net against provider dimension limits.
_MAX_SCREENSHOT_WIDTH = 1400
_MAX_SCREENSHOT_HEIGHT = 12000


def _save_resized(img: Image.Image, output_path: Path, target_size: tuple[int, int] | None) -> None:
    img = img.convert("RGB")  # drops alpha/palette modes JPEG can't encode
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    img.save(output_path, format="JPEG", quality=_JPEG_QUALITY, optimize=True)


def prepare_image_for_vision_api(image_path: str, output_dir: Path) -> str:
    """Resizes+re-encodes ONE photo into `output_dir` and returns the new
    path, capping its LONGEST edge — appropriate for a normal, roughly
    square-ish photo (see module docstring for why the Critic's
    screenshot needs different handling).

    Falls back to the ORIGINAL path, unchanged, if resizing fails for any
    reason (an unusual format PIL can't decode, a corrupt file, etc.) —
    a local resize failure shouldn't block an entire generation when the
    original file might still work fine sent as-is. Whatever rejects it
    next (the API itself) is a more informative failure than a resize
    step crashing first."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{uuid.uuid4().hex}.jpg"

        with Image.open(image_path) as img:
            longest_edge = max(img.size)
            target_size = None
            if longest_edge > _MAX_DIMENSION:
                scale = _MAX_DIMENSION / longest_edge
                target_size = (round(img.width * scale), round(img.height * scale))
            _save_resized(img, output_path, target_size)

        return str(output_path)
    except Exception:  # noqa: BLE001 - see docstring: fall back rather than fail the whole generation
        return image_path


def prepare_images_for_vision_api(image_paths: list[str], output_dir: Path) -> list[str]:
    return [prepare_image_for_vision_api(path, output_dir) for path in image_paths]


def prepare_screenshot_for_vision_api(screenshot_path: str, output_dir: Path) -> str:
    """Same purpose as `prepare_image_for_vision_api`, but for the
    Critic's full-page brochure screenshot specifically — see the module
    docstring for why a tall, multi-page image needs width-first, not
    longest-edge, capping. Same fall-back-to-original behavior on failure."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{uuid.uuid4().hex}.jpg"

        with Image.open(screenshot_path) as img:
            width_scale = min(1.0, _MAX_SCREENSHOT_WIDTH / img.width)
            height_scale = min(1.0, _MAX_SCREENSHOT_HEIGHT / img.height)
            scale = min(width_scale, height_scale)
            target_size = (round(img.width * scale), round(img.height * scale)) if scale < 1.0 else None
            _save_resized(img, output_path, target_size)

        return str(output_path)
    except Exception:  # noqa: BLE001 - fall back rather than fail the whole generation
        return screenshot_path

