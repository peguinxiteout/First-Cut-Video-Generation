"""Pan-and-zoom: camera motion over a still, made locally.

This is the provider every chain ends with. It needs no key, costs nothing
and cannot be out of quota, so a shot always gets a clip, and with no model
at all the workflow still makes its video. It also never alters the
picture: the building, the rooms and the views are the photograph's own.

Each frame is a LANCZOS resample of a moving window of the photograph, cut
at sub-pixel positions (a whole-pixel crop stutters at slow speeds) and
piped to ffmpeg, which encodes H.264 yuv420p. The window keeps the output's
aspect ratio, starts around the focal point the storyboard gives, and always
stays inside the picture.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps

from src.core.exceptions import RenderingError
from src.models.video_gen.base import (
    ImageToVideoRequest, VideoClip, VideoGenerationError, VideoProvider, check_request,
)
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

PUSH_ZOOM = 0.18        # a push-in ends 18% closer; a pull-out starts there
PAN_ZOOM = 1.15         # pans and tilts are zoomed in a little, to have room to travel
PAN_FRACTION = 0.30     # and travel at most this share of the window, however wide the photograph
# The source is scaled down to this many output widths across the window
# before the frames are cut, so a 6000 px photo does not cost 6000 px per frame.
SOURCE_OVERSAMPLE = 1.5


def _ease(t: float) -> float:
    return t * t * t * (t * (6 * t - 15) + 10)


def parse_focal(text: str) -> tuple[float, float]:
    """"37% 62%" -> (0.37, 0.62); anything unreadable keeps the centre."""
    try:
        x, y = (float(part.strip().rstrip("%")) / 100 for part in text.split())
    except ValueError:
        return 0.5, 0.5
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


@dataclass(frozen=True)
class Geometry:
    """The photograph, the largest output-shaped window that fits it, and
    where the subject is. All in source pixels."""
    src_w: int
    src_h: int
    win_w: float
    win_h: float
    focal_x: float
    focal_y: float

    @classmethod
    def of(cls, src_w: int, src_h: int, out_w: int, out_h: int, focal: str) -> "Geometry":
        aspect = out_w / out_h
        if src_w / src_h > aspect:
            win_h, win_w = float(src_h), src_h * aspect
        else:
            win_w, win_h = float(src_w), src_w / aspect
        fx, fy = parse_focal(focal)
        return cls(src_w, src_h, win_w, win_h, fx * src_w, fy * src_h)


def _axis_travel(centre: float, size: float, win: float, fraction_of_win: float) -> tuple[float, float]:
    """Start and end centres (along one axis) for a pan or tilt of a window
    `win` wide, over a picture `size` wide, moving as far as is allowed."""
    low, high = win / 2, size - win / 2
    travel = min(fraction_of_win * win, max(0.0, high - low))
    mid = min(max(centre, low + travel / 2), high - travel / 2)
    return mid + travel / 2, mid - travel / 2          # the larger end first


def motion(camera: str, g: Geometry) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(zoom, centre x, centre y) at the start and at the end of the move."""
    fx, fy = g.focal_x, g.focal_y
    if camera == "still":
        return (1.0, fx, fy), (1.0, fx, fy)
    if camera == "push_in":
        return (1.0, fx, fy), (1.0 + PUSH_ZOOM, fx, fy)
    if camera == "pull_out":
        return (1.0 + PUSH_ZOOM, fx, fy), (1.0, fx, fy)
    win_w, win_h = g.win_w / PAN_ZOOM, g.win_h / PAN_ZOOM
    if camera in ("pan_left", "pan_right"):
        a, b = _axis_travel(fx, g.src_w, win_w, PAN_FRACTION * PAN_ZOOM)
        start, end = (a, b) if camera == "pan_left" else (b, a)     # pan_left: the window moves left
        return (PAN_ZOOM, start, fy), (PAN_ZOOM, end, fy)
    a, b = _axis_travel(fy, g.src_h, win_h, PAN_FRACTION * PAN_ZOOM)
    start, end = (a, b) if camera == "tilt_up" else (b, a)           # tilt_up: the window moves up
    return (PAN_ZOOM, fx, start), (PAN_ZOOM, fx, end)


def window_at(camera: str, progress: float, g: Geometry) -> tuple[float, float, float, float]:
    """The crop box (left, top, right, bottom) `progress` (0..1, eased) of
    the way through the move."""
    (z0, x0, y0), (z1, x1, y1) = motion(camera, g)
    z = z0 + (z1 - z0) * progress
    cx, cy = x0 + (x1 - x0) * progress, y0 + (y1 - y0) * progress
    w, h = g.win_w / z, g.win_h / z
    cx = min(max(cx, w / 2), g.src_w - w / 2)
    cy = min(max(cy, h / 2), g.src_h - h / 2)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _even(value: int) -> int:
    return max(2, value - value % 2)        # yuv420p needs even sides


class KenBurnsProvider(VideoProvider):
    id = "kenburns"
    model_id = "kenburns"
    cacheable = False

    def estimate_cost(self, seconds: float) -> float:
        return 0.0

    def _source(self, request: ImageToVideoRequest, out_w: int, out_h: int) -> tuple[Image.Image, Geometry]:
        try:
            with Image.open(request.image_path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any unreadable picture is the same answer
            raise VideoGenerationError(f"cannot read {Path(request.image_path).name}: {exc}") from exc
        geometry = Geometry.of(image.width, image.height, out_w, out_h, request.focal)
        scale = SOURCE_OVERSAMPLE * out_w / geometry.win_w
        if scale < 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                 Image.LANCZOS)
            geometry = Geometry.of(image.width, image.height, out_w, out_h, request.focal)
        return image, geometry

    def generate(self, request: ImageToVideoRequest, output_path: Path) -> VideoClip:
        check_request(request)
        out_w, out_h = _even(request.width), _even(request.height)
        duration = min(request.duration_s, self.max_duration_s)
        frames = max(2, round(duration * request.fps))
        image, geometry = self._source(request, out_w, out_h)

        def render() -> Iterator[bytes]:
            for i in range(frames):
                box = window_at(request.camera, _ease(i / (frames - 1)), geometry)
                yield image.resize((out_w, out_h), Image.LANCZOS, box=box).tobytes()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ffmpeg.encode_frames(render(), output_path, width=out_w, height=out_h, fps=request.fps)
        except RenderingError as exc:
            raise VideoGenerationError(f"pan-and-zoom encode failed: {exc}") from exc

        meta = {"camera": request.camera}
        if duration < request.duration_s:
            meta["clamped_from_s"] = request.duration_s
            logger.info("Pan-and-zoom clip held to %.1fs (asked for %.1fs): max_duration_s", duration,
                        request.duration_s)
        return VideoClip(path=output_path, provider=self.id, model_id=self.model_id,
                         duration_s=round(frames / request.fps, 3), width=out_w, height=out_h,
                         fps=request.fps, cost_usd=0.0, meta=meta)
