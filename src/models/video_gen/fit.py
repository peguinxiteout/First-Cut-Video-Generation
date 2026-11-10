"""The "fit" layout: the whole photograph across the frame, on a copy of itself.

A photograph whose pixels would be enlarged too much to fill a tall frame
(a wide picture in a 9:16 video) is not cropped to the frame. It is laid across
the frame's width, in full, over a blurred, enlarged and darkened copy of itself
that fills what is left; only the photograph moves, gently (a slow push or a
slight drift), so the move never shows the blur's edges.

Like pan-and-zoom it is made here, needs no key and costs nothing, and it never
alters the picture: the building, the rooms and the views are the photograph's
own. Each frame is a LANCZOS resample cut at sub-pixel positions, piped to
ffmpeg (a whole-pixel move stutters at this speed).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.models.video_gen.base import VideoClip, VideoGenerationError
from src.models.video_gen.kenburns import _ease, crop_box
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

BLUR_WORK_DIVISOR = 4           # the copy is blurred at a quarter of the frame's size, then enlarged
SOURCE_OVERSAMPLE = 1.5         # the photograph is held at no more than this many frame widths


def peak_zoom(camera: str, zoom: float) -> float:
    """The most the photograph is enlarged beyond filling the frame's width."""
    return 1.0 if camera == "still" else 1.0 + zoom


def base_width(src_w: int, src_h: int, out_w: int, out_h: int) -> float:
    """How wide the photograph is laid: the frame's width, or less for a photograph
    so tall that the whole of it would not fit the frame's height."""
    return min(float(out_w), out_h * src_w / src_h)


def peak_upscale(src_w: int, src_h: int, out_w: int, out_h: int, camera: str, zoom: float) -> float:
    """The most this photograph's pixels are enlarged at any moment of a fit shot."""
    return base_width(src_w, src_h, out_w, out_h) * peak_zoom(camera, zoom) / src_w


def background(image: Image.Image, focal: str, width: int, height: int, dim: float, blur_px: float) -> Image.Image:
    """The picture cut to the frame's shape around the focal point, blurred,
    enlarged to the frame and darkened."""
    crop = image.crop(crop_box(image.width, image.height, width / height, focal))
    small = crop.resize((max(1, width // BLUR_WORK_DIVISOR), max(1, height // BLUR_WORK_DIVISOR)), Image.BICUBIC)
    blurred = small.filter(ImageFilter.GaussianBlur(radius=max(0.5, blur_px / BLUR_WORK_DIVISOR)))
    return ImageEnhance.Brightness(blurred.resize((width, height), Image.BICUBIC)).enhance(dim)


def _path(camera: str, zoom: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(zoom, drift x, drift y) at the start and end of the move. Drift is a
    share of the room the zoom leaves (-1 to 1): a pan or a tilt holds the
    zoom and slides across it."""
    z = 1.0 + zoom
    if camera == "push_in":
        return (1.0, 0.0, 0.0), (z, 0.0, 0.0)
    if camera == "pull_out":
        return (z, 0.0, 0.0), (1.0, 0.0, 0.0)
    if camera == "pan_left":
        return (z, -1.0, 0.0), (z, 1.0, 0.0)
    if camera == "pan_right":
        return (z, 1.0, 0.0), (z, -1.0, 0.0)
    if camera == "tilt_up":
        return (z, 0.0, -1.0), (z, 0.0, 1.0)
    if camera == "tilt_down":
        return (z, 0.0, 1.0), (z, 0.0, -1.0)
    return (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)


def compose_frame(bg: Image.Image, src: Image.Image, width: int, height: int, centre_y: float, camera: str,
                  zoom: float, progress: float) -> Image.Image:
    """One frame: the background, and the photograph across it at this point of the move."""
    (z0, x0, y0), (z1, x1, y1) = _path(camera, zoom)
    z = z0 + (z1 - z0) * progress
    w = base_width(src.width, src.height, width, height) * z
    h = w * src.height / src.width
    room_x, room_y = max(0.0, (w - width) / 2), (z - 1.0) * h / 2
    cx = width / 2 + (x0 + (x1 - x0) * progress) * room_x
    cy = height * centre_y + (y0 + (y1 - y0) * progress) * room_y
    left, top = cx - w / 2, cy - h / 2
    ix0, iy0 = max(0, int(left)), max(0, int(top))
    ix1, iy1 = min(width, int(left + w) + 1), min(height, int(top + h) + 1)
    frame = bg.copy()
    if ix1 > ix0 and iy1 > iy0:
        box = ((ix0 - left) / w * src.width, (iy0 - top) / h * src.height,
               (ix1 - left) / w * src.width, (iy1 - top) / h * src.height)
        box = (max(0.0, box[0]), max(0.0, box[1]), min(float(src.width), box[2]), min(float(src.height), box[3]))
        frame.paste(src.resize((ix1 - ix0, iy1 - iy0), Image.LANCZOS, box=box), (ix0, iy0))
    return frame


def render(image_path: Path, camera: str, duration_s: float, width: int, height: int, fps: int, output_path: Path,
           *, focal: str = "50% 50%", dim: float = 0.62, blur_px: float = 34.0, centre_y: float = 0.47,
           zoom: float = 0.06, keyframe_path: Optional[Path] = None) -> VideoClip:
    """A clip of `duration_s` with the photograph laid out as "fit". Raises
    VideoGenerationError when the picture cannot be read or encoded."""
    try:
        with Image.open(image_path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - any unreadable picture is the same answer
        raise VideoGenerationError(f"cannot read {Path(image_path).name}: {exc}") from exc
    unit = min(width, height) / 1080
    bg = background(image, focal, width, height, dim, blur_px * unit)
    limit = round(width * (1 + zoom) * SOURCE_OVERSAMPLE)
    src = image if image.width <= limit else image.resize((limit, round(image.height * limit / image.width)),
                                                          Image.LANCZOS)
    frames = max(2, round(duration_s * fps))

    def render_frames() -> Iterator[bytes]:
        for i in range(frames):
            yield compose_frame(bg, src, width, height, centre_y, camera, zoom, _ease(i / (frames - 1))).tobytes()

    if keyframe_path is not None:
        keyframe_path.parent.mkdir(parents=True, exist_ok=True)
        compose_frame(bg, src, width, height, centre_y, camera, zoom, 0.0).save(keyframe_path, "PNG")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg.encode_frames(render_frames(), output_path, width=width, height=height, fps=fps)
    except Exception as exc:  # noqa: BLE001 - a clip that cannot be encoded is reported as the shot's failure
        raise VideoGenerationError(f"fit encode failed: {exc}") from exc
    return VideoClip(path=output_path, provider="kenburns", model_id="kenburns-fit",
                     duration_s=round(frames / fps, 3), width=width, height=height, fps=fps, cost_usd=0.0,
                     meta={"camera": camera, "layout": "fit"})
