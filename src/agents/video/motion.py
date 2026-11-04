"""Node 7, motion: a keyframe and a clip for every shot.

Image first, then motion. A real shot starts from a KEYFRAME: the person's
photograph cut to the video's shape around the point the storyboard chose and,
when it is smaller than the frame, enlarged with LANCZOS so the motion has
pixels to work with. The clip is then made from that keyframe by the model
chain (a video model when one is chosen and available, pan-and-zoom always
last). The motion only moves the camera; the picture is the photograph's own.

Soft failure: a clip that cannot be made never fails the run. The chain falls
to pan-and-zoom on its own; a shot that cannot even be cut (an unreadable
picture, a generated shot with no image model) becomes a title card in the
palette and the reason is recorded.

A CARD is the palette as a background with the developer's real logo set on a
white plate (a logo is usually a JPEG on white); the words on it are drawn by
the overlay step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageOps

from src.data.video_schemas import BrandKit, ClipResult, Shot, Storyboard, VideoBrief
from src.models.video_gen import ImageToVideoRequest, VideoService, get_video_service
from src.models.video_gen.kenburns import parse_focal
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

MAX_KEYFRAME_SCALE = 2          # a keyframe is never kept larger than this many frame widths
LOGO_MAX_WIDTH = 0.5            # of the frame
LOGO_MAX_HEIGHT = 0.16
LOGO_CENTRE_Y = 0.32            # the logo's middle, from the top: above the words, clear of the safe-zone bottom
DEFAULT_DARK, DEFAULT_ACCENT = "#1B1B1F", "#B98A5E"


@dataclass
class Keyframe:
    path: Path
    width: int
    height: int
    upscaled_from: Optional[tuple[int, int]] = None


@dataclass
class MotionResult:
    keyframes: dict[int, str] = field(default_factory=dict)
    clips: list[ClipResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def crop_box(src_w: int, src_h: int, out_aspect: float, focal: str) -> tuple[int, int, int, int]:
    """The largest window of the picture in the video's shape, placed the way
    CSS object-position places it: "50% 50%" centres it, "0% 50%" pushes it
    against the left edge, "100% 50%" the right. (focal.focal_point speaks this.)"""
    fx, fy = parse_focal(focal)
    if src_w / src_h > out_aspect:
        win_w, win_h = round(src_h * out_aspect), src_h
        left, top = round(fx * (src_w - win_w)), 0
    else:
        win_w, win_h = src_w, round(src_w / out_aspect)
        left, top = 0, round(fy * (src_h - win_h))
    return left, top, left + win_w, top + win_h


def make_keyframe(source: str | Path, focal: str, out_w: int, out_h: int, dest: Path) -> Keyframe:
    with Image.open(source) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    image = image.crop(crop_box(image.width, image.height, out_w / out_h, focal))
    upscaled = None
    if image.width < out_w or image.height < out_h:
        upscaled = image.size
        image = image.resize((out_w, out_h), Image.LANCZOS)
    elif image.width > out_w * MAX_KEYFRAME_SCALE:
        image = image.resize((out_w * MAX_KEYFRAME_SCALE, out_h * MAX_KEYFRAME_SCALE), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "PNG")
    return Keyframe(dest, image.width, image.height, upscaled)


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def render_card(brand: BrandKit, out_w: int, out_h: int, dest: Path) -> bool:
    """The card's still: the palette's dark ground, a little lighter towards
    the foot, and the logo on a white plate. True when a logo was drawn."""
    dark = _rgb(brand.palette.get("dark", DEFAULT_DARK))
    foot = _blend(dark, _rgb(brand.palette.get("accent", DEFAULT_ACCENT)), 0.18)
    image = Image.new("RGB", (out_w, out_h), dark)
    draw = ImageDraw.Draw(image)
    for y in range(out_h):
        draw.line([(0, y), (out_w, y)], fill=_blend(dark, foot, y / max(1, out_h - 1)))

    drawn = False
    if brand.logo_path:
        try:
            with Image.open(brand.logo_path) as handle:
                logo = ImageOps.exif_transpose(handle)
                if logo.mode in ("RGBA", "LA", "P"):
                    logo = logo.convert("RGBA")
                    flat = Image.new("RGB", logo.size, "white")
                    flat.paste(logo, mask=logo.split()[-1])
                    logo = flat
                logo = logo.convert("RGB")
            # at most twice its own size: a 200 px logo blown up further is only blur
            scale = min(out_w * LOGO_MAX_WIDTH / logo.width, out_h * LOGO_MAX_HEIGHT / logo.height, 2.0)
            logo = logo.resize((max(1, round(logo.width * scale)), max(1, round(logo.height * scale))), Image.LANCZOS)
            pad = round(out_w * 0.04)
            plate = (round((out_w - logo.width) / 2) - pad, round(out_h * LOGO_CENTRE_Y - logo.height / 2) - pad,
                     round((out_w + logo.width) / 2) + pad, round(out_h * LOGO_CENTRE_Y + logo.height / 2) + pad)
            draw.rounded_rectangle(plate, radius=pad, fill="white")
            image.paste(logo, (plate[0] + pad, plate[1] + pad))
            drawn = True
        except Exception as exc:  # noqa: BLE001 - a logo that cannot be read leaves the card without one
            logger.info("The logo could not be drawn on the card (%s)", type(exc).__name__)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "PNG")
    return drawn


def still_clip(png: Path, seconds: float, fps: int, dest: Path) -> None:
    """A picture held for `seconds`, as H.264."""
    ffmpeg.run(["-loop", "1", "-framerate", str(fps), "-i", str(png), "-frames:v", str(max(1, round(seconds * fps))),
                "-r", str(fps), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(dest)])


def _card_clip(shot: Shot, brief: VideoBrief, brand: BrandKit, shots_dir: Path, dest: Path) -> bool:
    png = shots_dir / f"card_{shot.index:02d}.png"
    drawn = render_card(brand, brief.width, brief.height, png)
    still_clip(png, shot.seconds, brief.fps, dest)
    return drawn


def run(storyboard: Storyboard, brief: VideoBrief, brand: BrandKit, out_dir: Path,
        service: Optional[VideoService] = None) -> MotionResult:
    service = service or get_video_service(brief.model, brief.duration_s)
    shots_dir = out_dir / "shots"
    result = MotionResult()
    if service.note:
        result.notes.append(service.note)

    for shot in storyboard.shots:
        dest = shots_dir / f"shot_{shot.index:02d}.mp4"
        label = f"Shot {shot.index + 1}"
        reason = ""
        if shot.kind == "real":
            try:
                keyframe = make_keyframe(shot.source_path, shot.focal, brief.width, brief.height,
                                         shots_dir / f"keyframe_{shot.index:02d}.png")
                if keyframe.upscaled_from:
                    result.notes.append(f"{label}: the photograph is small ({keyframe.upscaled_from[0]}x"
                                        f"{keyframe.upscaled_from[1]} after cropping) and was enlarged.")
                clip = service.generate(ImageToVideoRequest(
                    image_path=keyframe.path, prompt=shot.motion_prompt, camera=shot.camera,
                    duration_s=shot.seconds, width=brief.width, height=brief.height, fps=brief.fps),
                    dest)
                result.keyframes[shot.index] = str(keyframe.path)
                result.clips.append(ClipResult(
                    shot_index=shot.index, path=str(clip.path), provider=clip.provider, model_id=clip.model_id,
                    mode="kenburns" if clip.provider == "kenburns" else "model", ok=True, reason=clip.reason,
                    from_cache=clip.from_cache, duration_s=clip.duration_s, cost_usd=clip.cost_usd))
                continue
            except Exception as exc:  # noqa: BLE001 - a shot that cannot be made is a card, not a failed run
                reason = f"{type(exc).__name__}: {exc}"
        elif shot.kind == "generated":
            reason = "generated pictures are not made yet"

        try:
            _card_clip(shot, brief, brand, shots_dir, dest)
        except Exception as exc:  # noqa: BLE001 - the card is the last resort; say it failed
            raise RuntimeError(f"{label} could not be made, not even as a card: {exc}") from exc
        failed = bool(reason)
        if failed:
            result.notes.append(f"{label} is a title card: {reason}.")
        result.clips.append(ClipResult(shot_index=shot.index, path=str(dest), mode="card", ok=not failed,
                                       reason=reason, duration_s=shot.seconds))
    return result
