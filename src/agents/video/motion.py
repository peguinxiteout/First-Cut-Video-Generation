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

LAYOUT. A photograph is cut to the frame ("fill") only when that enlarges its
pixels at most `layout.fill_max_upscale` (agents/video/layout.py) at the worst
moment of the move; otherwise the whole photograph is laid across the frame on
a blurred copy of itself ("fit", models/video_gen/fit.py) and only it moves.
The layout and the peak upscale of every shot are recorded on its clip. They
are worked out for the FINAL frame size whatever size is being rendered, so a
draft shows the layout the final video will have.

LENGTH. Every clip is made a little longer than its shot, to feed the
crossfades at its two ends (rendering/transitions.py).

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

from src.agents.video import layout as layouts
from src.core.config import load_yaml_config
from src.data.video_schemas import BrandKit, ClipResult, Shot, Storyboard, VideoBrief
from src.models.video_gen import ImageToVideoRequest, VideoService, fit, get_video_service
from src.models.video_gen.kenburns import crop_box, parse_focal  # noqa: F401 - crop_box is part of this module's surface
from src.rendering import ffmpeg, transitions
from src.rendering.video_compose import shot_frames

logger = logging.getLogger(__name__)

MAX_KEYFRAME_SCALE = 2          # a keyframe is never kept larger than this many frame widths
LOGO_MAX_WIDTH = 0.5            # of the frame
LOGO_MAX_HEIGHT = 0.16
LOGO_PAD = 0.04               # the plate's margin round the logo, of the frame width
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
    storyboard: Storyboard
    keyframes: dict[int, str] = field(default_factory=dict)
    clips: list[ClipResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as handle:
        return ImageOps.exif_transpose(handle).size


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


def _logo_size(width: int, height: int, out_w: int, out_h: int) -> tuple[int, int]:
    # at most twice its own size: a 200 px logo blown up further is only blur
    scale = min(out_w * LOGO_MAX_WIDTH / width, out_h * LOGO_MAX_HEIGHT / height, 2.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _plate(logo_w: int, logo_h: int, out_w: int, out_h: int) -> tuple[int, int, int, int]:
    pad = round(out_w * LOGO_PAD)
    return (round((out_w - logo_w) / 2) - pad, round(out_h * LOGO_CENTRE_Y - logo_h / 2) - pad,
            round((out_w + logo_w) / 2) + pad, round(out_h * LOGO_CENTRE_Y + logo_h / 2) + pad)


def logo_plate(brand: BrandKit, out_w: int, out_h: int) -> Optional[tuple[int, int, int, int]]:
    """Where the white plate behind the logo sits on a card (left, top, right,
    bottom), or None when the card has no readable logo."""
    if not brand.logo_path:
        return None
    try:
        with Image.open(brand.logo_path) as handle:
            width, height = ImageOps.exif_transpose(handle).size
    except Exception:  # noqa: BLE001 - the card is drawn without it too
        return None
    return _plate(*_logo_size(width, height, out_w, out_h), out_w, out_h)


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
            logo_w, logo_h = _logo_size(logo.width, logo.height, out_w, out_h)
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
            plate = _plate(logo_w, logo_h, out_w, out_h)
            pad = round(out_w * LOGO_PAD)
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


def _card_clip(shot: Shot, brief: VideoBrief, brand: BrandKit, shots_dir: Path, dest: Path, seconds: float) -> bool:
    png = shots_dir / f"card_{shot.index:02d}.png"
    drawn = render_card(brand, brief.width, brief.height, png)
    still_clip(png, seconds, brief.fps, dest)
    return drawn


def _fill_clip(shot: Shot, brief: VideoBrief, shots_dir: Path, dest: Path, seconds: float,
               service: VideoService, notes: list[str], label: str):
    keyframe = make_keyframe(shot.source_path, shot.focal, brief.width, brief.height,
                             shots_dir / f"keyframe_{shot.index:02d}.png")
    if keyframe.upscaled_from:
        notes.append(f"{label}: the photograph is small ({keyframe.upscaled_from[0]}x"
                     f"{keyframe.upscaled_from[1]} after cropping) and was enlarged.")
    clip = service.generate(ImageToVideoRequest(
        image_path=keyframe.path, prompt=shot.motion_prompt, camera=shot.camera, duration_s=seconds,
        width=brief.width, height=brief.height, fps=brief.fps), dest)
    return clip, str(keyframe.path)


def _fit_clip(shot: Shot, brief: VideoBrief, shots_dir: Path, dest: Path, seconds: float, config: dict):
    keyframe = shots_dir / f"keyframe_{shot.index:02d}.png"
    clip = fit.render(shot.source_path, shot.camera, seconds, brief.width, brief.height, brief.fps, dest,
                      focal=shot.focal, dim=config["dim"], blur_px=config["blur_px"], centre_y=config["centre_y"],
                      zoom=config["zoom"], keyframe_path=keyframe)
    return clip, str(keyframe)


def run(storyboard: Storyboard, brief: VideoBrief, brand: BrandKit, out_dir: Path,
        service: Optional[VideoService] = None) -> MotionResult:
    service = service or get_video_service(brief.model, brief.duration_s)
    shots_dir = out_dir / "shots"
    board = storyboard.model_copy(deep=True)
    result = MotionResult(board)
    if service.note:
        result.notes.append(service.note)

    layout_config = layouts.settings()
    final_w, final_h = load_yaml_config("video/pacing.yaml")["sizes"]["final"][brief.aspect]
    frames = shot_frames([s.seconds for s in board.shots], brief.fps)
    margin = transitions.margins(frames, transitions.transition_frames(
        load_yaml_config("video/pacing.yaml")["compose"]["transition_s"], brief.fps))

    for shot, n, (head, tail) in zip(board.shots, frames, margin):
        dest = shots_dir / f"shot_{shot.index:02d}.mp4"
        label = f"Shot {shot.index + 1}"
        seconds = (n + head + tail) / brief.fps           # the shot, and what its crossfades borrow
        reason = ""
        if shot.kind == "real":
            try:
                width, height = _size(shot.source_path)
                chosen = layouts.choose(width, height, final_w, final_h, shot.focal, shot.camera, layout_config)
                if chosen.layout == "fit":
                    clip, keyframe = _fit_clip(shot, brief, shots_dir, dest, seconds, layout_config["fit"])
                    result.notes.append(
                        f"{label}: laid out as fit, the whole photograph on a blurred copy: filling the frame "
                        f"would enlarge it {chosen.fill_peak:.2f}x (limit {layout_config['fill_max_upscale']:g}x); "
                        f"peak enlargement {chosen.peak_upscale:.2f}x.")
                else:
                    clip, keyframe = _fill_clip(shot, brief, shots_dir, dest, seconds, service, result.notes, label)
                shot.layout, shot.peak_upscale = chosen.layout, chosen.peak_upscale
                result.keyframes[shot.index] = keyframe
                result.clips.append(ClipResult(
                    shot_index=shot.index, path=str(clip.path), provider=clip.provider, model_id=clip.model_id,
                    mode="kenburns" if clip.provider == "kenburns" else "model", layout=chosen.layout,
                    peak_upscale=chosen.peak_upscale, ok=True, reason=clip.reason, from_cache=clip.from_cache,
                    duration_s=clip.duration_s, cost_usd=clip.cost_usd))
                continue
            except Exception as exc:  # noqa: BLE001 - a shot that cannot be made is a card, not a failed run
                reason = f"{type(exc).__name__}: {exc}"
        elif shot.kind == "generated":
            reason = "generated pictures are not made yet"

        try:
            _card_clip(shot, brief, brand, shots_dir, dest, seconds)
        except Exception as exc:  # noqa: BLE001 - the card is the last resort; say it failed
            raise RuntimeError(f"{label} could not be made, not even as a card: {exc}") from exc
        failed = bool(reason)
        if failed:
            result.notes.append(f"{label} is a title card: {reason}.")
        result.clips.append(ClipResult(shot_index=shot.index, path=str(dest), mode="card", ok=not failed,
                                       reason=reason, duration_s=round(seconds, 3)))
    return result
