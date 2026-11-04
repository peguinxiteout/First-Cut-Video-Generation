"""Node 8, overlays: what the screen prints, and the captions file.

Every shot with text gets one transparent picture of it (rendering/
video_overlay.py), timed to the shot:

    the hook         the name and, where the category shows it, the price
    a feature shot   its chip, when the shot's photograph matches it (the
                     storyboard decided that)
    the closing card the call to action, RERA if the data has one, and contact

The text comes from the storyboard's shots, sorted into those jobs by what
the script says each piece is. captions.srt is written from the real line
timings, so it follows the voice, whatever the voice was.
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.config import load_yaml_config
from src.data.video_schemas import BrandKit, LineTiming, OverlayItem, Storyboard, VideoBrief, VideoScript
from src.rendering import video_overlay
from src.rendering.video_overlay import OverlaySpec

logger = logging.getLogger(__name__)


@dataclass
class OverlayResult:
    items: list[OverlayItem] = field(default_factory=list)
    captions_path: str = ""
    notes: list[str] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)


def spec_for_shot(index: int, shot, script: VideoScript, brief: VideoBrief, brand: BrandKit, is_last: bool,
                  safe: dict) -> Optional[OverlaySpec]:
    """The overlay a shot needs, or None when it prints nothing."""
    texts = [t for t in shot.overlay_text if t]
    if not texts:
        return None
    common = dict(name=f"overlay_{shot.index:02d}", width=brief.width, height=brief.height, category=brief.category,
                  palette=brand.palette, safe=safe, has_logo=bool(brand.logo_path))
    if is_last and shot.kind == "card":
        legal = [t for t in texts if t.upper().startswith("RERA")]
        return OverlaySpec(kind="end_card", heading=brand.name, cta=script.cta if script.cta in texts else "",
                           legal=legal, contact=[t for t in texts if t not in legal and t != script.cta], **common)
    title = [t for t in texts if t in script.on_screen]
    chips = [t for t in texts if t not in title]
    return OverlaySpec(kind="title", heading=title[0] if title else "", sub=title[1:], chips=chips, **common)


def srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt(timings: list[LineTiming], script: VideoScript, width: int) -> str:
    """SubRip captions: one block per spoken line, two lines at most."""
    blocks = []
    for i, timing in enumerate(timings):
        rows = textwrap.wrap(script.lines[timing.line_index].text, width=width) or [""]
        if len(rows) > 2:
            rows = [rows[0], " ".join(rows[1:])]
        blocks.append(f"{i + 1}\n{srt_time(timing.start_s)} --> {srt_time(timing.end_s)}\n" + "\n".join(rows))
    return "\n\n".join(blocks) + "\n"


def run(storyboard: Storyboard, script: VideoScript, timings: list[LineTiming], brief: VideoBrief,
        brand: BrandKit, out_dir: Path, *, browser: bool = True) -> OverlayResult:
    config = load_yaml_config("video/pacing.yaml")["overlays"]
    safe = video_overlay.safe_zone(brief.aspect)
    result = OverlayResult()

    specs, windows, cursor = [], [], 0.0
    for position, shot in enumerate(storyboard.shots):
        start, end = cursor, cursor + shot.seconds
        cursor = end
        spec = spec_for_shot(position, shot, script, brief, brand, position == len(storyboard.shots) - 1, safe)
        if spec is None:
            continue
        delay = config["title_delay_s"] if (shot.index == 0 and spec.kind == "title") else 0.2
        specs.append((shot, spec))
        windows.append((start if spec.kind == "end_card" else start + delay,
                        end if spec.kind == "end_card" else end - config["tail_s"]))

    if specs:
        try:
            rendered = video_overlay.render_overlays([s for _, s in specs], out_dir / "overlays", browser=browser)
        except Exception as exc:  # noqa: BLE001 - a video without its text is worse than none, but not a failed run
            result.defects.append(f"The on-screen text could not be drawn: {exc}")
            rendered = None
        if rendered is not None:
            result.notes += rendered.notes
            result.defects += [f"On-screen text outside the safe zone: {p}" for p in rendered.problems]
            for (shot, spec), path, (start, end) in zip(specs, rendered.paths, windows):
                result.items.append(OverlayItem(
                    shot_index=shot.index, kind=spec.kind, path=str(path), start_s=round(start, 3),
                    end_s=round(max(end, start + 0.1), 3), renderer=rendered.renderer,
                    texts=[*([spec.heading] if spec.kind == "title" and spec.heading else []), *spec.sub,
                           *spec.chips, *([spec.cta] if spec.cta else []), *spec.legal, *spec.contact]))

    captions = out_dir / "captions.srt"
    captions.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(srt(timings, script, config["caption_chars_per_line"]), encoding="utf-8")
    result.captions_path = str(captions)
    return result
