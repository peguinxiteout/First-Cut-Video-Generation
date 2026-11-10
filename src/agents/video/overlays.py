"""Node 8, overlays: what the screen prints, and the captions file.

Every shot with text gets transparent pictures of it (rendering/
video_overlay.py), timed to the shot:

    the hook         the name and, where the category shows it, the price
    a feature shot   its chip, when the shot's photograph matches it (the
                     storyboard decided that)
    the closing card the call to action, RERA if the data has one, and contact

The text comes from the storyboard's shots, sorted into those jobs by what
the script says each piece is. Each piece (the title, a chip, a line of the
card) is its own picture so it can come in on its own: it fades in sliding up,
the next piece a beat after it, and on the closing card the lines appear in
sequence.

CAPTIONS. Unless the brief turns them off (`burn_captions`), each spoken line
is also printed, one at a time, from the line timings: a dark box with white
text in the lower part of the safe zone. A caption never covers a title, a
chip, a line of the closing card or the logo's plate: where one is showing at
the same moment it moves above or below it, and a caption with no free place
inside the safe zone is left off the picture (a note says so).

captions.srt is written from the real line timings, so it follows the voice,
whatever the voice was.
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from src.agents.video import motion
from src.core.config import load_yaml_config
from src.data.video_schemas import BrandKit, LineTiming, OverlayItem, Storyboard, VideoBrief, VideoScript
from src.rendering import video_overlay
from src.rendering.video_overlay import OverlaySpec

logger = logging.getLogger(__name__)

CAPTION_HOLD_S = 0.15            # a caption stays this long after its line ends, if the next line is not due


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


# ---------------------------------------------------------------------------
# Titles, chips and the closing card
# ---------------------------------------------------------------------------

def piece_texts(spec: OverlaySpec, key: str) -> list[str]:
    """The words one piece of a title or card prints."""
    if key in ("heading", "wordmark"):
        return [spec.heading]
    for prefix, items in (("sub", spec.sub), ("chip", spec.chips), ("legal", spec.legal), ("contact", spec.contact)):
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return [items[int(key[len(prefix):])]]
    return [spec.cta] if key == "cta" else []


def _offsets(keys: list[str], stagger_s: float, window_s: float) -> list[float]:
    """When each piece comes in, from the start of its overlay: the scrim and the
    first piece together, then one `stagger_s` apart, the last no later than half
    the window."""
    pieces = [k for k in keys if k != "scrim"]
    step = min(stagger_s, (window_s * 0.5) / (len(pieces) - 1)) if len(pieces) > 1 else 0.0
    return [0.0 if k == "scrim" else pieces.index(k) * step for k in keys]


def _item(shot_index: int, spec: OverlaySpec, key: str, path: Path, rects: list[dict], renderer: str,
          window: tuple[float, float], fade_s: float, slide_px: int) -> OverlayItem:
    start, end = window
    return OverlayItem(shot_index=shot_index, kind=spec.kind, key=key, path=str(path), start_s=round(start, 3),
                       end_s=round(max(end, start + 0.1), 3), texts=piece_texts(spec, key) if spec.kind != "caption"
                       else [spec.text], renderer=renderer, fade_s=fade_s, slide_px=slide_px, rects=rects)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def place_caption(height: int, window: tuple[float, float], obstacles: list[tuple[float, float, tuple]],
                  zone: tuple[int, int], default_bottom: int, gap: int) -> Optional[int]:
    """The lower edge for a caption `height` tall shown during `window`: the default
    when it is clear, else the nearest clear place inside `zone` (top, bottom), else
    None. `obstacles` are (from, to, (x, y, w, h)); only their height counts, a
    caption being as wide as it likes."""
    live = [(r[1] - gap, r[1] + r[3] + gap) for start, end, r in obstacles if _overlap(window, (start, end))]
    candidates = [default_bottom, *(top for top, _ in live), *(bottom + height for _, bottom in live)]
    clear = [b for b in candidates if b - height >= zone[0] and b <= zone[1]
             and all(b <= top or b - height >= bottom for top, bottom in live)]
    return min(clear, key=lambda b: abs(b - default_bottom)) if clear else None


def _shot_at(storyboard: Storyboard, t: float) -> int:
    edge = 0.0
    for shot in storyboard.shots:
        edge += shot.seconds
        if t < edge:
            return shot.index
    return storyboard.shots[-1].index


def _caption_windows(timings: list[LineTiming]) -> list[tuple[float, float]]:
    windows = []
    for i, timing in enumerate(timings):
        end = timing.end_s + CAPTION_HOLD_S
        if i + 1 < len(timings):
            end = min(end, timings[i + 1].start_s)
        windows.append((timing.start_s, round(max(end, timing.start_s + 0.2), 3)))
    return windows


def _captions(storyboard: Storyboard, script: VideoScript, timings: list[LineTiming], brief: VideoBrief, safe: dict,
              config: dict, obstacles: list, out_dir: Path, browser: bool, result: OverlayResult) -> None:
    """Draw the captions and add them to result.items, placed clear of `obstacles`."""
    unit = min(brief.width, brief.height) / 1080
    _, y0, _, y1 = video_overlay.safe_rect(brief.width, brief.height, safe)
    default_bottom, gap = round(brief.height * config["bottom"]), round(config["gap_px"] * unit)
    windows = _caption_windows(timings)
    specs = [OverlaySpec(name=f"caption_{i:02d}", kind="caption", width=brief.width, height=brief.height,
                         category=brief.category, palette={}, safe=safe, text=script.lines[t.line_index].text,
                         bottom_y=default_bottom, font_px=config["font_px"], width_share=config["width_share"])
             for i, t in enumerate(timings)]
    try:
        first = video_overlay.render_overlays(specs, out_dir / "overlays", browser=browser)
    except Exception as exc:  # noqa: BLE001 - captions are an addition; the line is still in captions.srt
        result.notes.append(f"The captions could not be drawn on the picture ({exc}); captions.srt has them.")
        return
    result.notes += [n for n in first.notes if n not in result.notes]

    paths, rects, problems = dict(enumerate(first.paths)), dict(enumerate(first.rects)), list(first.problems)
    moved: dict[int, OverlaySpec] = {}
    skipped: set[int] = set()
    for i, (spec, found, window) in enumerate(zip(specs, first.rects, windows)):
        bottom = place_caption(round(found[0]["h"]), window, obstacles, (y0, y1), default_bottom, gap)
        if bottom is None:
            skipped.add(i)
            result.notes.append(f"Caption {i + 1} is not drawn on the picture: there is no free place inside the "
                                "safe zone while the title or card shows (captions.srt has it).")
        elif bottom != default_bottom:
            moved[i] = replace(spec, bottom_y=bottom)
    if moved:
        again = video_overlay.render_overlays(list(moved.values()), out_dir / "overlays", browser=browser)
        for i, path, found in zip(moved, again.paths, again.rects):
            paths[i], rects[i] = path, found
        problems = [p for p in problems if not any(f"caption_{i:02d}" in p for i in moved)] + again.problems
    result.defects += [f"On-screen text outside the safe zone: {p}" for p in problems
                       if not any(f"caption_{i:02d}" in p for i in skipped)]
    for i, (spec, window) in enumerate(zip(specs, windows)):
        if i not in skipped:
            result.items.append(_item(_shot_at(storyboard, window[0]), moved.get(i, spec), "caption", paths[i],
                                      rects[i], first.renderer, window, config["fade_s"], 0))


# ---------------------------------------------------------------------------

def run(storyboard: Storyboard, script: VideoScript, timings: list[LineTiming], brief: VideoBrief,
        brand: BrandKit, out_dir: Path, *, browser: bool = True) -> OverlayResult:
    pacing = load_yaml_config("video/pacing.yaml")
    config, caption_config = pacing["overlays"], pacing["captions"]
    safe = video_overlay.safe_zone(brief.aspect)
    slide = round(config["slide_px"] * min(brief.width, brief.height) / 1080)
    result = OverlayResult()

    groups, cursor = [], 0.0
    for position, shot in enumerate(storyboard.shots):
        start, end = cursor, cursor + shot.seconds
        cursor = end
        spec = spec_for_shot(position, shot, script, brief, brand, position == len(storyboard.shots) - 1, safe)
        if spec is None:
            continue
        card = spec.kind == "end_card"
        delay = config["title_delay_s"] if (shot.index == 0 and spec.kind == "title") else 0.2
        groups.append((shot, spec, (start if card else start + delay, end if card else end - config["tail_s"])))

    # one picture for every piece of every title and card, and when each comes in
    pieces, timed = [], []
    for shot, spec, (start, end) in groups:
        end = max(end, start + 0.1)
        keys = video_overlay.parts(spec)
        stagger = config["card_stagger_s"] if spec.kind == "end_card" else config["title_stagger_s"]
        for key, offset in zip(keys, _offsets(keys, stagger, end - start)):
            pieces.append((shot, replace(spec, show=key, name=f"{spec.name}_{key}"), key))
            timed.append((start + offset, end))

    if pieces:
        try:
            rendered = video_overlay.render_overlays([spec for _, spec, _ in pieces], out_dir / "overlays",
                                                     browser=browser)
        except Exception as exc:  # noqa: BLE001 - a video without its text is worse than none, but not a failed run
            result.defects.append(f"The on-screen text could not be drawn: {exc}")
            rendered = None
        if rendered is not None:
            result.notes += rendered.notes
            result.defects += [f"On-screen text outside the safe zone: {p}" for p in rendered.problems]
            for (shot, spec, key), path, found, window in zip(pieces, rendered.paths, rendered.rects, timed):
                result.items.append(_item(shot.index, spec, key, path, found, rendered.renderer, window,
                                          config["fade_s"], 0 if key == "scrim" else slide))

    if brief.burn_captions and timings:
        obstacles = [(it.start_s, it.end_s, (r["x"], r["y"], r["w"], r["h"])) for it in result.items
                     for r in it.rects]
        for shot, spec, (start, end) in groups:                  # the logo's plate is not text, but it is not for captions
            plate = motion.logo_plate(brand, brief.width, brief.height) if spec.kind == "end_card" else None
            if plate:
                obstacles.append((start, end, (plate[0], plate[1], plate[2] - plate[0], plate[3] - plate[1])))
        _captions(storyboard, script, timings, brief, safe, caption_config, obstacles, out_dir, browser, result)

    captions = out_dir / "captions.srt"
    captions.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(srt(timings, script, config["caption_chars_per_line"]), encoding="utf-8")
    result.captions_path = str(captions)
    return result
