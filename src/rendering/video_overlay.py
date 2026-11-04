"""On-screen text as transparent pictures.

Every word a video prints (the name, the price, a highlight, the call to
action, RERA, contact) is drawn here by our own code, never by a video model,
so it is exactly as spelled and in the brand's type. One transparent PNG per
shot; ffmpeg lays it over the shot (rendering/video_compose.py).

Two renderers:

    browser    Chromium, in its own process (_video_overlay_worker.py, as for
               the brochure), from templates/video/overlay.html.j2, with the
               brochure palette and bundled fonts. Handles any script and the
               rupee sign.
    fallback   Pillow, when Chromium cannot run: English only, one built-in
               font, "Rs" for the rupee sign. Plainer, never missing.

Both keep every piece of text inside the safe zone (configs/video/styles.yaml:
the bands where a platform draws its own controls are left clear), and both
report where the text landed so the zone can be checked, not assumed.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from PIL import Image, ImageDraw, ImageFont

from src.agents.brochure.catalog import DESIGNS
from src.core.config import load_yaml_config, settings
from src.core.exceptions import RenderingError
from src.rendering.fonts import font_face_css

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates" / "video"
_WORKER = Path(__file__).resolve().parent / "_video_overlay_worker.py"
_SERIF_HEADINGS = ("Playfair Display", "Cormorant Garamond")
TOLERANCE_PX = 1.5


@dataclass
class OverlaySpec:
    name: str                                   # the file stem: overlay_00
    kind: str                                   # title | end_card
    width: int
    height: int
    category: str
    palette: dict
    safe: dict                                  # top, bottom, left, right: shares of the frame
    heading: str = ""
    sub: list[str] = field(default_factory=list)
    chips: list[str] = field(default_factory=list)
    cta: str = ""
    legal: list[str] = field(default_factory=list)
    contact: list[str] = field(default_factory=list)
    has_logo: bool = False

    @property
    def unit(self) -> float:
        return min(self.width, self.height) / 1080


@dataclass
class Rendered:
    paths: list[Path]
    renderer: str                               # browser | fallback
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)       # text outside the safe zone


def safe_zone(aspect: str) -> dict:
    return dict(load_yaml_config("video/styles.yaml")["safe_zones"][aspect])


def safe_rect(width: int, height: int, safe: dict) -> tuple[int, int, int, int]:
    return (round(width * safe["left"]), round(height * safe["top"]),
            round(width * (1 - safe["right"])), round(height * (1 - safe["bottom"])))


def check_rects(spec: OverlaySpec, rects: list[dict]) -> list[str]:
    """The pieces of text that are not wholly inside the safe zone."""
    x0, y0, x1, y1 = safe_rect(spec.width, spec.height, spec.safe)
    return [f"'{r['text']}' is outside the safe zone" for r in rects
            if r["x"] < x0 - TOLERANCE_PX or r["y"] < y0 - TOLERANCE_PX
            or r["x"] + r["w"] > x1 + TOLERANCE_PX or r["y"] + r["h"] > y1 + TOLERANCE_PX]


# ---------------------------------------------------------------------------
# browser
# ---------------------------------------------------------------------------

def build_html(spec: OverlaySpec) -> str:
    design = DESIGNS.get(spec.category) or DESIGNS["mid"]
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    x0, y0, x1, y1 = safe_rect(spec.width, spec.height, spec.safe)
    unit = spec.unit
    return env.get_template("overlay.html.j2").render(
        width=spec.width, height=spec.height, x0=x0, y0=y0, x1=x1, y1=y1, kind=spec.kind, p=spec.palette,
        s=lambda n: round(n * unit), font_face_css=font_face_css(("woff2",), design.families),
        heading_stack=design.stack(design.heading_font, serif=design.heading_font in _SERIF_HEADINGS),
        body_stack=design.stack(design.body_font), heading=spec.heading, sub=spec.sub, chips=spec.chips,
        cta=spec.cta, legal=spec.legal, contact=spec.contact, has_logo=spec.has_logo,
        card_top=max(0, round(spec.height * (0.50 if spec.has_logo else 0.36)) - y0))


def _render_browser(specs: list[OverlaySpec], paths: list[Path], work_dir: Path) -> list[list[dict]]:
    jobs = []
    for spec, png in zip(specs, paths):
        html = work_dir / f"{spec.name}.html"
        html.write_text(build_html(spec), encoding="utf-8")
        jobs.append({"html": str(html), "png": str(png), "width": spec.width, "height": spec.height})
    jobs_path = work_dir / "jobs.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    measure_path = Path(str(jobs_path) + ".measure.json")
    measure_path.unlink(missing_ok=True)
    try:
        result = subprocess.run([sys.executable, str(_WORKER), str(jobs_path), settings.playwright_browser_channel],
                                capture_output=True, text=True, timeout=settings.playwright_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RenderingError("Chromium did not finish drawing the overlays in time") from exc
    if result.returncode != 0:
        raise RenderingError((result.stderr or result.stdout).strip()[-300:] or f"exit code {result.returncode}")
    if not measure_path.exists() or not all(p.exists() and p.stat().st_size > 0 for p in paths):
        raise RenderingError("Chromium reported success but the overlay pictures are missing")
    return [entry["rects"] for entry in json.loads(measure_path.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------

def _hex(colour: str, default: str) -> tuple[int, int, int]:
    h = (colour or default).lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _plain(text: str) -> str:
    return text.replace("₹", "Rs ")            # the built-in font does not draw the rupee sign


def _render_fallback(spec: OverlaySpec, dest: Path) -> list[dict]:
    u = spec.unit
    x0, y0, x1, y1 = safe_rect(spec.width, spec.height, spec.safe)
    safe_w = x1 - x0
    image = Image.new("RGBA", (spec.width, spec.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    accent = _hex(spec.palette.get("accent"), "#B98A5E")
    on_accent = _hex(spec.palette.get("accent_ink"), "#FFFFFF")
    rects: list[dict] = []

    def font(size: int):
        return ImageFont.load_default(size=round(size * u))

    def box(text: str, f) -> tuple[int, int]:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
        return right - left, bottom          # the height from the drawing point to the foot of the glyphs

    def wrap(text: str, f) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if line and box(trial, f)[0] > safe_w:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines + [line] if line else lines

    def place(text: str, f, y: int, colour, centre: bool = False, pill=None, pad: tuple[int, int] = (0, 0)):
        """Draw one piece of text with its top at y; returns the height used."""
        w, h = box(text, f)
        x = (spec.width - (w + 2 * pad[0])) // 2 if centre else x0
        if pill:
            draw.rounded_rectangle((x, y, x + w + 2 * pad[0], y + h + 2 * pad[1]), radius=(h + 2 * pad[1]) // 2,
                                   fill=pill)
        draw.text((x + pad[0], y + pad[1]), text, font=f, fill=colour, anchor="la")
        rects.append({"x": x, "y": y, "w": w + 2 * pad[0], "h": h + 2 * pad[1], "text": text})
        return h + 2 * pad[1]

    gap = round(20 * u)
    if spec.kind == "title":
        for y in range(round(spec.height * 0.54), spec.height):                     # the scrim
            alpha = round(168 * (y - spec.height * 0.54) / (spec.height * 0.46))
            draw.line([(0, y), (spec.width, y)], fill=(0, 0, 0, alpha))
        cursor = y1
        heading_font = font(92)
        for line in reversed(wrap(_plain(spec.heading), heading_font)) if spec.heading else []:
            h = box(line, heading_font)[1]
            cursor -= h
            place(line, heading_font, cursor, (255, 255, 255, 255))
            cursor -= round(8 * u)
        for text in reversed(spec.sub):
            f = font(50)
            h = box(_plain(text), f)[1] + 2 * round(8 * u)
            cursor -= gap + h
            place(_plain(text), f, cursor, on_accent, pill=accent, pad=(round(22 * u), round(8 * u)))
        for text in reversed(spec.chips):
            f = font(44)
            h = box(_plain(text), f)[1] + 2 * round(12 * u)
            cursor -= gap + h
            place(_plain(text), f, cursor, on_accent, pill=accent, pad=(round(30 * u), round(12 * u)))
    else:
        cursor = round(spec.height * (0.50 if spec.has_logo else 0.36))
        if not spec.has_logo and spec.heading:
            f = font(84)
            cursor += place(_plain(spec.heading), f, cursor, (255, 255, 255, 255), centre=True) + gap
        if spec.cta:
            cursor += place(_plain(spec.cta), font(62), cursor, on_accent, centre=True, pill=accent,
                            pad=(round(54 * u), round(20 * u))) + gap
        for text in spec.legal:
            cursor += place(_plain(text), font(36), cursor, (200, 200, 205, 255), centre=True) + gap // 2
        for text in spec.contact:
            cursor += place(_plain(text), font(46), cursor, (255, 255, 255, 255), centre=True) + gap // 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "PNG")
    return rects


# ---------------------------------------------------------------------------

def render_overlays(specs: list[OverlaySpec], out_dir: Path, *, browser: bool = True) -> Rendered:
    """A PNG for every spec in out_dir. Chromium first; Pillow when Chromium
    cannot run (or `browser` is False). Never raises for want of a browser."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{spec.name}.png" for spec in specs]
    notes: list[str] = []
    rects = None
    renderer = "browser"
    if browser and specs:
        try:
            rects = _render_browser(specs, paths, out_dir)
        except Exception as exc:  # noqa: BLE001 - no browser means the plainer renderer, not no overlays
            logger.info("Overlay browser render failed (%s); using the fallback", type(exc).__name__)
            notes.append("The overlays were drawn by the fallback renderer (Chromium could not run: "
                         f"{str(exc)[:120]}): English only, one plain font.")
    if rects is None:
        renderer = "fallback"
        rects = [_render_fallback(spec, path) for spec, path in zip(specs, paths)]
    problems = [f"{spec.name}: {problem}" for spec, found in zip(specs, rects) for problem in check_rects(spec, found)]
    return Rendered(paths, renderer, notes, problems)
