"""Draw a planned and written Book as HTML.

The Book carries everything: each page's layout and slots in millimetres,
the images placed in them, the structured content and the copy. This module
supplies the template with what a template cannot compute: file URIs, the
icons, a QR code, the category's design tokens, its palette and fonts, and
the typesetting filters that keep units, prices and figures from breaking
across lines.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from src.agents.brochure import facts, voice
from src.agents.brochure.book import Book
from src.agents.brochure.catalog import TemplateSpec
from src.agents.brochure.palettes import ORNAMENT, Palette
from src.core.config import PROJECT_ROOT
from src.rendering import icons
from src.rendering.fonts import font_face_css
from src.rendering.renderer import qr_code_data_uri

TEMPLATES_DIR = PROJECT_ROOT / "templates" / "brochure"

NBSP = " "
# A number and its unit, a currency and its figure, a label and its number:
# these never break across a line.
_GLUE = [
    (re.compile(r"(\d\+?)\s+(?=(?:BHK|sq ft|sq m|sq\.ft|km|m|min|acres?|Acres?|Lakh|Cr|tonnes|towers|"
                r"Towers|trees|units|Units|residences|quality checks)\b)"), r"\1" + NBSP),
    (re.compile(r"₹\s+"), "₹"),
    (re.compile(r"\b(Line|Tower|Towers|Phase|Type|No\.|Sector|Gate)\s+(?=[\dA-Z])"), r"\1" + NBSP),
    (re.compile(r"(\d)\s+–\s+(?=[\d₹])"), r"\1" + NBSP + "–" + NBSP),
    (re.compile(r"\s+\+\s+(?=(?:Study|Staff))"), NBSP + "+" + NBSP),
]


def nobreak(value) -> Markup:
    text = str(value or "")
    for pattern, repl in _GLUE:
        text = pattern.sub(repl, text)
    # never a single short word alone on the last line of a paragraph
    text = re.sub(r"\s+(\S{1,12})$", NBSP + r"\1", text)
    return Markup(escape(text))


def capitalize_first(value) -> str:
    return facts.capitalise(str(value or ""))


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)),
                      autoescape=select_autoescape(["html"]))
    env.filters["nobreak"] = nobreak
    env.filters["capitalize_first"] = capitalize_first
    return env


@lru_cache(maxsize=32)
def logo_traits(path: str) -> tuple[bool, float]:
    """(transparent, mean luminance of the mark) for a logo file."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            rgba.thumbnail((200, 200))
            px = list(rgba.getdata())
    except Exception:  # noqa: BLE001
        return False, 0.5
    transparent = sum(1 for p in px if p[3] < 30) > len(px) * 0.05
    marks = [p for p in px if p[3] > 128]
    if not marks:
        return transparent, 0.5
    lum = sum(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in marks) / (255 * len(marks))
    return transparent, lum


def _km(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?\s*(km|m)\b", text or "")
    if not m:
        return None
    value = float(m.group(2) or m.group(1)) if m.group(2) else float(m.group(1))
    return value / 1000 if m.group(3) == "m" else value


def distance_rings(places: list) -> Markup:
    """A plan of the neighbourhood drawn from the distances the person gave:
    the project at the centre, rings at round distances, each place on its
    ring. Drawn only when at least three places have a distance in km or m;
    it is a diagram of stated distances, not a map, and says so."""
    pts = [(p, _km(p.get("distance", ""))) for p in places]
    pts = [(p, d) for p, d in pts if d is not None]
    if len(pts) < 3:
        return Markup("")
    far = max(d for _, d in pts)
    steps = [s for s in (0.5, 1, 2, 3, 4, 5, 8, 10, 15, 20, 30) if s <= far * 1.05] or [far]
    rings = steps[-3:] if len(steps) > 3 else steps
    R = 46.0
    scale = R / max(rings[-1], far)
    import math
    parts = ['<svg viewBox="-80 -54 160 108" class="rings-svg" aria-label="Distances from the project">']
    for r in rings:
        rr = r * scale
        label = f"{r:g} km" if r >= 1 else f"{int(r * 1000)} m"
        parts.append(f'<circle r="{rr:.2f}" class="ring"/><text x="{rr * 0.71 + 1:.2f}" y="{-rr * 0.71 - 1:.2f}" class="ring-l">{label}</text>')
    parts.append('<circle r="3.2" class="hub"/><circle r="6" class="hub-halo"/>')
    n = len(pts)
    for i, (p, d) in enumerate(sorted(pts, key=lambda x: x[1])):
        ang = -math.pi / 2 + i * 2 * math.pi / n + 0.35
        rr = max(9.0, d * scale)
        x, y = rr * math.cos(ang), rr * math.sin(ang)
        anchor = "start" if x >= 0 else "end"
        dx = 3.2 if x >= 0 else -3.2
        raw = str(p.get("name", ""))
        if len(raw) > 30:
            # a long name is shortened at a word, never mid-word
            raw = re.split(r"\s*[/(,]\s*", raw)[0]
            while len(raw) > 30 and " " in raw:
                raw = raw.rsplit(" ", 1)[0]
        name = escape(raw)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.7" class="pt"/>'
                     f'<text x="{x + dx:.2f}" y="{y + 1.1:.2f}" text-anchor="{anchor}" class="pt-l">{name}</text>')
    parts.append('</svg><p class="rings-note">Distances as stated; diagram not to scale.</p>')
    return Markup("".join(parts))


BLEED_MM = 3.0
PRINT_DPI = 300


def _focal(position: str) -> tuple[float, float]:
    try:
        x, y = (float(v.strip().rstrip("%")) / 100 for v in position.split()[:2])
        return x, y
    except (ValueError, AttributeError):
        return 0.5, 0.5


def placed_photo(placed, source: str, out_dir: Path) -> str:
    """The photograph cut to its slot — bleed included — at the planned focal
    point, and no larger than 300 dpi at its printed size, saved as a JPEG.

    Chromium re-encodes an image it has to crop (object-fit: cover) as a
    lossless bitmap, which made a ten-page brochure 40 MB. An image that
    already has its slot's proportions is drawn whole and embedded as the
    JPEG it is. Nothing is ever enlarged."""
    from PIL import Image

    bt, br, bb, bl = (BLEED_MM if side else 0.0 for side in placed.bleed)
    w_mm, h_mm = placed.w + bl + br, placed.h + bt + bb
    key = f"{Path(source).stem}_{w_mm:.1f}x{h_mm:.1f}_{placed.position.replace('%', '').replace(' ', '-')}"
    out = out_dir / f"{key}.jpg"
    if out.exists():
        return str(out)
    try:
        with Image.open(source) as im:
            im = im.convert("RGB")
            W, H = im.size
            target = w_mm / h_mm
            if W / H > target:                      # too wide: lose the sides
                cw, ch = H * target, H
            else:                                   # too tall: lose top/bottom at the focal point
                cw, ch = W, W / target
            fx, fy = _focal(placed.position)
            x0 = max(0.0, min(W - cw, (W - cw) * fx))
            y0 = max(0.0, min(H - ch, (H - ch) * fy))
            crop = im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch)))
            max_w = round(w_mm / 25.4 * PRINT_DPI)
            if crop.width > max_w:
                crop = crop.resize((max_w, max(1, round(max_w / target))), Image.LANCZOS)
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out, "JPEG", quality=88, optimize=True, progressive=True, subsampling=0)
        return str(out)
    except Exception:  # noqa: BLE001 - the original, cropped by the browser, is still correct
        return source


LEGAL = ("Plans, specifications, images and visuals are indicative and subject to change as per the "
         "approvals and the agreement for sale. This brochure is not an offer, an invitation to offer "
         "or a contract. Please verify all details with the developer before purchase.")


def render_book(book: Book, prop, spec: TemplateSpec, palette: Palette,
                images_by_id: dict, output_path: Path) -> Path:
    content = facts.build(prop, book.category)
    subjects = {i: book.roles.get(f"topic:{i}", "") for i in images_by_id}

    def image_uri(image_id: Optional[str]) -> Optional[str]:
        if not image_id or image_id not in images_by_id:
            return None
        return Path(images_by_id[image_id].path).resolve().as_uri()

    placed_dir = output_path.parent / "placed"

    def placed_uri(placed) -> Optional[str]:
        if placed.image_id not in images_by_id:
            return None
        if placed.kind != "image":
            return image_uri(placed.image_id)
        return Path(placed_photo(placed, images_by_id[placed.image_id].path, placed_dir)).resolve().as_uri()

    def caption_for(image_id: str) -> str:
        return voice.CAPTIONS.get(subjects.get(image_id, ""), "")

    needs_plate = False
    if book.logo and book.logo in images_by_id:
        transparent, lum = logo_traits(images_by_id[book.logo].path)
        # a dark mark on a transparent ground disappears on a dark page
        needs_plate = transparent and lum < 0.55

    design = spec.design
    html = _env().get_template("base.html").render(
        book=book, spec=spec, palette=palette, content=content, design=design, t=design.type,
        margins=[round(v, 2) for v in spec.margins()], ornament=ORNAMENT.get(book.category, palette.accent),
        font_face_css=font_face_css(("woff2",), design.families),
        icon=icons.svg, image_uri=image_uri, placed_uri=placed_uri, caption_for=caption_for,
        logo_needs_plate=needs_plate,
        distance_rings=distance_rings,
        qr_code_uri=lambda data: qr_code_data_uri(data, dark_color="#111111"),
        legal_line=LEGAL,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
