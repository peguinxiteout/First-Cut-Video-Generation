"""The brochure templates: three categories x three formats.

    Affordable - Landscape   Affordable - Portrait   Affordable - Digital 16:9
    Mid-Tier   - Landscape   Mid-Tier   - Portrait   Mid-Tier   - Digital 16:9
    Luxury     - Landscape   Luxury     - Portrait   Luxury     - Digital 16:9

A template is two independent choices:

    the CATEGORY  its design system (`Design`): typefaces, the type scale in
                  points, margins, gutter, corner radius, the white-space it
                  aims for — and, in strategy.py, what it says and in which
                  order. The three differ in hierarchy, composition and
                  voice, not only colour.
    the SHEET     the paper. Print is A4 (297 x 210 mm landscape, 210 x 297
                  mm portrait). The digital variant is 16:9, authored at
                  A4 width (297 x 167.06 mm) so the type scale is shared and
                  exported to 1920 x 1080 px images.

Everything is authored in millimetres and points — the browser lays the
page out at its printed size, so 10pt in the stylesheet is 10pt on paper.
Every page is composed on a 12-column grid with a ~4mm gutter and a 4pt
baseline (layouts.py).

Old template ids (`luxury_modern`, `meridian`, `compass`) are accepted as
aliases because stored generation records and API callers use them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PT_TO_MM = 25.4 / 72.0
BASELINE_MM = 4 * PT_TO_MM          # the 4pt baseline rhythm, 1.411mm


def snap(value_mm: float) -> float:
    """Round a vertical measure to the baseline."""
    return round(value_mm / BASELINE_MM) * BASELINE_MM


@dataclass(frozen=True)
class Sheet:
    name: str
    w: float                         # mm
    h: float                         # mm
    digital: bool = False

    @property
    def landscape(self) -> bool:
        return self.w > self.h

    @property
    def trim_w_pt(self) -> float:
        return self.w / PT_TO_MM

    @property
    def trim_h_pt(self) -> float:
        return self.h / PT_TO_MM

    @property
    def aspect(self) -> float:
        return self.w / self.h


A4_LANDSCAPE = Sheet("a4_landscape", 297.0, 210.0)
A4_PORTRAIT = Sheet("a4_portrait", 210.0, 297.0)
DIGITAL = Sheet("digital_16x9", 297.0, 167.0625, digital=True)


@dataclass(frozen=True)
class TypeScale:
    """Point sizes and line heights (as multiples of the size)."""
    display: float        # the cover's name
    h1: float
    h2: float
    h3: float             # card / unit titles
    lead: float           # standfirst
    body: float
    small: float          # list items, table rows
    caption: float
    kicker: float
    legal: float
    figure: float         # a statistic set large
    lh_h1: float = 1.12
    lh_h2: float = 1.3
    lh_body: float = 1.55


@dataclass(frozen=True)
class Design:
    category: str
    heading_font: str
    body_font: str
    numeral_font: str                 # tabular figures in tables and prices
    heading_weight: int
    body_weight: int
    type: TypeScale
    margins: tuple[float, float, float, float]    # top right bottom left, mm
    gutter: float                     # mm
    radius: float                     # mm (cards)
    whitespace: tuple[float, float]   # the share of a page it aims to leave open
    # average glyph width of the body face, in em — for copy budgets
    char_em: float = 0.5
    heading_char_em: float = 0.52
    tracking_kicker: str = "0.16em"
    families: tuple[str, ...] = field(default_factory=tuple)

    def stack(self, family: str, serif: bool = False) -> str:
        fallback = "Georgia, serif" if serif else "Arial, sans-serif"
        # Inter is in every stack because it draws the rupee sign; Jost and
        # Outfit do not (src/rendering/fonts.py).
        return f"'{family}', 'Inter', {fallback}"


AFFORDABLE = Design(
    category="affordable",
    heading_font="Plus Jakarta Sans", body_font="Inter", numeral_font="Inter",
    heading_weight=800, body_weight=400,
    type=TypeScale(display=40, h1=24, h2=14, h3=11.5, lead=11.5, body=9.6, small=8.8, caption=8,
                   kicker=7.5, legal=6.5, figure=26, lh_h1=1.12, lh_h2=1.28, lh_body=1.5),
    margins=(12, 12, 12, 12), gutter=4.0, radius=2.2, whitespace=(0.18, 0.27),
    char_em=0.53, heading_char_em=0.58, tracking_kicker="0.12em",
    families=("Plus Jakarta Sans", "Inter"),
)

MID = Design(
    category="mid",
    heading_font="Playfair Display", body_font="Outfit", numeral_font="Inter",
    heading_weight=500, body_weight=400,
    type=TypeScale(display=46, h1=28, h2=15, h3=12.5, lead=12.5, body=10.2, small=9.2, caption=8,
                   kicker=7.5, legal=6.5, figure=30, lh_h1=1.12, lh_h2=1.3, lh_body=1.58),
    margins=(16, 18, 16, 18), gutter=4.0, radius=0.6, whitespace=(0.28, 0.37),
    char_em=0.5, heading_char_em=0.52, tracking_kicker="0.2em",
    families=("Playfair Display", "Outfit", "Inter"),
)

LUXURY = Design(
    category="luxury",
    heading_font="Cormorant Garamond", body_font="Jost", numeral_font="Jost",
    heading_weight=400, body_weight=300,
    type=TypeScale(display=58, h1=36, h2=17, h3=14, lead=13.5, body=10.6, small=9.4, caption=8,
                   kicker=7.2, legal=6.5, figure=38, lh_h1=1.1, lh_h2=1.28, lh_body=1.65),
    margins=(22, 25, 22, 25), gutter=4.5, radius=0.0, whitespace=(0.36, 0.52),
    char_em=0.49, heading_char_em=0.42, tracking_kicker="0.3em",
    families=("Cormorant Garamond", "Jost", "Inter"),
)

DESIGNS = {d.category: d for d in (AFFORDABLE, MID, LUXURY)}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 1.0


@dataclass(frozen=True)
class TemplateSpec:
    id: str
    name: str                 # "Luxury - Landscape"
    category: str             # affordable | mid | luxury
    orientation: str          # landscape | portrait | digital
    sheet: Sheet
    design: Design

    @property
    def landscape(self) -> bool:
        return self.sheet.landscape

    @property
    def portrait(self) -> bool:
        return not self.sheet.landscape

    def margins(self) -> tuple[float, float, float, float]:
        t, r, b, l = self.design.margins
        if self.sheet.digital:            # a shorter sheet keeps proportionate head room
            k = 0.66 if self.category == "luxury" else 0.8
            t, b = t * k, b * k
        if self.portrait:                 # portrait swaps the long and short sides
            t, r, b, l = r, t, l, b
        return t, r, b, l

    def frame(self) -> Rect:
        t, r, b, l = self.margins()
        return Rect(l, t, self.sheet.w - l - r, self.sheet.h - t - b)


CATEGORY_LABEL = {"affordable": "Affordable", "mid": "Mid-Tier", "luxury": "Luxury"}
FORMAT_LABEL = {"landscape": "Landscape", "portrait": "Portrait", "digital": "Digital 16:9"}
SHEETS = {"landscape": A4_LANDSCAPE, "portrait": A4_PORTRAIT, "digital": DIGITAL}


def _spec(category: str, orientation: str) -> TemplateSpec:
    return TemplateSpec(id=f"{category}_{orientation}",
                        name=f"{CATEGORY_LABEL[category]} - {FORMAT_LABEL[orientation]}",
                        category=category, orientation=orientation, sheet=SHEETS[orientation],
                        design=DESIGNS[category])


TEMPLATES: dict[str, TemplateSpec] = {
    t.id: t for t in (_spec(c, o) for c in ("affordable", "mid", "luxury")
                      for o in ("landscape", "portrait", "digital"))
}

ALIASES = {
    "luxury_modern": "luxury_landscape",
    "signature": "luxury_landscape",
    "meridian": "mid_landscape",
    "compass": "affordable_portrait",
}

DEFAULT_TEMPLATE = "luxury_landscape"


def normalise_orientation(value: Optional[str]) -> str:
    text = (value or "landscape").strip().lower()
    if text.startswith("p"):
        return "portrait"
    if text.startswith("d") or "16" in text:
        return "digital"
    return "landscape"


def resolve(template: Optional[str] = None, category: Optional[str] = None,
            orientation: Optional[str] = None) -> TemplateSpec:
    """The template a request means, however it named it.

    Accepts an id ("mid_portrait"), a display name ("Mid-Tier - Portrait"),
    an old id ("meridian"), or a category and an orientation."""
    key = (template or "").strip()
    if key in TEMPLATES:
        return TEMPLATES[key]
    if key in ALIASES:
        return TEMPLATES[ALIASES[key]]
    for spec in TEMPLATES.values():
        if key and key.lower() == spec.name.lower():
            return spec
    cat = normalise_category(category)
    if cat:
        return TEMPLATES[f"{cat}_{normalise_orientation(orientation)}"]
    return TEMPLATES[DEFAULT_TEMPLATE]


def normalise_category(value: Optional[str]) -> str:
    """'Mid-Tier', 'mid tier', 'Mid', 'premium' -> 'mid'; '' -> ''."""
    text = (value or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return ""
    if text.startswith("afford") or text in ("budget", "value", "economy"):
        return "affordable"
    if text.startswith("lux") or text in ("ultra", "ultra luxury", "super luxury"):
        return "luxury"
    if text.startswith("mid") or text in ("premium", "upper mid", "middle"):
        return "mid"
    return ""


def choices() -> list[tuple[str, str]]:
    """(id, display name) in the order a form should offer them."""
    return [(t.id, t.name) for t in TEMPLATES.values()]
