"""Colour, per category — and every pairing that carries type is legible.

Each palette names its colours by JOB, so a template never picks a colour
by eye:

    paper        the sheet
    paper_alt    a second ground: cards, bands, quiet panels
    ink          headings and body on either ground
    ink_soft     secondary type (captions, labels) on either ground
    accent       the brand colour: rules, icons, badges, kickers
    accent_ink   type set ON the accent
    support      a second accent (green / sage / navy): distance badges,
                 sustainability figures
    highlight    a third, used sparingly (amber / ochre / cobalt)
    dark         a deep ground for statement pages and the back cover
    on_dark      type on `dark`
    on_dark_soft secondary type on `dark`
    rule         hairlines and card borders

`check_palette` measures every type pairing against WCAG 2.1 (body 4.5:1,
ink 7:1). The test suite runs it over every palette.

The first palette of each category is the category's reference identity
(Affordable after Mahindra Happinest, Mid-Tier after Brigade Citrine,
Luxury after SOBHA Neopolis). The Creative Director may pick another from
the same closed set to suit a project's own brand; a property may also
carry its own brand accent (PropertyInput.brand_color), which replaces
`accent` only when it passes the same checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Palette:
    name: str
    paper: str
    paper_alt: str
    ink: str
    ink_soft: str
    accent: str
    accent_ink: str
    accent_text: str     # the accent when it carries small type (kickers, prices)
    support: str
    support_text: str
    highlight: str
    dark: str
    on_dark: str
    on_dark_soft: str
    rule: str
    mood: str = ""

    def tokens(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("name", "mood")}


PALETTES: dict[str, tuple[Palette, ...]] = {
    "affordable": (
        Palette("crimson_slate", paper="#FFFFFF", paper_alt="#F4F5F7", ink="#1E293B",
                ink_soft="#475569", accent="#E31837", accent_ink="#FFFFFF", accent_text="#C3132F", support="#15803D", support_text="#15803D",
                highlight="#D97706", dark="#1E293B", on_dark="#FFFFFF", on_dark_soft="#CBD5E1",
                rule="#E2E5EA", mood="crimson and slate on white — bright, trustworthy, value-led"),
        Palette("forest_amber", paper="#FFFFFF", paper_alt="#F2F6F3", ink="#1A2B22",
                ink_soft="#46564D", accent="#15803D", accent_ink="#FFFFFF", accent_text="#15803D", support="#0F766E", support_text="#0F766E",
                highlight="#D97706", dark="#173826", on_dark="#FFFFFF", on_dark_soft="#C9DCD0",
                rule="#DCE6E0", mood="forest green and amber — fresh, green, family-friendly"),
    ),
    "mid": (
        Palette("citrine_bronze", paper="#FAF7F2", paper_alt="#EFECE6", ink="#2C221E",
                ink_soft="#5E524B", accent="#7C4D2B", accent_ink="#FAF7F2", accent_text="#7C4D2B", support="#5F7A61", support_text="#4E6A50",
                highlight="#D48C46", dark="#2C221E", on_dark="#FAF7F2", on_dark_soft="#D9CEC3",
                rule="#D8CBBB", mood="soft sand, espresso and bronze — warm, contemporary, sustainable"),
        Palette("sage_stone", paper="#F7F7F2", paper_alt="#ECEDE6", ink="#22281F",
                ink_soft="#535B4E", accent="#4F6B52", accent_ink="#F7F7F2", accent_text="#4F6B52", support="#7C4D2B", support_text="#7C4D2B",
                highlight="#C58A3F", dark="#26302A", on_dark="#F7F7F2", on_dark_soft="#CED6CC",
                rule="#D3D7CB", mood="sage and stone — calm, green, lifestyle-led"),
    ),
    "luxury": (
        Palette("aegean_champagne", paper="#FDFBF7", paper_alt="#F4EFE6", ink="#18181B",
                ink_soft="#52525B", accent="#8C6B2F", accent_ink="#FDFBF7", accent_text="#7A5C24", support="#1B365D", support_text="#1B365D",
                highlight="#2B5B84", dark="#1B365D", on_dark="#FDFBF7", on_dark_soft="#D9CBA6",
                rule="#C5A059", mood="alabaster, Aegean navy and champagne gold — editorial, timeless"),
        Palette("obsidian_gold", paper="#FDFBF7", paper_alt="#F4EFE6", ink="#18181B",
                ink_soft="#52525B", accent="#8C6B2F", accent_ink="#FDFBF7", accent_text="#7A5C24", support="#18181B", support_text="#18181B",
                highlight="#2B5B84", dark="#18181B", on_dark="#FDFBF7", on_dark_soft="#CDB98A",
                rule="#C5A059", mood="obsidian and gold — nocturnal, dramatic, private"),
    ),
}

# Champagne gold (#C5A059) is the luxury ornament colour: rules, frames,
# numerals set large. It is never small type on the light paper (2.4:1);
# `accent` carries the darker bronze-gold for type.
ORNAMENT = {"luxury": "#C5A059", "mid": "#B98A5E", "affordable": "#E31837"}


# --- WCAG 2.1 -----------------------------------------------------------------


def _channel(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# (foreground, background, minimum)
PAIRINGS = (
    ("ink", "paper", 7.0), ("ink", "paper_alt", 7.0),
    ("ink_soft", "paper", 4.5), ("ink_soft", "paper_alt", 4.5),
    ("accent_text", "paper", 4.5), ("accent_text", "paper_alt", 4.5),
    ("accent_ink", "accent", 4.5),
    ("support_text", "paper", 4.5), ("support_text", "paper_alt", 4.5),
    ("on_dark", "dark", 7.0), ("on_dark_soft", "dark", 4.5),
)


def check_palette(p: Palette) -> list[str]:
    """Every type pairing in this palette that fails its minimum."""
    tokens = p.tokens()
    failures = []
    for fg, bg, minimum in PAIRINGS:
        ratio = contrast(tokens[fg], tokens[bg])
        if ratio < minimum:
            failures.append(f"{p.name}: {fg} on {bg} is {ratio:.2f}:1 (needs {minimum}:1)")
    return failures


def with_brand(p: Palette, brand: Optional[str]) -> Palette:
    """The palette with the property's own brand colour as its accent, if
    that colour can carry type on this palette's grounds."""
    brand = (brand or "").strip()
    if not brand.startswith("#") or len(brand) not in (4, 7):
        return p
    try:
        trial = replace(p, accent=brand, accent_text=brand)
        return trial if not check_palette(trial) else p
    except ValueError:
        return p


def for_category(category: str, name: Optional[str] = None) -> Palette:
    options = PALETTES.get(category) or PALETTES["mid"]
    for p in options:
        if p.name == name:
            return p
    return options[0]


def options(category: str) -> tuple[Palette, ...]:
    return PALETTES.get(category) or PALETTES["mid"]
