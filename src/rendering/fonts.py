"""Bundled web fonts, embedded into rendered HTML as data URIs.

Fonts are bundled (assets/fonts, SIL Open Font License, from Fontsource) so
typography is identical on every machine and needs no network at render
time. Each family ships two subsets:

    latin       the alphabet, figures and common punctuation
    latin-ext   extended Latin AND the currency block — the Indian rupee
                sign (U+20B9) lives here

Both subsets are declared with their `unicode-range`, so Chromium downloads
(here: decodes) only what a page uses. Some families do not draw ₹ at all
(Jost, Outfit, and the older Montserrat/Lora latin files): every category's
font stack therefore ends in Inter, which does, so a price never falls back
to a system font or a missing-glyph box. tests/brochure/test_fonts.py
checks that every brochure stack can draw ₹.
"""
from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

FAMILIES = {
    "plus-jakarta-sans": "Plus Jakarta Sans",
    "inter": "Inter",
    "playfair-display": "Playfair Display",
    "outfit": "Outfit",
    "cormorant-garamond": "Cormorant Garamond",
    "jost": "Jost",
    "montserrat": "Montserrat",
    "lora": "Lora",
}

UNICODE_RANGES = {
    "latin": ("U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, "
              "U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, "
              "U+FEFF, U+FFFD"),
    "latin-ext": ("U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, "
                  "U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, "
                  "U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF"),
}

_FILE = re.compile(r"^(?P<family>[a-z-]+?)-(?P<subset>latin-ext|latin)-(?P<weight>\d{3})-(?P<style>normal|italic)$")


def faces() -> list[tuple[str, str, int, str, Path]]:
    """(family, subset, weight, style, path) for every bundled woff2."""
    out = []
    for path in sorted(FONTS_DIR.glob("*.woff2")):
        m = _FILE.match(path.stem)
        if not m or m.group("family") not in FAMILIES:
            continue
        out.append((FAMILIES[m.group("family")], m.group("subset"), int(m.group("weight")),
                    m.group("style"), path))
    return out


@lru_cache(maxsize=8)
def font_face_css(formats: tuple[str, ...] = ("woff2",), families: tuple[str, ...] = ()) -> str:
    """@font-face rules with the font files inlined.

    `families` limits the output to the families a document uses (a
    brochure embeds only its own category's faces)."""
    rules = []
    for family, subset, weight, style, path in faces():
        if families and family not in families:
            continue
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            f"@font-face {{ font-family: '{family}'; font-style: {style}; font-weight: {weight}; "
            f"font-display: block; src: url(data:font/woff2;base64,{data}) format('woff2'); "
            f"unicode-range: {UNICODE_RANGES[subset]}; }}")
    return "\n".join(rules)
