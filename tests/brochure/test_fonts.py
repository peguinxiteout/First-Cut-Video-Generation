"""Every typeface a brochure asks for is bundled, and every category's
font stack can draw the rupee sign — the corrupted ₹ of earlier brochures
was a fallback to a font without the glyph."""
from __future__ import annotations

import pytest

from src.agents.brochure.catalog import DESIGNS
from src.rendering import fonts


def _cmap(path):
    from fontTools.ttLib import TTFont

    return TTFont(str(path)).getBestCmap()


@pytest.mark.parametrize("category", ["affordable", "mid", "luxury"])
def test_every_family_is_bundled(category):
    bundled = {family for family, *_ in fonts.faces()}
    for family in DESIGNS[category].families:
        assert family in bundled, family


@pytest.mark.parametrize("category", ["affordable", "mid", "luxury"])
def test_the_rupee_sign_is_drawn_by_the_stack(category):
    pytest.importorskip("fontTools")
    pytest.importorskip("brotli")
    design = DESIGNS[category]
    stack = [f.strip(" '") for f in design.stack(design.body_font).split(",")]
    drawn = False
    for family in stack:
        for fam, subset, _w, _s, path in fonts.faces():
            if fam == family and subset == "latin-ext" and 0x20B9 in _cmap(path):
                drawn = True
                break
        if drawn:
            break
    assert drawn, f"no bundled face in {stack} draws ₹"


def test_faces_declare_their_unicode_range():
    css = fonts.font_face_css(("woff2",), ("Inter",))
    assert "unicode-range" in css and "U+20AD-20C0" in css
    assert "font-family: 'Inter'" in css and "Jost" not in css
