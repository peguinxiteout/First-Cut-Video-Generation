"""The nine templates, the design system's numbers, the three positionings
and the palettes' contrast."""
from __future__ import annotations

import pytest

from src.agents.brochure import catalog, palettes, strategy

NAMES = [f"{c} - {f}" for c in ("Affordable", "Mid-Tier", "Luxury")
         for f in ("Landscape", "Portrait", "Digital 16:9")]


def test_exactly_the_nine_named_templates():
    assert [name for _id, name in catalog.choices()] == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_every_template_resolves_by_name_and_id(name):
    spec = catalog.resolve(name)
    assert spec.name == name
    assert catalog.resolve(spec.id) is spec
    frame = spec.frame()
    assert 0 < frame.w < spec.sheet.w and 0 < frame.h < spec.sheet.h


def test_print_is_a4_and_digital_is_16_9():
    assert (catalog.A4_LANDSCAPE.w, catalog.A4_LANDSCAPE.h) == (297.0, 210.0)
    assert (catalog.A4_PORTRAIT.w, catalog.A4_PORTRAIT.h) == (210.0, 297.0)
    assert catalog.DIGITAL.w / catalog.DIGITAL.h == pytest.approx(16 / 9, rel=1e-3)


def test_category_and_orientation_choose_the_template():
    assert catalog.resolve(None, "Mid-Tier", "Portrait").id == "mid_portrait"
    assert catalog.resolve(None, "luxury", "landscape").id == "luxury_landscape"
    assert catalog.resolve(None, "Affordable", "Digital 16:9").id == "affordable_digital"
    assert catalog.resolve("luxury_modern").id == "luxury_landscape"      # old ids still work


@pytest.mark.parametrize("raw,expected", [("Mid-Tier", "mid"), ("mid tier", "mid"), ("Premium", "mid"),
                                          ("Luxury", "luxury"), ("affordable", "affordable"), ("", "")])
def test_category_names_normalise(raw, expected):
    assert catalog.normalise_category(raw) == expected


# the type scale the brief asks for, per category (points)
TYPE = {"affordable": ((22, 26), (13, 15), (9.5, 10)),
        "mid": ((26, 30), (14, 16), (10, 10.5)),
        "luxury": ((32, 40), (16, 18), (10.5, 11))}


@pytest.mark.parametrize("category", ["affordable", "mid", "luxury"])
def test_type_scale_and_margins_follow_the_brief(category):
    d = catalog.DESIGNS[category]
    (h1lo, h1hi), (h2lo, h2hi), (blo, bhi) = TYPE[category]
    assert h1lo <= d.type.h1 <= h1hi and h2lo <= d.type.h2 <= h2hi and blo <= d.type.body <= bhi
    assert 6 <= d.type.kicker <= 9 and d.type.caption == 8 and d.type.legal >= 6
    assert 1.1 <= d.type.lh_h1 <= 1.15 and 1.45 <= d.type.lh_body <= 1.65
    assert 3.5 <= d.gutter <= 4.5
    t, r, b, l = d.margins
    assert {"affordable": 12, "mid": 16, "luxury": 22}[category] == t
    assert d.radius == (0 if category == "luxury" else d.radius)


def test_the_categories_look_different():
    a, m, lux = (catalog.DESIGNS[c] for c in ("affordable", "mid", "luxury"))
    assert len({a.heading_font, m.heading_font, lux.heading_font}) == 3
    assert len({a.body_font, m.body_font, lux.body_font}) == 3
    assert a.whitespace[1] < m.whitespace[1] < lux.whitespace[1]


def test_positionings_are_genuinely_different():
    a, m, lux = (strategy.for_category(c) for c in ("affordable", "mid", "luxury"))
    assert a.show_price and m.show_price and not lux.show_price
    assert a.book.index("residences") < a.book.index("amenities")     # value first
    assert lux.book.index("residences") > lux.book.index("amenities")  # experience first
    assert "contents" not in a.book + m.book + lux.book                # no generic pages
    assert "booking" not in a.book + m.book + lux.book
    assert "value for money" in lux.avoid and "palatial" in a.avoid


@pytest.mark.parametrize("category", ["affordable", "mid", "luxury"])
def test_every_palette_passes_contrast(category):
    for palette in palettes.options(category):
        assert palettes.check_palette(palette) == []


def test_a_brand_colour_is_used_only_when_it_stays_legible():
    base = palettes.for_category("affordable")
    assert palettes.with_brand(base, "#C3132F").accent == "#C3132F"
    assert palettes.with_brand(base, "#FFFF00").accent == base.accent       # too pale for type
    assert palettes.with_brand(base, "not a colour") is base


def test_contrast_maths():
    assert palettes.contrast("#000000", "#FFFFFF") == pytest.approx(21.0, rel=0.01)
    assert palettes.contrast("#777777", "#777777") == pytest.approx(1.0)
