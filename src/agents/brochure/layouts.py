"""The page compositions — every page drawn on a 12-column grid.

A layout is a function of the template (sheet, margins, gutter) that
returns named SLOTS in millimetres:

    image slots   "img", "img2" ...  a photograph, cropped to the slot
                                     (object-fit: cover) at a focal point
                                     that keeps tower crowns and rooflines
    plan slots    "plan", "plan2" ...a drawing, shown WHOLE (contain) on a
                                     quiet card — plans are never cropped
    text slots    "text", "aside" ...where the page's copy flows

The planner chooses a layout per page from the material the page has (how
many photographs, their shapes, how much copy), so the book is not one
composition repeated. Slots on the page edge "bleed" (run 3mm past the
trim in the print file) — the photograph is drawn to the sheet edge, never
stopped short at a margin by accident.

The grid: 12 columns, the category's gutter (~4mm), the category's
margins; vertical measures snap to the 4pt baseline (catalog.snap).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.agents.brochure.catalog import Rect, TemplateSpec, snap


@dataclass
class Slot:
    name: str
    rect: Rect
    kind: str                     # image | plan | text
    bleed: tuple[bool, bool, bool, bool] = (False, False, False, False)   # top right bottom left
    role: str = ""                # what the slot is for, for the planner's choice


@dataclass
class Layout:
    name: str
    slots: dict[str, Slot] = field(default_factory=dict)
    variant: str = ""             # a CSS hint for the template (e.g. "mirror")

    def images(self) -> list[Slot]:
        return [s for s in self.slots.values() if s.kind == "image"]

    def plans(self) -> list[Slot]:
        return [s for s in self.slots.values() if s.kind == "plan"]

    def text(self) -> list[Slot]:
        return [s for s in self.slots.values() if s.kind == "text"]


class Grid:
    def __init__(self, spec: TemplateSpec):
        self.spec = spec
        self.W, self.H = spec.sheet.w, spec.sheet.h
        self.f = spec.frame()
        self.g = spec.design.gutter
        self.cw = (self.f.w - 11 * self.g) / 12

    def x(self, col: int) -> float:
        """Left edge of column `col` (1-based)."""
        return self.f.x + (col - 1) * (self.cw + self.g)

    def cols(self, a: int, b: int) -> tuple[float, float]:
        x = self.x(a)
        return x, self.x(b) + self.cw - x

    def rect(self, a: int, b: int, y: float, h: float) -> Rect:
        x, w = self.cols(a, b)
        return Rect(x, y, w, h)

    def fy(self, frac: float) -> float:
        """A y position at a fraction of the frame, on the baseline."""
        return snap(self.f.y + self.f.h * frac)

    # -- bleeds: a rect extended to the sheet edge on the named sides -----
    def bleed(self, r: Rect, top=False, right=False, bottom=False, left=False) -> Rect:
        x0 = 0.0 if left else r.x
        y0 = 0.0 if top else r.y
        x1 = self.W if right else r.right
        y1 = self.H if bottom else r.bottom
        return Rect(x0, y0, x1 - x0, y1 - y0)


def _img(name: str, r: Rect, *, top=False, right=False, bottom=False, left=False, role="") -> Slot:
    return Slot(name, r, "image", (top, right, bottom, left), role)


def _txt(name: str, r: Rect, role="") -> Slot:
    return Slot(name, r, "text", role=role)


def _plan(name: str, r: Rect, role="") -> Slot:
    return Slot(name, r, "plan", role=role)


def _layout(name: str, *slots: Slot, variant: str = "") -> Layout:
    return Layout(name, {s.name: s for s in slots}, variant)


PT = 25.4 / 72


def heading_mm(spec: TemplateSpec, title_lines: int = 2, lead_lines: int = 0) -> float:
    """The height a kicker, a title of `title_lines` and a standfirst of
    `lead_lines` need at this category's type sizes — so a heading area is
    never drawn shorter than the heading it holds."""
    t = spec.design.type
    kicker = t.kicker * PT * 1.3 + 3.5
    title = title_lines * t.h1 * t.lh_h1 * PT
    rule = 9.0 if spec.category == "luxury" else (5.0 if spec.category == "mid" else 4.0)
    lead = lead_lines * t.lead * 1.45 * PT + (4.2 if lead_lines else 0)
    return snap(kicker + title + rule + lead + 2)


def _band(name: str, spec: TemplateSpec, share: float, *, bleed: bool, variant: str = "band") -> Layout:
    """A photograph across the top of the page, the copy in a band beneath
    — for wide photographs on a landscape sheet, which a half-page column
    would crop by half."""
    g = Grid(spec)
    h = snap(g.f.h * share)
    # the copy band below keeps room for a two-line title and a standfirst
    need = heading_mm(spec, 2, 2) + 3 * g.g
    if g.f.h - h < need:
        h = snap(g.f.h - need)
    if bleed:
        img = _img("img", Rect(0, 0, g.W, g.f.y + h), top=True, right=True, left=True)
    else:
        img = _img("img", Rect(g.f.x, g.f.y, g.f.w, h))
    ty = g.f.y + h + 3 * g.g
    return _layout(name, img, _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)), variant=variant)


def _column(name: str, spec: TemplateSpec, *, split: int = 6, bleed: bool = True,
            variant: str = "column") -> Layout:
    """A tall photograph down the left of a portrait page, the copy in a
    column beside it — for portrait-shaped pictures, which a full-width band
    would crop to a strip."""
    g = Grid(spec)
    r = g.rect(1, split, g.f.y, g.f.h)
    if bleed:
        r = g.bleed(r, top=True, bottom=True, left=True)
        r = Rect(r.x, r.y, r.w + g.g / 2, r.h)
        img = _img("img", r, top=True, bottom=True, left=True)
    else:
        img = _img("img", r)
    return _layout(name, img, _txt("text", g.rect(split + 1, 12, g.f.y, g.f.h)), variant=variant)


# ---------------------------------------------------------------------------
# cover and back
# ---------------------------------------------------------------------------


def cover(spec: TemplateSpec) -> Layout:
    g = Grid(spec)
    cat = spec.category
    if cat == "luxury":
        full = Rect(0, 0, g.W, g.H)
        # the name block sits low over the photograph, but never in less room
        # than a two-line name, its ornament, a two-line tagline and the
        # residences line need at this type scale (a 16:9 sheet is short)
        t = spec.design.type
        need = snap(t.kicker * PT * 1.3 + 4 + 2 * t.display * 1.02 * PT + 16
                    + 2 * t.h2 * 1.4 * PT + t.kicker * PT * 1.3 + 8)
        top = min(g.fy(0.55 if spec.landscape else 0.62), snap(g.f.bottom - need))
        return _layout("cover", _img("img", full, top=True, right=True, bottom=True, left=True, role="hero"),
                       _txt("text", Rect(g.f.x, top, g.f.w, g.f.bottom - top)))
    if cat == "mid":
        split = 0.66 if spec.landscape else 0.56
        img_r = Rect(g.f.x, g.f.y, g.f.w, snap(g.f.h * split))
        ty = img_r.bottom + g.g * 2.5
        return _layout("cover", _img("img", img_r, role="hero"),
                       _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)))
    # affordable: the photograph bleeds off the top, the facts sit on a white band
    split = 0.63 if spec.landscape else 0.54
    img_r = Rect(0, 0, g.W, snap(g.H * split))
    ty = img_r.bottom + g.g * 2
    return _layout("cover", _img("img", img_r, top=True, right=True, left=True, role="hero"),
                   _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)))


def closing(spec: TemplateSpec, with_image: bool = True, wide: bool = False, tall: bool = False) -> Layout:
    g = Grid(spec)
    if not with_image:
        return _layout("closing", _txt("text", g.f), variant="noimage")
    if tall and spec.portrait:
        return _column("closing", spec, split=5)
    if wide and spec.landscape:
        return _band("closing", spec, 0.5, bleed=True)
    if spec.landscape:
        split = 6 if spec.category != "luxury" else 7
        if spec.sheet.digital:
            split = 5 if spec.category == "luxury" else 6
        img_r = g.bleed(g.rect(1, split, g.f.y, g.f.h), top=True, bottom=True, left=True)
        img_r = Rect(img_r.x, img_r.y, img_r.w + g.g / 2, img_r.h)
        return _layout("closing", _img("img", img_r, top=True, bottom=True, left=True),
                       _txt("text", g.rect(split + 2 if spec.category == "luxury" else split + 1, 12,
                                           g.f.y, g.f.h)))
    img_r = Rect(0, 0, g.W, snap(g.H * 0.42))
    ty = img_r.bottom + g.g * 2
    return _layout("closing", _img("img", img_r, top=True, right=True, left=True),
                   _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)))


# ---------------------------------------------------------------------------
# narrative pages
# ---------------------------------------------------------------------------


def overview(spec: TemplateSpec, wide: bool = False, tall: bool = False) -> Layout:
    """The story of the project beside one photograph, with its figures."""
    g = Grid(spec)
    cat = spec.category
    if tall and spec.portrait:
        return _column("overview", spec, bleed=cat != "affordable")
    if wide and spec.landscape:
        return _band("overview", spec, 0.5, bleed=cat != "affordable")
    if spec.portrait:
        img_r = Rect(0, 0, g.W, snap(g.H * 0.42))
        ty = img_r.bottom + g.g * 2.5
        return _layout("overview", _img("img", img_r, top=True, right=True, left=True),
                       _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)), variant="stack")
    if cat == "luxury":
        # type on the left in generous space; the photograph bleeds right
        img_r = g.bleed(g.rect(7, 12, g.f.y, g.f.h), top=True, right=True, bottom=True)
        return _layout("overview", _img("img", img_r, top=True, right=True, bottom=True),
                       _txt("text", g.rect(1, 5, g.f.y, g.f.h)), variant="mirror")
    if cat == "mid":
        img_r = g.bleed(g.rect(1, 6, g.f.y, g.f.h), top=True, bottom=True, left=True)
        return _layout("overview", _img("img", img_r, top=True, bottom=True, left=True),
                       _txt("text", g.rect(8, 12, g.f.y, g.f.h)))
    return _layout("overview", _img("img", g.rect(1, 6, g.f.y, g.f.h)),
                   _txt("text", g.rect(7, 12, g.f.y, g.f.h)))


def highlights(spec: TemplateSpec, with_image: bool = True, wide: bool = False) -> Layout:
    """The reasons to buy, one card or line each."""
    g = Grid(spec)
    cat = spec.category
    if wide and with_image and spec.landscape:
        h = snap(g.f.h * 0.36)
        ty = g.f.y + h + 3 * g.g
        return _layout("highlights", _img("img", Rect(g.f.x, g.f.y, g.f.w, h)),
                       _txt("head", g.rect(1, 4, ty, g.f.bottom - ty)),
                       _txt("text", g.rect(5, 12, ty, g.f.bottom - ty)), variant="band")
    if spec.portrait:
        if not with_image:
            return _layout("highlights", _txt("text", g.f), variant="stack")
        img_h = snap(g.f.h * 0.3)
        head_h = snap(g.f.h * 0.17)
        return _layout("highlights",
                       _txt("head", Rect(g.f.x, g.f.y, g.f.w, head_h)),
                       _img("img", Rect(g.f.x, g.f.y + head_h + g.g, g.f.w, img_h)),
                       _txt("text", Rect(g.f.x, g.f.y + head_h + img_h + 3 * g.g, g.f.w,
                                         g.f.h - head_h - img_h - 3 * g.g)), variant="stack")
    if cat == "affordable":
        head_h = snap(g.f.h * 0.36)
        slots = [_txt("head", g.rect(1, 4, g.f.y, head_h)),
                 _txt("text", g.rect(5, 12, g.f.y, g.f.h))]
        if with_image:
            slots.append(_img("img", g.rect(1, 4, g.f.y + head_h + g.g, g.f.h - head_h - g.g)))
        else:
            slots[0] = _txt("head", g.rect(1, 4, g.f.y, g.f.h))
        return _layout("highlights", *slots)
    if cat == "mid":
        if not with_image:
            return _layout("highlights", _txt("head", g.rect(1, 4, g.f.y, g.f.h)),
                           _txt("text", g.rect(6, 12, g.f.y, g.f.h)))
        img_r = g.bleed(g.rect(1, 5, g.f.y, g.f.h), top=True, bottom=True, left=True)
        return _layout("highlights", _img("img", img_r, top=True, bottom=True, left=True),
                       _txt("text", g.rect(7, 12, g.f.y, g.f.h)))
    # luxury: an editorial list beside a framed photograph
    if not with_image:
        return _layout("highlights", _txt("head", g.rect(1, 4, g.f.y, g.f.h)),
                       _txt("text", g.rect(6, 12, g.f.y, g.f.h)))
    return _layout("highlights", _img("img", g.rect(1, 5, g.f.y, g.f.h)),
                   _txt("text", g.rect(7, 12, g.f.y, g.f.h)))


def figures(spec: TemplateSpec, wide: bool = False, tall: bool = False) -> Layout:
    """Sustainability or performance figures set large, beside a photograph."""
    g = Grid(spec)
    if tall and spec.portrait:
        return _column("figures", spec, bleed=spec.category != "affordable")
    if wide and spec.landscape:
        return _band("figures", spec, 0.42, bleed=spec.category != "affordable")
    if spec.portrait:
        img_r = Rect(0, 0, g.W, snap(g.H * 0.36))
        ty = img_r.bottom + g.g * 2.5
        return _layout("figures", _img("img", img_r, top=True, right=True, left=True),
                       _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)), variant="stack")
    img_r = g.bleed(g.rect(8, 12, g.f.y, g.f.h), top=True, right=True, bottom=True)
    return _layout("figures", _img("img", img_r, top=True, right=True, bottom=True),
                   _txt("text", g.rect(1, 7, g.f.y, g.f.h)))


# ---------------------------------------------------------------------------
# photograph-led pages
# ---------------------------------------------------------------------------


def feature(spec: TemplateSpec, n: int, mirror: bool = False, wide: bool = False,
            tall: bool = False) -> Layout:
    """One to three photographs of one subject and the copy that sells it."""
    g = Grid(spec)
    cat = spec.category
    if tall and n == 1 and spec.portrait:
        return _column("feature", spec, split=7, bleed=cat != "affordable")
    if wide and n == 1 and spec.landscape:
        return _band("feature", spec, 0.68, bleed=cat != "affordable")
    if spec.portrait:
        if n == 1:
            img_h = snap(g.f.h * 0.6)
            return _layout("feature", _img("img", Rect(0, 0, g.W, g.f.y + img_h), top=True, right=True, left=True),
                           _txt("text", Rect(g.f.x, g.f.y + img_h + 3 * g.g, g.f.w,
                                             g.f.h - img_h - 3 * g.g)), variant="stack")
        if n == 2:
            h1 = snap(g.f.h * 0.5)
            h2 = snap(g.f.h - h1 - g.g)
            return _layout("feature",
                           _img("img", Rect(g.f.x, g.f.y, g.f.w, h1)),
                           _img("img2", g.rect(1, 7, g.f.y + h1 + g.g, h2)),
                           _txt("text", g.rect(8, 12, g.f.y + h1 + g.g, g.f.h - h1 - g.g)),
                           variant="stack")
        h1 = snap(g.f.h * 0.38)
        h2 = snap(g.f.h * 0.24)
        return _layout("feature",
                       _img("img", Rect(g.f.x, g.f.y, g.f.w, h1)),
                       _img("img2", g.rect(1, 6, g.f.y + h1 + g.g, h2)),
                       _img("img3", g.rect(7, 12, g.f.y + h1 + g.g, h2)),
                       _txt("text", Rect(g.f.x, g.f.y + h1 + h2 + 3 * g.g, g.f.w,
                                         g.f.h - h1 - h2 - 3 * g.g)), variant="stack")
    if n == 1:
        text_cols = (1, 4) if mirror else (9, 12)
        img_cols = (5, 12) if mirror else (1, 8)
        if cat == "luxury":
            text_cols = (1, 4) if mirror else (10, 12)
            img_cols = (6, 12) if mirror else (1, 8)
        img_r = g.rect(*img_cols, g.f.y, g.f.h)
        if cat != "affordable":
            img_r = g.bleed(img_r, top=True, bottom=True, right=mirror, left=not mirror)
        return _layout("feature", _img("img", img_r, top=cat != "affordable", bottom=cat != "affordable",
                                       right=mirror and cat != "affordable", left=not mirror and cat != "affordable"),
                       _txt("text", g.rect(*text_cols, g.f.y, g.f.h)), variant="mirror" if mirror else "")
    if n == 2:
        img_h = snap(g.f.h * (0.56 if spec.sheet.digital else 0.64))
        img_h = min(img_h, snap(g.f.h - heading_mm(spec, 2, 1) - 3 * g.g))
        ty = g.f.y + img_h + 3 * g.g
        return _layout("feature",
                       _img("img", g.rect(1, 7, g.f.y, img_h)),
                       _img("img2", g.rect(8, 12, g.f.y, img_h)),
                       _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)), variant="band")
    # three: one large, two stacked, copy beneath the stack
    small_h = snap((g.f.h * 0.62 - g.g) / 2)
    return _layout("feature",
                   _img("img", g.rect(1, 7, g.f.y, g.f.h)),
                   _img("img2", g.rect(8, 12, g.f.y, small_h)),
                   _img("img3", g.rect(8, 12, g.f.y + small_h + g.g, small_h)),
                   _txt("text", g.rect(8, 12, g.f.y + 2 * small_h + 3 * g.g, g.f.h - 2 * small_h - 3 * g.g)),
                   variant="trio")


def gallery(spec: TemplateSpec, n: int, wide: bool = False) -> Layout:
    """Interiors: a mosaic of rooms beside a short column of copy."""
    g = Grid(spec)
    n = max(2, min(4, n))
    if wide:
        # wide renders: rows of the frame's full width, the copy above
        head_h = max(snap(g.f.h * (0.2 if spec.landscape else 0.16)), heading_mm(spec, 2, 0))
        y = g.f.y + head_h + 2 * g.g
        h = g.f.bottom - y
        half = (g.f.w - g.g) / 2
        slots = [_txt("text", Rect(g.f.x, g.f.y, g.f.w, head_h))]
        if n == 2:
            if spec.landscape:
                slots += [_img("img", Rect(g.f.x, y, half, h)), _img("img2", Rect(g.f.x + half + g.g, y, half, h))]
            else:
                hh = (h - g.g) / 2
                slots += [_img("img", Rect(g.f.x, y, g.f.w, hh)), _img("img2", Rect(g.f.x, y + hh + g.g, g.f.w, hh))]
        elif n == 3 or not spec.landscape:
            hh = snap(h * 0.56)
            slots += [_img("img", Rect(g.f.x, y, g.f.w, hh)),
                      _img("img2", Rect(g.f.x, y + hh + g.g, half, h - hh - g.g)),
                      _img("img3", Rect(g.f.x + half + g.g, y + hh + g.g, half, h - hh - g.g))]
        else:
            hh = (h - g.g) / 2
            slots += [_img("img", Rect(g.f.x, y, half, hh)), _img("img2", Rect(g.f.x + half + g.g, y, half, hh)),
                      _img("img3", Rect(g.f.x, y + hh + g.g, half, hh)),
                      _img("img4", Rect(g.f.x + half + g.g, y + hh + g.g, half, hh))]
        return _layout("gallery", *slots, variant="rows")
    if spec.portrait:
        head_h = max(snap(g.f.h * 0.2), heading_mm(spec, 2, 1))
        y = g.f.y + head_h + g.g
        h = g.f.bottom - y
        slots = [_txt("text", Rect(g.f.x, g.f.y, g.f.w, head_h))]
        if n == 2:
            hh = (h - g.g) / 2
            slots += [_img("img", Rect(g.f.x, y, g.f.w, hh)), _img("img2", Rect(g.f.x, y + hh + g.g, g.f.w, hh))]
        elif n == 3:
            hh = snap(h * 0.55)
            slots += [_img("img", Rect(g.f.x, y, g.f.w, hh)),
                      _img("img2", g.rect(1, 6, y + hh + g.g, h - hh - g.g)),
                      _img("img3", g.rect(7, 12, y + hh + g.g, h - hh - g.g))]
        else:
            hh = (h - g.g) / 2
            slots += [_img("img", g.rect(1, 6, y, hh)), _img("img2", g.rect(7, 12, y, hh)),
                      _img("img3", g.rect(1, 6, y + hh + g.g, hh)), _img("img4", g.rect(7, 12, y + hh + g.g, hh))]
        return _layout("gallery", *slots, variant="stack")
    text = _txt("text", g.rect(1, 3, g.f.y, g.f.h))
    y, h = g.f.y, g.f.h
    if n == 2:
        hh = (h - g.g) / 2
        return _layout("gallery", text, _img("img", g.rect(4, 12, y, hh)),
                       _img("img2", g.rect(4, 12, y + hh + g.g, hh)))
    if n == 3:
        hh = (h - g.g) / 2
        return _layout("gallery", text, _img("img", g.rect(4, 12, y, hh)),
                       _img("img2", g.rect(4, 8, y + hh + g.g, hh)),
                       _img("img3", g.rect(9, 12, y + hh + g.g, hh)))
    hh = (h - g.g) / 2
    return _layout("gallery", text,
                   _img("img", g.rect(4, 8, y, hh)), _img("img2", g.rect(9, 12, y, hh)),
                   _img("img3", g.rect(4, 7, y + hh + g.g, hh)), _img("img4", g.rect(8, 12, y + hh + g.g, hh)))


def amenities(spec: TemplateSpec, n: int, wide: bool = False) -> Layout:
    """Amenities by zone, under a strip of the photographs that show them."""
    g = Grid(spec)
    n = max(0, min(3, n))
    if n == 0:
        return _layout("amenities", _txt("head", g.rect(1, 4, g.f.y, g.f.h)),
                       _txt("text", g.rect(5, 12, g.f.y, g.f.h)), variant="noimage")
    strip = snap(g.f.h * (0.46 if spec.landscape else 0.34) * (0.74 if wide else 1.0)
                 * (0.82 if spec.sheet.digital else 1.0))
    slots = []
    spans = {1: [(1, 12)], 2: [(1, 7), (8, 12)], 3: [(1, 5), (6, 9), (10, 12)]}[n]
    if spec.portrait and n == 3:
        spans = [(1, 6), (7, 12)]
        n = 2
    for i, (a, b) in enumerate(spans):
        slots.append(_img("img" if i == 0 else f"img{i + 1}", g.rect(a, b, g.f.y, strip)))
    ty = g.f.y + strip + 3 * g.g
    if spec.portrait:
        slots.append(_txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)))
        return _layout("amenities", *slots, variant="stack")
    slots.append(_txt("head", g.rect(1, 3, ty, g.f.bottom - ty)))
    slots.append(_txt("text", g.rect(4, 12, ty, g.f.bottom - ty)))
    return _layout("amenities", *slots)


# ---------------------------------------------------------------------------
# drawings
# ---------------------------------------------------------------------------


def masterplan(spec: TemplateSpec) -> Layout:
    """The site plan as large as the page allows, whole, with a narrow
    column of what a reader needs to read it."""
    g = Grid(spec)
    if spec.portrait:
        ph = snap(g.f.h * 0.7)
        return _layout("masterplan", _plan("plan", Rect(g.f.x, g.f.y, g.f.w, ph)),
                       _txt("text", Rect(g.f.x, g.f.y + ph + 3 * g.g, g.f.w, g.f.h - ph - 3 * g.g)),
                       variant="stack")
    return _layout("masterplan", _plan("plan", g.rect(1, 9, g.f.y, g.f.h)),
                   _txt("text", g.rect(10, 12, g.f.y, g.f.h)))


def plans(spec: TemplateSpec, n: int) -> Layout:
    """Unit plans or typical floor plates, each on its own card with its
    name beneath."""
    g = Grid(spec)
    n = max(1, min(4, n))
    head_h = max(snap(g.f.h * (0.16 if spec.landscape else 0.11)), heading_mm(spec, 1, 0) + 6)
    y = g.f.y + head_h + g.g
    h = g.f.bottom - y
    slots = [_txt("head", Rect(g.f.x, g.f.y, g.f.w, head_h))]
    if n == 1:
        if spec.landscape:
            return _layout("plans", _plan("plan", g.rect(1, 8, g.f.y, g.f.h)),
                           _txt("text", g.rect(9, 12, g.f.y, g.f.h)), variant="single")
        slots.append(_plan("plan", Rect(g.f.x, y, g.f.w, h)))
    elif n == 2:
        if spec.landscape:
            slots += [_plan("plan", g.rect(1, 6, y, h)), _plan("plan2", g.rect(7, 12, y, h))]
        else:
            hh = (h - g.g) / 2
            slots += [_plan("plan", Rect(g.f.x, y, g.f.w, hh)), _plan("plan2", Rect(g.f.x, y + hh + g.g, g.f.w, hh))]
    elif n == 3 and spec.landscape:
        slots += [_plan("plan", g.rect(1, 4, y, h)), _plan("plan2", g.rect(5, 8, y, h)),
                  _plan("plan3", g.rect(9, 12, y, h))]
    else:
        hh = (h - g.g) / 2
        names = ["plan", "plan2", "plan3", "plan4"][:n]
        cells = [g.rect(1, 6, y, hh), g.rect(7, 12, y, hh), g.rect(1, 6, y + hh + g.g, hh),
                 g.rect(7, 12, y + hh + g.g, hh)]
        slots += [_plan(nm, c) for nm, c in zip(names, cells)]
    return _layout("plans", *slots)


def residences(spec: TemplateSpec, with_image: bool = True, wide: bool = False, plans: int = 0,
               tall: bool = False) -> Layout:
    """The unit schedule beside a photograph of the interiors — or, where the
    project has unit plans and no interiors, beside the plans themselves."""
    g = Grid(spec)
    if plans:
        n = min(2, plans)
        if spec.landscape:
            y, h = g.f.y, g.f.h
            hh = (h - g.g) / 2 if n == 2 else h
            slots = [_txt("text", g.rect(1, 6, g.f.y, g.f.h))]
            slots += [_plan("plan" if i == 0 else "plan2", g.rect(7, 12, y + i * (hh + g.g), hh)) for i in range(n)]
            return _layout("residences", *slots, variant="plans")
        th = snap(g.f.h * 0.5)
        y = g.f.y + th + 2 * g.g
        h = g.f.bottom - y
        slots = [_txt("text", Rect(g.f.x, g.f.y, g.f.w, th))]
        if n == 2:
            slots += [_plan("plan", g.rect(1, 6, y, h)), _plan("plan2", g.rect(7, 12, y, h))]
        else:
            slots.append(_plan("plan", Rect(g.f.x, y, g.f.w, h)))
        return _layout("residences", *slots, variant="plans")
    if wide and with_image and spec.landscape:
        return _band("residences", spec, 0.36, bleed=spec.category != "affordable")
    if tall and with_image and spec.portrait and not plans:
        return _column("residences", spec, split=5, bleed=spec.category != "affordable")
    if spec.portrait:
        if not with_image:
            return _layout("residences", _txt("text", g.f), variant="stack")
        img_r = Rect(0, 0, g.W, snap(g.H * 0.34))
        ty = img_r.bottom + g.g * 2.5
        return _layout("residences", _img("img", img_r, top=True, right=True, left=True),
                       _txt("text", Rect(g.f.x, ty, g.f.w, g.f.bottom - ty)), variant="stack")
    if not with_image:
        return _layout("residences", _txt("head", g.rect(1, 4, g.f.y, g.f.h)),
                       _txt("text", g.rect(5, 12, g.f.y, g.f.h)), variant="noimage")
    split = 7 if spec.category != "luxury" else 6
    img_r = g.bleed(g.rect(split + 1, 12, g.f.y, g.f.h), top=True, right=True, bottom=True)
    if spec.category == "affordable":
        img_r = g.rect(split + 1, 12, g.f.y, g.f.h)
        return _layout("residences", _img("img", img_r), _txt("text", g.rect(1, split, g.f.y, g.f.h)))
    return _layout("residences", _img("img", img_r, top=True, right=True, bottom=True),
                   _txt("text", g.rect(1, split, g.f.y, g.f.h)))


def location(spec: TemplateSpec, has_map: bool, with_image: bool = True, wide: bool = False,
             tall: bool = False) -> Layout:
    g = Grid(spec)
    if tall and with_image and not has_map and spec.portrait:
        return _column("location", spec, split=5, bleed=spec.category != "affordable")
    if not has_map and not with_image:
        if spec.landscape:
            return _layout("location", _txt("head", g.rect(1, 4, g.f.y, g.f.h)),
                           _txt("text", g.rect(6, 12, g.f.y, g.f.h)), variant="noimage")
        hh = snap(g.f.h * 0.3)
        return _layout("location", _txt("head", Rect(g.f.x, g.f.y, g.f.w, hh)),
                       _txt("text", Rect(g.f.x, g.f.y + hh + g.g, g.f.w, g.f.h - hh - g.g)),
                       variant="noimage")
    if wide and not has_map and spec.landscape:
        return _band("location", spec, 0.42, bleed=spec.category != "affordable")
    if spec.portrait:
        mh = snap(g.f.h * 0.48)
        first = _plan("plan", Rect(g.f.x, g.f.y, g.f.w, mh)) if has_map else \
            _img("img", Rect(0, 0, g.W, g.f.y + mh), top=True, right=True, left=True)
        return _layout("location", first,
                       _txt("text", Rect(g.f.x, g.f.y + mh + 3 * g.g, g.f.w, g.f.h - mh - 3 * g.g)),
                       variant="stack")
    if has_map:
        return _layout("location", _plan("plan", g.rect(1, 7, g.f.y, g.f.h)),
                       _txt("text", g.rect(8, 12, g.f.y, g.f.h)))
    img_r = g.bleed(g.rect(1, 6, g.f.y, g.f.h), top=True, bottom=True, left=True)
    if spec.category == "affordable":
        return _layout("location", _img("img", g.rect(1, 6, g.f.y, g.f.h)),
                       _txt("text", g.rect(7, 12, g.f.y, g.f.h)))
    return _layout("location", _img("img", img_r, top=True, bottom=True, left=True),
                   _txt("text", g.rect(7, 12, g.f.y, g.f.h)))


BUILDERS: dict[str, Callable[..., Layout]] = {
    "cover": cover, "closing": closing, "overview": overview, "highlights": highlights,
    "figures": figures, "feature": feature, "gallery": gallery, "amenities": amenities,
    "masterplan": masterplan, "plans": plans, "residences": residences, "location": location,
}


def build(kind: str, spec: TemplateSpec, **options) -> Layout:
    return BUILDERS[kind](spec, **options)


def capacity_chars(rect: Rect, size_pt: float, line_height: float, char_em: float,
                   share: float = 1.0) -> int:
    """How many characters of running text a box holds at a type size."""
    pt_mm = 25.4 / 72
    line_mm = size_pt * pt_mm * line_height
    per_line = rect.w / (size_pt * pt_mm * char_em)
    lines = max(0, int(rect.h * share / line_mm))
    return int(per_line * lines * 0.92)        # ragged right loses ~8%


def focal(topic: str, kind: str, slot: Rect, aspect: Optional[float]) -> str:
    """Where a photograph is anchored when it is cropped to its slot.

    Architecture keeps its top — a tower's crown, a roofline — when a
    landscape render is cut down to a taller slot it loses its sides
    evenly, and when a tall render is cut to a wide slot it loses its
    foreground, never the sky line. Interiors and gardens keep the centre."""
    if not aspect:
        return "50% 50%"
    image_taller = aspect < slot.aspect           # cropping top/bottom
    if image_taller and topic in ("exterior", "night", "arrival", "clubhouse", "aerial"):
        return "50% 18%"
    if image_taller:
        return "50% 42%"
    return "50% 50%"


# The most of a photograph a slot may cut away. Beyond it the slot is
# reshaped to the picture (planner._cap_crop) rather than the picture cut.
CROP_CAP = 0.35


def crop_loss(image_aspect: float, slot_aspect: float) -> float:
    """The share of the photograph a cover-fit crop removes."""
    if image_aspect <= 0 or slot_aspect <= 0:
        return 1.0
    return 1.0 - min(image_aspect, slot_aspect) / max(image_aspect, slot_aspect)
