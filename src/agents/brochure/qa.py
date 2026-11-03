"""Measure the rendered brochure, decide what is wrong, and repair it.

The audit (src/rendering/brochure_audit.js) reports what a reader would
see. This module turns it into

    DEFECTS     reasons the brochure must not go out as it is: copy running
                past its box or off the page, a unit name, price or area
                broken across lines, an image that did not load, a font that
                did not load (the rupee sign would fall back), type under
                6pt, the same title or sentence on two pages
    ADVISORIES  things a reviewer should know: a text box its copy leaves
                thin, a page outside its category's white-space range, a
                photograph printed below 150 dpi
    REPAIRS     the smallest change that addresses a page's worst problem,
                by whole sentences and optional elements — never by type
                size, which is fixed by the design system:
                    overflow   one sentence fewer; then the optional extras
                               (a lead, a point, a figure) in order of what a
                               reader misses least
                    thin       one more sentence from the page's pool
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.agents.brochure import strategy
from src.agents.brochure.book import Book, BookPage
from src.agents.brochure.catalog import DESIGNS

MIN_PT = 6.0
OVERFLOW_TOLERANCE_MM = 0.6
FILL_FLOOR = {"affordable": 0.72, "mid": 0.62, "luxury": 0.5}
MIN_DPI = 150

BLOCKING = {
    "overflow": "copy runs past its box",
    "outside": "type runs off the page",
    "over_image": "type is set over a photograph",
    "broken": "a unit, price or figure breaks across lines",
    "image_missing": "an image did not load",
    "font_missing": "a typeface did not load",
    "small_text": "type is below 6pt",
    "repeated_text": "the same text appears on two pages",
}


@dataclass
class PageReport:
    number: int
    kind: str
    whitespace: float
    overflow_mm: float = 0.0
    thin_boxes: list[str] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    fill: dict = field(default_factory=dict)


@dataclass
class Report:
    pages: list[PageReport]
    defects: list[str]
    advisories: list[str]

    @property
    def passed(self) -> bool:
        return not self.defects

    def mean_whitespace(self) -> float:
        body = [p.whitespace for p in self.pages if p.kind not in ("cover", "closing")]
        return sum(body) / len(body) if body else 0.0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def evaluate(audit: dict, book: Optional[Book] = None) -> Report:
    category = book.category if book else "mid"
    design = DESIGNS.get(category)
    ws_lo, ws_hi = design.whitespace if design else (0.2, 0.4)
    floor = FILL_FLOOR.get(category, 0.6)
    pages: list[PageReport] = []
    defects: list[str] = []
    advisories: list[str] = []
    by_number = {p.number: p for p in (book.pages if book else [])}

    missing_fonts = [f for f, ok in (audit.get("fonts") or {}).items() if not ok]
    if missing_fonts:
        msg = f"{BLOCKING['font_missing']}: {', '.join(missing_fonts)}"
        defects.append(msg)

    for info in audit.get("pages", []):
        number = int(info.get("page", 0))
        rep = PageReport(number=number, kind=info.get("kind", ""), whitespace=float(info.get("whitespace") or 0))

        def flag(kind: str, detail: str = "") -> None:
            message = f"Page {number}: {BLOCKING[kind]}" + (f" ({detail})" if detail else "")
            rep.defects.append(message)
            defects.append(message)

        for box in info.get("boxes", []):
            over = max(float(box.get("overflow_mm") or 0), float(box.get("overflow_x_mm") or 0))
            rep.fill[box.get("name")] = float(box.get("fill") or 0)
            if over > OVERFLOW_TOLERANCE_MM:
                rep.overflow_mm = max(rep.overflow_mm, over)
                flag("overflow", f"{box.get('name')} box, {over:.1f}mm")
            elif rep.kind not in ("cover", "closing", "plans", "masterplan") and \
                    float(box.get("fill") or 0) < floor and box.get("name") == "text":
                rep.thin_boxes.append(box.get("name"))
        for text in info.get("broken", [])[:3]:
            flag("broken", text)
        if info.get("outside"):
            flag("outside", info["outside"][0])
        if info.get("over_image"):
            flag("over_image", info["over_image"][0])
        for text in info.get("small_text", [])[:2]:
            flag("small_text", text)
        for img in info.get("images", []):
            if not img.get("loaded"):
                flag("image_missing", img.get("id", ""))
                continue
            if img.get("kind") == "image" and img.get("w_mm") and img.get("nat_w"):
                # object-fit: cover prints at the larger of the two scales
                scale = max(img["w_mm"] / img["nat_w"], img["h_mm"] / max(1, img["nat_h"]))
                dpi = 25.4 / scale if scale else 0
                if dpi and dpi < MIN_DPI:
                    rep.advisories.append(f"Page {number}: a photograph prints at about {dpi:.0f} dpi")
        page = by_number.get(number)
        if rep.kind not in ("cover", "closing") and not (ws_lo - 0.12 <= rep.whitespace <= ws_hi + 0.15):
            rep.advisories.append(f"Page {number}: {rep.whitespace:.0%} open space "
                                  f"(this category aims for {ws_lo:.0%}–{ws_hi:.0%})")
        if rep.thin_boxes and page is not None and page.take >= len(page.pool):
            rep.advisories.append(f"Page {number}: its copy leaves the text area thin")
        pages.append(rep)
        advisories.extend(rep.advisories)

    first: dict[str, int] = {}
    for item in audit.get("texts", []):
        key = _norm(item.get("text", ""))
        if len(key) < 14:
            continue
        page_no = int(item.get("page", 0))
        if key in first and first[key] != page_no:
            message = f"Page {page_no}: {BLOCKING['repeated_text']} (also on page {first[key]})"
            defects.append(message)
            for rep in pages:
                if rep.number == page_no:
                    rep.defects.append(message)
        first.setdefault(key, page_no)
    return Report(pages=pages, defects=defects, advisories=advisories)


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------

MIN_TAKE = {"overview": 1, "feature": 1, "gallery": 0, "location": 0}


def _cut_sentence(page: BookPage) -> bool:
    if page.take > MIN_TAKE.get(page.kind, 0) and page.words.body:
        page.take -= 1
        page.set_body()
        return True
    return False


def _cut_point(page: BookPage) -> bool:
    if page.kind == "overview" and page.words.cards:
        page.words.cards.pop()
        return True
    return False


def _cut_lead(page: BookPage) -> bool:
    if page.words.lead and page.kind not in ("cover", "closing", "highlights", "residences"):
        page.words.lead = ""
        return True
    return False


def _cut_figure(page: BookPage) -> bool:
    for key in ("figures", "summary"):
        figures = page.data.get(key)
        if isinstance(figures, list) and len(figures) > 2:
            figures.pop()
            return True
    return False


def _cut_kicker(page: BookPage) -> bool:
    if page.words.kicker and page.kind not in ("cover",):
        page.words.kicker = ""
        return True
    return False


def _cut_place(page: BookPage) -> bool:
    if page.kind == "location" and len(page.data.get("places", [])) > 4:
        page.data["places"].pop()
        return True
    return False


def _cut_amenity(page: BookPage) -> bool:
    if page.kind != "amenities":
        return False
    zones = page.data.get("zones") or []
    longest = max(zones, key=lambda z: len(z.get("items", [])), default=None)
    if longest and len(longest.get("items", [])) > 3:
        longest["items"].pop()
        return True
    return False


def _cut_card_text(page: BookPage) -> bool:
    """Where a card carries its key figure, the figure and the headline say
    enough: the details of those cards go, together."""
    cards = [c for c in page.words.cards if c.text and c.value]
    if page.kind in ("highlights", "figures") and cards:
        for c in cards:
            c.text = ""
        return True
    return False


def _cut_last_card_text(page: BookPage) -> bool:
    cards = [c for c in page.words.cards if c.text]
    if page.kind in ("highlights", "figures", "location") and cards:
        max(cards, key=lambda c: len(c.text)).text = ""
        return True
    return False


def _cut_note(page: BookPage) -> bool:
    if page.words.note and page.kind not in ("residences", "masterplan"):
        page.words.note = ""
        return True
    return False


def _cut_dev(page: BookPage) -> bool:
    if page.kind == "closing" and page.data.get("developer"):
        page.data["developer"] = ""
        return True
    return False


# In order of what a reader misses least.
SHORTEN = (("sentence", _cut_sentence), ("point", _cut_point), ("lead", _cut_lead),
           ("figure", _cut_figure), ("place", _cut_place), ("amenity", _cut_amenity),
           ("card_text", _cut_card_text), ("card_text_any", _cut_last_card_text),
           ("dev", _cut_dev), ("note", _cut_note),
           ("kicker", _cut_kicker))


def repair(book: Book, report: Report, blocked: Optional[dict] = None) -> dict[int, str]:
    """One change per page. Returns {page number: what was done}.

    `blocked` holds, per page, the repairs already tried that did not reduce
    the overflow (they are reverted by the caller and not tried again)."""
    blocked = blocked if blocked is not None else {}
    by_number = {p.number: p for p in book.pages}
    changed: dict[int, str] = {}
    for rep in report.pages:
        page = by_number.get(rep.number)
        if page is None:
            continue
        tried = blocked.setdefault(rep.number, set())
        if rep.overflow_mm > OVERFLOW_TOLERANCE_MM:
            for name, action in SHORTEN:
                if name in tried:
                    continue
                if action(page):
                    changed[rep.number] = name
                    break
            continue
        if rep.thin_boxes and page.take < len(page.pool) and "sentence" not in tried \
                and "grow" not in tried:
            page.take += 1
            page.set_body()
            changed[rep.number] = "grow"
        elif (rep.thin_boxes or (page.variant in ("stack", "band") and rep.fill.get("text", 1.0) < 0.8))                 and "expand" not in tried and page.take >= len(page.pool)                 and expand_pictures(page, rep.fill):
            changed[rep.number] = "expand"
    return changed


def expand_pictures(page: BookPage, fills: dict) -> bool:
    """The copy does not need its whole box and has nothing more to say: the
    photographs directly above it grow down into the space, within the crop
    a picture may bear (layouts.CROP_CAP), and EVERY box beneath them moves
    down with them and shrinks to its copy — a heading box beside the text
    box included. Nothing is moved if any of those boxes cannot give up the
    space, or if a grown picture would reach another picture. The type is
    untouched."""
    from src.agents.brochure.layouts import CROP_CAP

    box = page.boxes.get("text")
    if box is None or fills.get("text", 1.0) >= 0.9:
        return False
    above = [p for p in page.images if p.kind == "image" and p.y + p.h <= box.y + 0.5
             and p.x < box.x + box.w and p.x + p.w > box.x]
    if not above:
        return False
    lowest = max(p.y + p.h for p in above)
    row = [p for p in page.images if p.kind == "image" and abs(p.y + p.h - lowest) < 0.6]
    left, right = min(p.x for p in row), max(p.x + p.w for p in row)
    # every box under the row moves with it; each must have the room to give
    below = [(name, b) for name, b in page.boxes.items()
             if b.y >= lowest - 0.5 and b.x < right and b.x + b.w > left]
    spare = min(b.h * (1 - fills.get(name, 1.0)) - 8.0 for name, b in below) if below else 0.0
    if spare < 6:
        return False
    for p in row:                      # how far each picture can grow before the crop is too deep
        if not p.aspect:
            return False
        for step in range(int(spare), 0, -1):
            slot_aspect = p.w / (p.h + step)
            crop = 1 - min(p.aspect, slot_aspect) / max(p.aspect, slot_aspect)
            if crop <= max(CROP_CAP, p.crop):
                spare = min(spare, step)
                break
        else:
            return False
    if spare < 6:
        return False
    # a grown picture must not reach any other picture
    for p in row:
        for other in page.images:
            if other is p or other in row:
                continue
            if other.x < p.x + p.w and other.x + other.w > p.x and other.y < lowest + spare + 2                     and other.y + other.h > p.y:
                return False
    for p in row:
        p.h += spare
        slot_aspect = p.w / p.h
        p.crop = round(1 - min(p.aspect, slot_aspect) / max(p.aspect, slot_aspect), 3)
    for _name, b in below:
        b.y += spare
        b.h -= spare
    return True


def check_pdf(pdf_path, book: Book, report: Report) -> None:
    """The printed file itself: every page's title is in the PDF's text.
    Chromium's print layout can differ from its screen layout; this is the
    check that what was measured is what was printed."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:  # noqa: BLE001
        return
    by_number = {p.number: p for p in report.pages}
    for page in book.pages:
        title = _norm(page.words.title)
        if not title or page.number > len(doc):
            continue
        text = _norm(doc[page.number - 1].get_text("text", flags=0))   # ligatures expanded
        if title.replace(" ", "") not in text.replace(" ", ""):
            message = f"Page {page.number}: the title is missing from the printed PDF"
            report.defects.append(message)
            if page.number in by_number:
                by_number[page.number].defects.append(message)
    doc.close()


def unresolved(report: Report) -> list[PageReport]:
    return [p for p in report.pages if p.overflow_mm > OVERFLOW_TOLERANCE_MM]


def positioning_for(book: Book) -> strategy.Positioning:
    return strategy.for_category(book.category)
