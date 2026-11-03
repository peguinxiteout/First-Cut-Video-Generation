"""The brochure as planned, page by page.

A Book is made in three passes:

    1. PLAN     (planner.py)     which pages the project's material supports,
                                 in the category's order; which photograph,
                                 plan or map goes in which slot of which
                                 layout (layouts.py)
    2. WRITE    (copywriter.py)  every page's words, from the content model
                                 (facts.py), to the room its text slots have
    3. MEASURE  (qa.py)          rendered in Chromium; copy that overflows a
                                 slot is shortened by whole sentences, copy
                                 that leaves a slot thin is given the next
                                 sentence from its page's pool. Type sizes
                                 never change.

The templates draw exactly what is here: every slot has a position in
millimetres on its sheet.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Placed(BaseModel):
    """An image in a slot."""
    slot: str
    image_id: str
    x: float
    y: float
    w: float
    h: float
    kind: str = "image"           # image (cropped to fill) | plan (shown whole)
    position: str = "50% 50%"     # object-position for a cropped photograph
    bleed: tuple[bool, bool, bool, bool] = (False, False, False, False)
    label: str = ""               # a plan's name, printed beneath it
    sublabel: str = ""            # its area, when the schedule gives one
    dpi: float = 0.0              # effective print resolution at this size
    crop: float = 0.0             # share of the picture the crop removes
    aspect: float = 0.0           # the picture's own width / height


class Box(BaseModel):
    slot: str
    x: float
    y: float
    w: float
    h: float


class Card(BaseModel):
    title: str = ""
    text: str = ""
    icon: str = ""
    value: str = ""               # a statistic set large
    topic: str = ""


class Copy(BaseModel):
    kicker: str = ""
    title: str = ""
    lead: str = ""
    body: list[str] = Field(default_factory=list)      # paragraphs
    cards: list[Card] = Field(default_factory=list)
    note: str = ""                # a footnote (e.g. "Prices are indicative")
    cta: str = ""
    source: str = ""              # model | offline


class BookPage(BaseModel):
    number: int = 0
    kind: str                     # cover overview highlights figures feature gallery amenities
                                  # masterplan plans residences location closing
    layout: str = ""
    variant: str = ""
    topic: str = ""               # a feature page's subject family
    images: list[Placed] = Field(default_factory=list)
    boxes: dict[str, Box] = Field(default_factory=dict)
    words: Copy = Field(default_factory=Copy)
    # the structured content the page sets (zones, units, places, figures,
    # stats, claims ...) — from facts.py, never typed by the copywriter
    data: dict[str, Any] = Field(default_factory=dict)
    # running copy: candidate sentences in order, and how many are set
    pool: list[str] = Field(default_factory=list)
    take: int = 0
    paragraphs: int = 1
    notes: list[str] = Field(default_factory=list)

    def set_body(self) -> None:
        """Body paragraphs from the first `take` sentences of the pool."""
        chosen = self.pool[: max(0, self.take)]
        if not chosen:
            self.words.body = []
            return
        n = max(1, min(self.paragraphs, len(chosen) // 2 or 1))
        size = -(-len(chosen) // n)
        self.words.body = [" ".join(chosen[i:i + size]) for i in range(0, len(chosen), size)]


class Book(BaseModel):
    template: str
    template_name: str
    category: str
    orientation: str
    width_mm: float
    height_mm: float
    pages: list[BookPage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    palette: str = ""
    logo: Optional[str] = None
    logo_on_dark: bool = True     # whether the logo reads on a dark ground
    roles: dict = Field(default_factory=dict)
