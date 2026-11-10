"""What the script can say about the property from its own facts, beyond its amenities.

The brochure's `slots()` (agents/brochure/copywriter.py) is shared and gives values with their noun
attached ("420 residences", "4.3 acres"). The video's lines need the bare numbers and a few things the
brochure has no use for, so they are worked out here, from the property's own fields only:

    units, home_noun   "420" and "homes" from a project size such as "4.3 Acres, 420 Units"
    acres, towers      "4.3" and "4"
    bed_list           "1, 2, 3 and 4" from the unit types' bedroom counts, only when EVERY unit type
                       gives one (a list that leaves a type out would say less than the property has)

`claim()` makes a selling point short enough to say by cutting, never by rewording: a parenthetical goes,
and so does a label before a colon ("Strict privacy design: ..."); what is left is the property's own
words or nothing. Carpet areas are never used: their decimals are not something a narrator can say.
"""
from __future__ import annotations

import re

from src.agents.brochure import facts
from src.data.schemas import PropertyInput

HOME_NOUNS = {"unit": "homes", "units": "homes", "home": "homes", "homes": "homes", "residence": "residences",
              "residences": "residences", "apartment": "apartments", "apartments": "apartments", "flat": "flats",
              "flats": "flats", "villa": "villas", "villas": "villas"}
_UNITS = re.compile(r"(\d[\d,]*)\s*(units?|homes?|residences?|apartments?|flats?|villas?)\b", re.I)
_ACRES = re.compile(r"(\d+(?:\.\d+)?)\s*acres?\b", re.I)
_TOWERS = re.compile(r"(\d+)\s*towers?\b", re.I)
_BEDS = re.compile(r"\b(\d+)\s*(?:bed|bhk)\b", re.I)
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")
_LABEL = re.compile(r"^([^:]{1,48}):\s*(\S.*)$")


def _count(pattern: re.Pattern, text: str) -> str:
    """The number the pattern finds, or "" (a count of 1 is left out: "1 acres" is not a line)."""
    match = pattern.search(text or "")
    value = match.group(1).replace(",", "") if match else ""
    return "" if value in ("", "0", "1") else value


def bed_list(prop: PropertyInput) -> str:
    """"1, 2, 3 and 4" from the unit types, or "" when there are none or one of them gives no bedroom count."""
    counts: list[int] = []
    for unit in prop.unit_types:
        match = _BEDS.search(unit.name or "")
        if not match:
            return ""
        counts.append(int(match.group(1)))
    return facts.prose_list([str(c) for c in sorted(set(counts))]) if counts else ""


def video_slots(prop: PropertyInput) -> dict[str, str]:
    """The slots the script's general lines use. A slot with nothing behind it is "" and the line is skipped."""
    size = prop.project_size or ""
    noun = _UNITS.search(size)
    return {"units": _count(_UNITS, size), "home_noun": HOME_NOUNS.get(noun.group(2).lower(), "") if noun else "",
            "acres": _count(_ACRES, size), "towers": _count(_TOWERS, size), "bed_list": bed_list(prop)}


def claim(text: str, max_words: int = 10, min_words: int = 3) -> str:
    """A selling point cut down to something a narrator can say, or "" when that cannot be done without
    rewording it: no parenthetical, no label before a colon, ending without a full stop. Between `min_words`
    and `max_words` words, else nothing."""
    text = facts.clean(text)
    text = _PARENTHETICAL.sub("", text)
    label = _LABEL.match(text)
    if label and len(label.group(1).split()) <= 6:
        text = label.group(2)
    text = re.sub(r"\s+", " ", text).strip(" .;,:-*")
    return text if min_words <= len(text.split()) <= max_words else ""


def interior_photographs(assets) -> bool:
    """Whether a usable photograph shows a room of a home (read by a vision model): the homes line wants one."""
    return any(a.usable and a.room for a in assets or [])


# An area is never spoken: a narrator cannot say "1,017.62 sq.ft" and a built-up area is not a selling point.
# Acres are the one exception, in the scale line ("420 homes across 4.3 acres"), which is made from the slots.
_AREA_UNIT = (r"(?:sq(?:uare)?\.?\s*(?:ft|feet|foot|m|mt|mtr|mtrs|meters?|metres?|yd|yards?)\b\.?"
              r"|sqft|sqm|sft|hectares?)")
_AREA = re.compile(_AREA_UNIT, re.I)
_ACRE = re.compile(r"\bacres?\b", re.I)
_CLAUSES = re.compile(r"\s*(?:(?<!\d),|,(?!\d)|;|\s[-\u2013\u2014]\s)\s*")      # the comma in 25,000 is no break


_DANGLING = re.compile(r"^(?:with|and|but|or|for|of|in|at|on|by|to|from|near|along|which|that|where)\b", re.I)


def has_area(text: str, acres_ok: bool = False) -> bool:
    """Whether a text gives an area (sq ft, sq.ft, sqm, hectares, and acres unless `acres_ok`)."""
    return bool(_AREA.search(text or "")) or (not acres_ok and bool(_ACRE.search(text or "")))


def without_area(text: str) -> str:
    """The text without its clauses (cut at commas, semicolons and dashes) that give an area; "" when that
    leaves nothing. Only cutting: what is left is the text's own words."""
    text = facts.clean(text)
    if not has_area(text):
        return text
    kept = [c for c in _CLAUSES.split(text) if c.strip() and not has_area(c)]
    cut = ", ".join(kept).strip(" .;,:-*")
    return "" if _DANGLING.match(cut) else cut                      # "with 4 towers" is the rest of a sentence, not a line

