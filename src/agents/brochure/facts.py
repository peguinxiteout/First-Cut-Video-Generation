"""The content model: what the person typed, made ready to print.

Raw form fields are never poured into a layout as typed. This module turns
them into the structures a brochure is designed from — and nothing in it
adds a fact the person did not give:

    configurations   "3 Bed + 3 Toilet + Study" -> 3 BHK + Study, 3 baths;
                     "1 BHK-A" -> 1 BHK, type A; the summary line
                     "1, 2, 3 & 4 BHK" for the cover and the overview
    unit rows        name, bath count, carpet area and price, formatted
                     (thousands separators, one "sq ft" style, a price range
                     written once: "₹29.95 – 31.15 Lakh")
    project figures  "25 Acres, 19 Towers, 1,875 Units" -> three figures
    claims           each selling point split into a headline and a detail
                     ("Pre-certified IGBC Gold project" / "Ensures 45% water
                     savings and 70% common area solar power"), with its
                     topic, its icon and any statistics in it
    amenity zones    "Recreational: 3 clubhouses, Meandering Pool, ..." ->
                     a zone with items; un-zoned lines grouped by topic
    landmarks        with icons and tidy distances
    audience         a short phrase for the copy ("first-time homebuyers
                     and young families")

The copywriter (copywriter.py) and the planner (planner.py) read only this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.agents.brochure import topics

# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def clean(text: Optional[str]) -> str:
    text = (text or "").replace(" ", " ").replace("’", "'")
    return _WS.sub(" ", text).strip(" \t\r\n;•-")


def sentence(text: str) -> str:
    """Capitalised, ending in one full stop."""
    text = clean(text).rstrip(".;,: ")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def capitalise(text: str) -> str:
    text = clean(text)
    return text[0].upper() + text[1:] if text else ""


def lower_first(text: str) -> str:
    """Lower-case the first letter unless the first word is an acronym or a
    proper noun (two capitals, or a capital after the first letter)."""
    text = clean(text)
    if not text:
        return ""
    first = text.split(" ", 1)[0]
    if len(first) > 1 and (first[1:2].isupper() or any(c.isdigit() for c in first)):
        return text
    if first[0].isupper() and first[1:].islower() and first.lower() in _COMMON_STARTS:
        return text[0].lower() + text[1:]
    return text


_COMMON_STARTS = {
    "landscaped", "jogging", "two", "three", "clubhouse", "swimming", "multi-play", "wellness",
    "lavish", "smart", "strict", "eco-conscious", "strategic", "sprawling", "proximity", "built",
    "located", "zero-wastage", "pre-certified", "primary", "hour", "outdoor", "indoor", "kids",
    "kids'", "budget-conscious", "environmentally", "high-net-worth", "tech", "affluent",
    "young", "first-time", "working", "families", "professionals", "nuclear", "growing",
}


def prose_list(items: list[str], conj: str = "and") -> str:
    items = [clean(i) for i in items if clean(i)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def split_top(text: str) -> list[str]:
    """Split a list typed as prose — 'A, B (x, y), and C' — at top-level
    commas and a final 'and', never inside brackets."""
    parts, depth, buf = [], 0, ""
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        digit_comma = ch == "," and i > 0 and text[i - 1].isdigit() and text[i + 1:i + 2].isdigit()
        if ch == "," and depth == 0 and not digit_comma:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    out = []
    for part in parts:
        part = clean(part).rstrip(".")
        part = re.sub(r"^and\s+", "", part, flags=re.I)
        if part:
            out.append(part)
    # "A and B" left in the last element (or the only one): two items, when
    # both are short names and neither is a clause ("X with Y for Z").
    if out and re.search(r"\sand\s", out[-1]) and "(" not in out[-1] \
            and not re.search(r"\b(with|for|of|in|to)\b", out[-1], re.I):
        a, b = re.split(r"\sand\s", out[-1], maxsplit=1)
        if len(a.split()) <= 7 and len(b.split()) <= 7:
            out[-1:] = [a, capitalise(b) if len(out) == 1 else b]
    return out


# ---------------------------------------------------------------------------
# numbers, areas, prices
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def group_digits(text: str) -> str:
    """1611 -> 1,611 (Indian buyers read lakh/crore; areas are printed with
    western grouping as in the reference brochures). Decimals untouched."""
    def fix(m):
        raw = m.group(0)
        if "," in raw or "." in raw.split(",")[0] and len(raw.split(".")[0]) < 4:
            return raw
        whole, _, frac = raw.partition(".")
        if len(whole) < 4:
            return raw
        return f"{int(whole):,}" + (f".{frac}" if frac else "")
    return re.sub(r"(?<![\d.])\d{4,}(?:\.\d+)?(?![\d])", fix, text)


def tidy_area(text: Optional[str]) -> str:
    """'1611 – 1915 sq.ft' -> '1,611 – 1,915 sq ft'."""
    text = clean(text)
    if not text:
        return ""
    text = re.sub(r"\s*(?:-|–|—|to)\s*(?=\d)", " – ", text)
    text = re.sub(r"\bsq\.?\s*ft\.?|\bsqft\b|\bsft\b|square\s+feet", "sq ft", text, flags=re.I)
    text = re.sub(r"\bsq\.?\s*m\.?(?![a-z])|\bsqm\b|square\s+metres?", "sq m", text, flags=re.I)
    text = group_digits(text)
    if re.fullmatch(r"[\d,.\s–]+", text):
        text += " sq ft"
    return text


def area_numbers(text: Optional[str]) -> list[float]:
    return [float(n.replace(",", "")) for n in _NUM.findall(text or "")]


def tidy_price(text: Optional[str]) -> str:
    """'₹29.95 Lakh – ₹31.15 Lakh' -> '₹29.95 – 31.15 Lakh';
    'Rs 52 Lakh onwards' -> '₹52 Lakh onwards'."""
    text = clean(text)
    if not text:
        return ""
    text = re.sub(r"\b(?:Rs\.?|INR)\s*", "₹", text, flags=re.I)
    text = re.sub(r"₹\s+", "₹", text)
    text = re.sub(r"\blakhs?\b", "Lakh", text, flags=re.I)
    text = re.sub(r"\b(?:crores?|cr)\b\.?", "Cr", text, flags=re.I)
    m = re.fullmatch(r"₹([\d.,]+)\s*(Lakh|Cr)\s*(?:-|–|—|to)\s*₹?([\d.,]+)\s*(Lakh|Cr)", text)
    if m and m.group(2) == m.group(4):
        return f"₹{m.group(1)} – {m.group(3)} {m.group(2)}"
    return re.sub(r"\s*(?:-|–|—)\s*", " – ", text)


def price_from(text: Optional[str]) -> str:
    """The starting figure of a price or price range: '₹29.95 Lakh'."""
    t = tidy_price(text)
    m = re.match(r"(₹[\d.,]+)(?:\s*–\s*[\d.,]+)?\s*(Lakh|Cr)?", t)
    if not m:
        return t
    return f"{m.group(1)} {m.group(2)}".strip() if m.group(2) else m.group(1)


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------


@dataclass
class UnitRow:
    label: str              # "3 BHK + Study"
    bhk: Optional[float]
    variant: str = ""       # "Type A"
    baths: Optional[int] = None
    study: bool = False
    area: str = ""          # "1,017.62 sq ft"
    area_lo: Optional[float] = None
    area_hi: Optional[float] = None
    price: str = ""
    plan_image: str = ""
    raw: str = ""

    @property
    def detail(self) -> str:
        """'3 baths' / 'Type A' — what distinguishes rows of one BHK."""
        bits = []
        if self.variant:
            bits.append(self.variant)
        if self.baths:
            bits.append(f"{self.baths} bath" + ("s" if self.baths > 1 else ""))
        return " · ".join(bits)


_BHK = re.compile(r"(\d+(?:\.\d+)?)\s*(?:BHK|BR|bed(?:room)?s?)\b", re.I)
_BATH = re.compile(r"(\d+)\s*(?:toilets?|baths?|bathrooms?|T)\b", re.I)


def _num(value: float) -> str:
    return f"{value:g}"


def parse_unit(name: str, area: str = "", price: str = "", plan: str = "") -> UnitRow:
    raw = clean(name)
    m = _BHK.search(raw)
    bhk = float(m.group(1)) if m else None
    bath = _BATH.search(raw)
    study = bool(re.search(r"\bstudy\b", raw, re.I))
    servant = bool(re.search(r"\bservant|\bmaid|\bstaff", raw, re.I))
    variant = ""
    v = re.search(r"BHK\s*[-–]?\s*(?:type\s*)?([A-Z])\b", raw, re.I) or re.search(r"\btype\s*([A-Z0-9]+)\b", raw, re.I)
    if v:
        variant = f"Type {v.group(1).upper()}"
    if bhk is not None:
        label = f"{_num(bhk)} BHK"
        if study:
            label += " + Study"
        if servant:
            label += " + Staff room"
        extra = re.sub(r"(\d+(?:\.\d+)?)\s*(?:BHK|BR|bed(?:room)?s?)\b|\d+\s*(?:toilets?|baths?|bathrooms?)\b|"
                       r"\bstudy\b|\+|[-–]\s*[A-Z]\b|\btype\s*[A-Z0-9]+\b|\bservant(?:\s*room)?|\bmaid(?:\s*room)?",
                       " ", raw, flags=re.I)
        extra = re.sub(r"[(\[]\s*[)\]]", " ", extra)        # brackets left empty: "3 BHK (Type A)"
        extra = clean(re.sub(r"\s+", " ", extra)).strip(" -–,")
        if extra and len(extra) <= 24 and not re.search(r"\d", extra):
            label += f" {extra}"
    else:
        label = raw
    areas = area_numbers(area)
    return UnitRow(label=label, bhk=bhk, variant=variant, baths=int(bath.group(1)) if bath else None,
                   study=study, area=tidy_area(area), area_lo=min(areas) if areas else None,
                   area_hi=max(areas) if areas else None, price=tidy_price(price), plan_image=plan,
                   raw=raw)


def config_summary(units: list[UnitRow], typed: str = "") -> str:
    """'1, 2, 3 & 4 BHK' — every BHK count once, in order."""
    counts = sorted({u.bhk for u in units if u.bhk is not None})
    if counts:
        nums = [_num(c) for c in counts]
        return (", ".join(nums[:-1]) + " & " + nums[-1] if len(nums) > 1 else nums[0]) + " BHK"
    typed = clean(typed)
    if typed:
        return typed
    return prose_list([u.label for u in units])


def residence_noun(category: str, property_type: str = "") -> str:
    pt = (property_type or "").lower()
    if "villa" in pt:
        return "Villas"
    if "row" in pt or "town" in pt:
        return "Townhomes"
    if "plot" in pt:
        return "Plots"
    if "penthouse" in pt:
        return "Penthouses"
    return {"affordable": "Homes", "mid": "Residences", "luxury": "Residences"}.get(category, "Homes")


# ---------------------------------------------------------------------------
# project figures and statistics
# ---------------------------------------------------------------------------


@dataclass
class Figure:
    value: str             # "25"
    unit: str              # "Acres"
    label: str = ""        # what it counts, in words


_FIGURE_UNITS = {
    "acre": "Acres", "acres": "Acres", "tower": "Towers", "towers": "Towers", "unit": "Residences",
    "units": "Residences", "homes": "Homes", "home": "Homes", "residences": "Residences",
    "floors": "Floors", "storeys": "Storeys", "villas": "Villas", "blocks": "Blocks",
    "apartments": "Residences", "wings": "Wings",
}


def count_figures(lines: list[str]) -> list[Figure]:
    """Counts stated in amenity lines that make a figure: '1,200+ trees'."""
    out = []
    for line in lines or []:
        for m in re.finditer(r"(\d[\d,]*\+?)\s+(trees|clubhouses|amenities|courts|parks)\b", line, re.I):
            out.append(Figure(value=group_digits(m.group(1)), unit=m.group(2).capitalize()))
    return out


def project_figures(text: Optional[str]) -> list[Figure]:
    """'25 Acres, 19 Towers, 1,875 Units' -> three figures."""
    out = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?\+?)\s*([A-Za-z]+)", clean(text)):
        unit = _FIGURE_UNITS.get(m.group(2).lower())
        if unit:
            out.append(Figure(value=group_digits(m.group(1)), unit=unit))
    return out


@dataclass
class Stat:
    value: str              # "45%"
    label: str              # "water savings"
    source: str = ""        # the claim it came from


_STOP = re.compile(r"\b(is|are|was|were|from|by|for|that|which|annually|across|of carpet|"
                   r"yielding|in the)\b", re.I)
_STAT_VALUE = re.compile(r"(\d[\d,]*(?:\.\d+)?\+?)\s*(%|tonnes)(?=\s)")


def stats_in(text: str) -> list[Stat]:
    """Statistics a sentence states, each with a short label read straight
    from the words after it ("45% water savings"). A figure whose label
    cannot be read that way is not made into a statistic."""
    out = []
    for m in _STAT_VALUE.finditer(text):
        value = m.group(1) + ("%" if m.group(2) == "%" else f" {m.group(2)}")
        rest = re.split(r"\s+and\s+|[.;(),]", text[m.end():], maxsplit=1)[0]
        words = re.sub(r"^\s*(of|in)\s+", "", rest, flags=re.I)
        cut = _STOP.search(words)
        if cut:
            words = words[:cut.start()]
        tokens = clean(words).split()
        words = " ".join(tokens[:6])
        if len(tokens) >= 2:
            out.append(Stat(value=value, label=words.lower(), source=text))
    return out


# ---------------------------------------------------------------------------
# claims: the selling points, headline + detail
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    text: str               # the selling point as given, cleaned
    headline: str
    detail: str
    topic: str              # connectivity | green | design | ...
    icon: str
    stats: list[Stat] = field(default_factory=list)
    figure: str = ""        # the number the claim turns on: "45%", "₹2.67 Lakh"


CLAIM_TOPICS: list[tuple[str, str, tuple[str, ...]]] = [
    ("offer", "tag", ("rebate", "subsidy", "pmay", "savings of", "offer", "discount", "emi",
                      "payment plan", "registration savings", "cashback", "assured")),
    ("green", "leaf", ("net zero", "igbc", "leed", "griha", "solar", "green building", "rainwater",
                       "carbon", "water savings", "waste", "sustainab", "eco", "recycl",
                       "water management", "energy")),
    ("connectivity", "transit", ("metro", "railway", "station", "airport", "highway", "expressway",
                                 "corridor", "orr", "ring road", "connect", "access to", "minutes from",
                                 "proximity", "tech hub", "it hub", "equidistant", "located")),
    ("privacy", "security", ("privacy", "private", "no two", "face each other", "secure", "security")),
    ("design", "building", ("architecture", "design", "inspired", "facade", "façade", "columns",
                            "arches", "classical", "contemporary")),
    ("space", "area", ("open space", "acre", "layout", "vehicle-free", "zero-wastage", "carpet area",
                       "space", "landscap")),
    ("leisure", "club", ("clubhouse", "club", "pool", "amenit", "sports", "theatre")),
    ("quality", "check", ("quality", "construction", "backward-integrated", "precast", "checks",
                          "built with", "trust", "legacy")),
]


def claim_topic(text: str) -> tuple[str, str]:
    low = text.lower()
    best, best_hits = ("general", "star"), 0
    for topic, icon, keys in CLAIM_TOPICS:
        hits = sum(1 for k in keys if k in low)
        if hits > best_hits:
            best, best_hits = (topic, icon), hits
    return best


# Verbs and connectives that separate a selling point's headline from its
# detail. Participles become present tense in the detail.
_SPLITS = [
    (r":\s+", ""), (r"\s+\((?=[^()]+\)$)", "("), (r"\s+(?:is|are)\s+", ""), (r",?\s+featuring\s+", "Features "), (r",?\s+ensuring\s+", "Ensures "),
    (r",?\s+offering\s+", "Offers "), (r",?\s+spanning\s+", "Spans "),
    (r",?\s+yielding\s+", "Yields "), (r"\s+diverts\s+", "Diverts "),
    (r",?\s+providing\s+", "Provides "), (r",?\s+delivering\s+", "Delivers "),
    (r"\s+with\s+", "With "), (r"\s+along\s+", "Along "), (r"\s+up to\s+", "Up to "),
    (r"\s+near\s+", "Near "), (r"\s+for\s+", "For "),
]

# Register polish: a word the category's voice would not use, replaced by
# one it would. The fact is unchanged.
POLISH = {
    "luxury": {r"\bmassive\b": "grand", r"\bhuge\b": "generous", r"\blavish\b": "distinguished",
               r"\bcheap\b": "", r"\bbudget\b": ""},
    "mid": {r"\bmassive\b": "expansive", r"\bhuge\b": "generous"},
    "affordable": {r"\bmassive\b": "large"},
}

# Headlines when a selling point has no natural headline of its own (it
# begins "Located on...").
TOPIC_HEADLINES = {
    "affordable": {"connectivity": "Metro-ready location", "green": "Green, lower-cost living",
                   "offer": "Extra savings", "privacy": "Safe and secure", "design": "Smart design",
                   "space": "More usable space", "leisure": "Leisure on site", "quality": "Built to last",
                   "general": "Good to know"},
    "mid": {"connectivity": "Connected by design", "green": "Sustainable by design",
            "offer": "Thoughtful value", "privacy": "Privacy, planned in", "design": "Considered design",
            "space": "Room to breathe", "leisure": "Leisure at home", "quality": "Built with care",
            "general": "Worth knowing"},
    "luxury": {"connectivity": "A connected address", "green": "Quietly sustainable",
               "offer": "Privileges", "privacy": "Privacy, assured", "design": "Architecture",
               "space": "Space, held in reserve", "leisure": "The art of leisure",
               "quality": "Uncompromising craft", "general": "Distinction"},
}


def polish(text: str, category: str) -> str:
    for pattern, repl in POLISH.get(category, {}).items():
        text = re.sub(pattern, repl, text, flags=re.I)
    return clean(re.sub(r"\s{2,}", " ", text))


def parse_claim(text: str, category: str = "mid") -> Claim:
    text = polish(clean(text).rstrip("."), category)
    topic, icon = claim_topic(text)
    headline, detail = "", ""
    lead = re.match(r"^(?:located|situated|set|positioned)\s+(?:on|in|at|along|near)\s+", text, re.I)
    if lead:
        headline = TOPIC_HEADLINES.get(category, TOPIC_HEADLINES["mid"])[topic]
        detail = sentence(capitalise(re.sub(r"^\w+\s+", "", text[lead.start():])))
    else:
        for pattern, verb in _SPLITS:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            head, rest = text[:m.start()].strip(" ,"), text[m.end():].strip()
            if verb == "(":
                verb, rest = "", rest.rstrip(")")
            if 10 <= len(head) <= 72 and len(rest) >= 6:
                headline = head
                detail = sentence((verb + rest) if verb else capitalise(rest))
                break
    if not headline:
        if len(text) <= 60:
            headline, detail = text, ""
        else:
            headline = TOPIC_HEADLINES.get(category, TOPIC_HEADLINES["mid"])[topic]
            detail = sentence(text)
    headline = re.sub(r"^(built with|with)\s+", "", headline, flags=re.I)
    headline = capitalise(headline.rstrip("."))
    return Claim(text=sentence(text), headline=headline, detail=detail, topic=topic, icon=icon,
                 stats=stats_in(text), figure=key_figure(text))


_FIGURE = re.compile(
    r"(₹\s?\d[\d,]*(?:\.\d+)?\+?(?:\s?(?:Lakh|Cr|crore))?|\d[\d,]*(?:\.\d+)?\+?\s?(?:%|sq\.?\s?ft\.?|"
    r"km|acres?|tonnes|quality checks|units|towers|trees|clubhouses))", re.I)


def key_figure(text: str) -> str:
    """The first figure a selling point states, tidied ('25+ acre' -> '25+
    acres', '7 sq.ft' -> '7 sq ft'). Empty when it states none."""
    m = _FIGURE.search(text or "")
    if not m:
        return ""
    value = re.sub(r"\s+", " ", m.group(1)).strip()
    value = re.sub(r"sq\.?\s?ft\.?", "sq ft", value, flags=re.I)
    value = re.sub(r"(\d\+?)\s?acre\b", r"\1 acres", value, flags=re.I)
    return group_digits(value)


ALT_HEADLINES = {
    "green": ("Greener every day", "Lighter on the planet", "Built for tomorrow"),
    "connectivity": ("Everything within reach", "Shorter commutes", "On the growth corridor"),
    "offer": ("More value for you", "Savings that add up", "Helpful benefits"),
    "design": ("Thoughtful design", "Design with purpose", "Made to last"),
    "space": ("Space that works harder", "Open by design", "Room to grow"),
    "leisure": ("Leisure, close at hand", "Weekends sorted", "Time well spent"),
    "quality": ("Quality you can trust", "Built with care", "Attention to detail"),
    "privacy": ("Peace of mind", "Your own world", "Quiet by design"),
    "general": ("Good to know", "Worth knowing", "One more reason"),
}


def _unique_headlines(claims: list[Claim], category: str) -> None:
    """No two cards share a headline."""
    used: set[str] = set()
    for claim in claims:
        if claim.headline.lower() in used:
            for alt in ALT_HEADLINES.get(claim.topic, ALT_HEADLINES["general"]):
                if alt.lower() not in used:
                    claim.headline = alt
                    break
        used.add(claim.headline.lower())


# ---------------------------------------------------------------------------
# amenities
# ---------------------------------------------------------------------------


@dataclass
class Zone:
    title: str
    topic: str
    icon: str
    items: list[str] = field(default_factory=list)


ZONE_TITLES = {
    "care": "Health & childcare", "green": "Sustainable living",
    "pool": "Pool & water", "clubhouse": "Clubhouse", "fitness": "Health & wellness",
    "sports": "Sports & fitness", "kids": "For children", "landscape": "Gardens & open spaces",
    "security": "Safety & security", "convenience": "Everyday convenience", "other": "More on site",
}


def _amenity_item(text: str) -> str:
    text = clean(text).rstrip(".")
    text = re.sub(r"\s+featuring\s+", " with ", text, flags=re.I)
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.I)
    return capitalise(text)


def amenity_zones(amenities: list[str], groups: Optional[list] = None) -> list[Zone]:
    """Zones as the person named them ("Kids' Zone: ..."), else grouped by
    topic. Every item once."""
    zones: list[Zone] = []
    seen: set[str] = set()

    def add(zone: Zone, items: list[str]) -> None:
        for item in items:
            key = item.lower()
            if key and key not in seen:
                seen.add(key)
                zone.items.append(item)

    for g in groups or []:
        title = clean(getattr(g, "zone", "") or "")
        topic = topics.of_amenity(title) or "other"
        zone = Zone(title=title or ZONE_TITLES.get(topic, "Amenities"), topic=topic,
                    icon=topics.icon(topic) if topic != "other" else "star")
        add(zone, [_amenity_item(i) for i in getattr(g, "items", []) or []])
        if zone.items:
            zones.append(zone)

    by_topic: dict[str, Zone] = {}
    for line in amenities or []:
        line = clean(line)
        if not line:
            continue
        m = re.match(r"^([A-Z][A-Za-z'&/ ]{2,28}?)\s*:\s*(.+)$", line)
        if m:
            title, rest = m.group(1).strip(), m.group(2)
            items = [_amenity_item(i) for i in split_top(rest)]
            topic = topics.match(title, topics.AMENITY_TOPICS) or topics.of_amenity(rest)
            zone = Zone(title=_zone_title(title), topic=topic,
                        icon=topics.icon(topic) if topic != "other" else "star")
            add(zone, items)
            if zone.items:
                zones.append(zone)
            continue
        items = [_amenity_item(i) for i in split_top(line)]
        topic = topics.of_amenity(line) or "other"
        if claim_topic(line)[0] == "green":
            topic = "green"
        # "Clubhouse with a hall, games ..." -> the clubhouse zone lists
        # "Multi-purpose hall", not "Clubhouse with multi-purpose hall"
        if items:
            head = re.match(r"^(.+?)\s+(?:with|featuring)\s+(.+)$", items[0], re.I)
            if head and topics.of_amenity(head.group(1)) == topic != "other":
                items[0] = capitalise(head.group(2))
        if topic not in by_topic:
            mark = "leaf" if topic == "green" else (topics.icon(topic) if topic != "other" else "star")
            by_topic[topic] = Zone(title=ZONE_TITLES.get(topic, "More on site"), topic=topic, icon=mark)
        add(by_topic[topic], items)
    zones.extend(z for z in by_topic.values() if z.items)
    return zones


def _zone_title(title: str) -> str:
    title = clean(title)
    if title.isupper():
        title = title.title()
    return title


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------


@dataclass
class Place:
    name: str
    distance: str
    kind: str
    icon: str


_KIND_LABEL = {"transit": "Metro & rail", "school": "Schools", "hospital": "Healthcare",
               "retail": "Shopping", "work": "Workplaces", "airport": "Airport",
               "leisure": "Leisure", "road": "Roads", "hotel": "Hospitality"}


def tidy_distance(text: str) -> str:
    text = clean(text)
    if not text or text.lower() in ("proximity", "nearby", "close by", "near"):
        return ""
    text = re.sub(r"(\d)\s*(km|m|min|mins)\b", r"\1 \2", text, flags=re.I)
    text = text.replace("~", "≈ ").replace("≈  ", "≈ ")
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1–\2", text)
    text = re.sub(r"\(\s*(\d+)\s*min\s*\)", r"· \1 min", text)
    return text


def places(landmarks: list) -> list[Place]:
    out = []
    for lm in landmarks or []:
        name = clean(getattr(lm, "name", ""))
        if not name:
            continue
        kind = (getattr(lm, "category", "") or "").lower().strip()
        low = (kind + " " + name).lower()
        if "tech" in low or "it park" in low or "business park" in low or "office" in low:
            kind = "work"
        elif "metro" in low or "station" in low or "rail" in low:
            kind = "transit"
        elif "hotel" in low:
            kind = "hotel"
        elif "mall" in low or "supermarket" in low or "retail" in low or "dmart" in low or "market" in low:
            kind = "retail"
        elif "hospital" in low or "clinic" in low:
            kind = "hospital"
        elif "school" in low or "college" in low:
            kind = "school"
        dist = " · ".join(x for x in (tidy_distance(getattr(lm, "distance", "")),
                                       tidy_distance(getattr(lm, "minutes", ""))) if x)
        out.append(Place(name=name, distance=dist, kind=kind or "other",
                         icon=topics.landmark_icon(kind, name)))
    # the places a buyer weighs first; a hotel is a landmark, not a need
    priority = {"transit": 0, "airport": 1, "work": 2, "school": 3, "hospital": 4, "retail": 5,
                "road": 6, "leisure": 7, "other": 8, "hotel": 9}
    return sorted(out, key=lambda p: priority.get(p.kind, 8))


def kind_label(kind: str) -> str:
    return _KIND_LABEL.get(kind, "Nearby")


# ---------------------------------------------------------------------------
# the audience
# ---------------------------------------------------------------------------


def audience_phrase(text: Optional[str], limit: int = 2) -> str:
    """'Budget-conscious nuclear families, first-time homebuyers, and working
    professionals in ...' -> 'budget-conscious nuclear families and
    first-time homebuyers'."""
    text = clean(text)
    if not text:
        return ""
    text = re.split(r"\s+(?:seeking|looking|who|wanting|in search)\b", text, maxsplit=1)[0]
    groups = []
    for part in split_top(text):
        part = re.split(r"\s+(?:in|working in|from|traversing|along|across)\s+(?:the\s+)?[A-Z]", part)[0]
        part = re.sub(r"\s+(?:working|traversing)$", "", part)
        part = clean(part)
        if part:
            groups.append(lower_first(part))
    return prose_list(groups[:limit])


# ---------------------------------------------------------------------------
# the whole model
# ---------------------------------------------------------------------------


@dataclass
class Content:
    name: str
    developer: str
    city: str
    locality: str
    address: str
    category: str
    property_type: str
    tagline: str
    units: list[UnitRow]
    config: str                 # "1, 2, 3 & 4 BHK"
    noun: str                   # Homes | Residences | Villas
    price: str                  # as typed, tidy
    price_from: str             # "₹29.95 Lakh"
    area_range: str
    figures: list[Figure]
    possession: str
    rera: str
    claims: list[Claim]
    zones: list[Zone]
    amenity_names: list[str]
    amenity_lines: list[str]
    places: list[Place]
    audience: str
    audience_full: str
    cta: str
    phone: str
    email: str
    website: str
    visit_address: str
    description: str
    developer_story: str

    @property
    def stats(self) -> list[Stat]:
        out, seen = [], set()
        sources = [c.stats for c in self.claims] + [stats_in(a) for a in self.amenity_lines]
        for group in sources:
            for s in group:
                if s.value not in seen:
                    seen.add(s.value)
                    out.append(s)
        return out

    def claims_on(self, *topic_names: str) -> list[Claim]:
        return [c for c in self.claims if c.topic in topic_names]

    @property
    def place_phrase(self) -> str:
        return ", ".join(x for x in (self.locality, self.city) if x)

    @property
    def short_name(self) -> str:
        return self.name


def build(prop, category: str) -> Content:
    units = [parse_unit(u.name, u.carpet_area or u.built_up_area, u.price, u.plan_image)
             for u in (prop.unit_types or []) if clean(u.name)]
    lo = [u.area_lo for u in units if u.area_lo]
    hi = [u.area_hi for u in units if u.area_hi]
    if lo and hi:
        unit = "sq ft"
        area_range = (f"{group_digits(_num(min(lo)))} – {group_digits(_num(max(hi)))} {unit}"
                      if max(hi) > min(lo) else f"{group_digits(_num(min(lo)))} {unit}")
    else:
        area_range = tidy_area(prop.area)
    loc = prop.location
    contact = prop.contact
    price = tidy_price(prop.price) or next((u.price for u in units if u.price), "")
    claims = [parse_claim(u, category) for u in (prop.usps or []) if clean(u)]
    for h in getattr(prop, "highlights", None) or []:
        if clean(h.label):
            claims.append(parse_claim(h.label, category))
    _unique_headlines(claims, category)
    zones =amenity_zones(prop.amenities or [], getattr(prop, "amenity_groups", None))
    names = [i for z in zones for i in z.items]
    return Content(
        name=clean(prop.name), developer=clean(prop.developer), city=clean(loc.city),
        locality=clean(loc.area), address=clean(loc.address), category=category,
        property_type=clean(prop.property_type), tagline=clean(prop.tagline),
        units=units, config=config_summary(units, prop.configuration or ""),
        noun=residence_noun(category, prop.property_type or ""),
        price=price, price_from=price_from(price) if price else "",
        area_range=area_range,
        figures=project_figures(prop.project_size), possession=clean(prop.possession),
        rera=clean(prop.rera_number), claims=claims, zones=zones, amenity_names=names,
        amenity_lines=[clean(a) for a in prop.amenities or [] if clean(a)],
        places=places(prop.landmarks), audience=audience_phrase(prop.target_audience),
        audience_full=clean(prop.target_audience), cta=clean(prop.cta),
        phone=clean(contact.phone) if contact else "", email=clean(contact.email) if contact else "",
        website=re.sub(r"^https?://", "", clean(contact.website) if contact else "").rstrip("/"),
        visit_address=clean(contact.address) if contact and contact.address else clean(loc.address),
        description=clean(prop.description), developer_story=clean(getattr(prop, "developer_story", "")),
    )
