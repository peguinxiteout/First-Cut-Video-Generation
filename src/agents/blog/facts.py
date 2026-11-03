"""Facts the post may state, each with a category, a source and a status.

The writer's raw material is a list of strings from the brief plus a
research pack. Turning them into typed facts does three jobs at once:

  * it lets a section draw the facts that belong to it (connectivity
    facts in the commute section, prices in the value section), which is
    what makes a post specific instead of generic;
  * every fact carries its STATUS — supplied by the publisher, measured,
    an estimate, merely announced — so the post can say which is which
    (the brief asked for verified facts, estimates and recommendations to
    be clearly distinguished);
  * every fact keeps its SOURCE, so the references section is built from
    where facts actually came from, never from model output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.agents.blog import blog_config
from src.data.schemas import BlogInput, ResearchPack

CATEGORY_TITLES = {
    "price": "Prices",
    "connectivity": "Connectivity",
    "employment": "Jobs and offices",
    "infrastructure": "Upcoming infrastructure",
    "legal": "Registration & approvals",
    "social": "Schools & healthcare",
    "amenity": "Everyday amenities",
    "project": "Project details",
    "market": "Market signals",
    "general": "Other facts",
}

_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
# The first figure in a fact — where its label ends and its value begins.
_FIGURE = re.compile(
    r"(?:₹|rs\.?\s?|inr\s?|\$)\s?\d"
    r"|(?:" + _MONTHS + r"\s+)?\b(?:19|20)\d{2}\b"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:%|per\s?cent|lakh|crore|cr\b|km|kms|m\b|min|mins|minutes|hours?|"
    r"sq\.?\s?ft|sq\.?\s?m|sqft|acres?|years?|storeys|floors|towers|units|homes)"
    r"|\b\d[\d,]{2,}",
    re.I)
_STATUS_SPLIT = re.compile(r"\b(under construction|proposed|upcoming|planned|sanctioned|"
                           r"expected to [a-z]+|being built|announced)\b", re.I)
# Words that belong with the value, not the label: "listed at ₹55 lakh".
_VALUE_LEAD = re.compile(r"\s*(?:\b(?:is|are|was|were|be)\s+)?(?:\b(?:listed|priced|located|available|"
                         r"starting|expected|approximately|roughly|about|around|nearly|within|at|from|"
                         r"just|only|over|under|up\s+to)\b\s*)+$", re.I)
_SOURCE_SUFFIX = re.compile(r"\s*[\(\[]\s*(?:source|src|via|per|according to)\s*[:\-]?\s*([^)\]]+)[\)\]]\s*\.?$"
                            r"|\s+[—-]+\s*source\s*:\s*(.+)$", re.I)


@dataclass
class Fact:
    text: str
    category: str
    label: str
    value: str
    source: str
    status: str
    url: str = ""
    origin: str = "brief"            # brief | research
    numbers: set = field(default_factory=set)

    @property
    def sentence(self) -> str:
        """The fact as a sentence, ending in a full stop."""
        text = self.text.rstrip(" .;")
        return text[:1].upper() + text[1:] + "."


def categorise(text: str) -> str:
    lowered = text.lower()
    for category, patterns in blog_config.editorial().get("fact_categories", {}).items():
        if any(re.search(p, lowered) for p in patterns):
            return category
    return "general"


def _numbers(text: str) -> set[str]:
    from src.agents.blog.fact_check import numbers_in

    return numbers_in(text)


_FROM_PLACE = re.compile(
    r"^(?P<place>.+?)\s+(?:is|lies|sits)\s+(?P<q>(?:about|around|approximately|roughly|just|only|nearly)\s+)?"
    r"(?P<fig>\d[\d.,]*\s?(?:km|kms|kilometres|m|metres|min|mins|minutes))\s+(?:away\s+)?(?:from|to)\s+"
    r"(?P<dest>.+)$", re.I)


_PREDICATE = re.compile(r"^(?P<subj>[^,]{3,60}?)\s+(?:is|are|lies|sits)\s+(?P<rest>(?:a|an|about|around|"
                        r"within|just|only|nearly|roughly|approximately)\b.*\d.*)$", re.I)


def split_label_value(text: str, category: str) -> tuple[str, str]:
    """"Metro station 1.2 km away" -> ("Metro station", "1.2 km away")."""
    text = text.strip().rstrip(".")
    predicate = _PREDICATE.match(text)
    if predicate and not _FROM_PLACE.match(text):   # "Lake Garden is a 10-minute walk from..."
        return _cap(predicate.group("subj").strip()), _cap(predicate.group("rest").strip())
    trip = _FROM_PLACE.match(text)
    if trip:     # "Shela is about 4 km from SP Ring Road" -> ("SP Ring Road", "About 4 km from Shela")
        q = (trip.group("q") or "").strip()
        return _cap(trip.group("dest").strip()), _cap(f"{q} {trip.group('fig')} from {trip.group('place')}".strip())
    if ":" in text:
        head, _, tail = text.partition(":")
        if 0 < len(head.split()) <= 6 and tail.strip():
            return head.strip(), _cap(tail.strip())
    for pattern in (_FIGURE, _STATUS_SPLIT):
        match = pattern.search(text)
        if match and match.start() > 0:
            head, tail = text[:match.start()], text[match.start():]
            lead = _VALUE_LEAD.search(head)
            if lead and lead.start() > 0:
                head, tail = head[:lead.start()], head[lead.start():].strip() + " " + tail
            head = head.strip(" ,;-")
            tail = re.sub(r"^(?:is|are|was|were)\s+", "", tail.strip(), flags=re.I)
            if head and len(head.split()) <= 9:
                return _cap(head), _cap(tail)
    return CATEGORY_TITLES.get(category, "Fact"), _cap(text)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _publisher(blog_input: BlogInput) -> str:
    brand = (getattr(blog_input, "brand_name", "") or "").strip()
    return f"Supplied by {brand}" if brand else "Supplied by the publisher"


def from_brief(blog_input: BlogInput) -> list[Fact]:
    facts: list[Fact] = []
    for raw in blog_input.local_facts:
        text = " ".join(str(raw).split())
        if not text:
            continue
        source = _publisher(blog_input)
        match = _SOURCE_SUFFIX.search(text)
        if match:
            named = (match.group(1) or match.group(2) or "").strip()
            text = text[:match.start()].strip()
            if named:
                source = named
        category = categorise(text)
        label, value = split_label_value(text, category)
        status = "Announced" if category == "infrastructure" else "Supplied"
        facts.append(Fact(text=text, category=category, label=label, value=value, source=source,
                          status=status, origin="brief", numbers=_numbers(text)))
    return facts


_KIND_CATEGORY = {
    "school": "social", "college": "social", "university": "social", "hospital": "social",
    "clinic": "social", "doctors": "social", "kindergarten": "social",
    "station": "connectivity", "metro_station": "connectivity", "railway_station": "connectivity",
    "bus_station": "connectivity", "airport": "connectivity",
}


def from_research(research: Optional[ResearchPack]) -> list[Fact]:
    if research is None:
        return []
    facts: list[Fact] = []
    for place in research.nearby:
        kind = (place.kind or "place").replace("_", " ")
        category = _KIND_CATEGORY.get(place.kind, "amenity")
        value = f"{_cap(kind)}, about {place.distance_km:g} km away in a straight line"
        facts.append(Fact(text=f"{place.name} ({kind}) is about {place.distance_km:g} km away in a straight line",
                          category=category, label=place.name, value=value, source="OpenStreetMap",
                          status="Measured", origin="research", numbers=_numbers(str(place.distance_km))))
    for journey in research.journeys:
        dest = _cap(journey.destination)
        value = (f"{journey.distance_km:g} km by road; about {journey.free_flow_minutes} min "
                 "in clear traffic")
        facts.append(Fact(text=f"{dest}: {value}", category="connectivity", label=dest, value=value,
                          source="OSRM road routing", status="Estimate (free-flow)", origin="research",
                          numbers=_numbers(f"{journey.distance_km} {journey.free_flow_minutes}")))
    for item in research.news:
        if not item.title:
            continue
        category = "infrastructure" if _STATUS_SPLIT.search(item.title) or categorise(item.title) == \
            "infrastructure" else categorise(item.title)
        facts.append(Fact(text=item.title, category=category if category != "general" else "market",
                          label=f"“{item.title.strip()}”", value=item.published or "Date not given",
                          source=item.source or "News report", status="Reported", url=item.url,
                          origin="research", numbers=_numbers(item.title)))
    for ref in research.official_sources:
        if not ref.note:
            continue
        category = categorise(ref.note)
        facts.append(Fact(text=ref.note, category=category, label=ref.title,
                          value=_cap(ref.note[:180]), source=ref.title, status="Official source",
                          url=ref.url, origin="research", numbers=_numbers(ref.note)))
    return facts


def collect(blog_input: BlogInput, research: Optional[ResearchPack] = None) -> list[Fact]:
    """Brief facts first (the publisher vouches for them), then research."""
    return from_brief(blog_input) + from_research(research)


def by_category(facts: list[Fact]) -> dict[str, list[Fact]]:
    out: dict[str, list[Fact]] = {}
    for fact in facts:
        out.setdefault(fact.category, []).append(fact)
    return out


def ordered(facts: list[Fact], priorities: list[str]) -> list[Fact]:
    """Facts in the order that matters to this reader."""
    rank = {c: i for i, c in enumerate(priorities)}
    return sorted(facts, key=lambda f: (rank.get(f.category, len(rank)), f.origin != "brief"))


def mentioning(facts: list[Fact], name: str) -> list[Fact]:
    """Facts that name `name` (a comparison option, a place)."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 2]
    if not words:
        return []
    return [f for f in facts if all(w in f.text.lower() for w in words)]
