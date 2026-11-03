"""The copy: written from the project's own facts, in the category's voice.

Every page of the planned book (planner.py) is written here. Two writers,
one standard:

    the model    one call for the whole book (Gemini on the free tier, or the
                 OpenRouter fallback): the fact sheet, the voice, and for
                 each page what it must say and how much room it has
    the writer   offline — frames from voice.py filled ONLY from the content
                 model (facts.py): the project's name, place, figures,
                 amenities, places and selling points. Several frames per
                 job, chosen by a stable hash of the project's name, so two
                 projects in one category do not read alike

Running copy is written as a POOL of sentences in order; the page sets the
first `take` of them. The render-and-measure loop (qa.py) moves `take` up
or down by whole sentences until the copy fits its slot — type sizes never
change to fill or to fit.

THE GUARD — nothing is printed that:
    * states a number not in the property's own data
    * repeats a sentence or a title used elsewhere in the brochure
    * uses a word the category's register rules out
    * describes the photograph ("this image shows", "pictured")
"""
from __future__ import annotations

import json
import logging
import random
import re
import zlib
from string import Template
from typing import Optional

from src.agents.base import with_retries
from src.agents.brochure import facts, strategy, voice
from src.agents.brochure.book import Book, BookPage, Card, Copy
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)

DESCRIPTIVE = re.compile(
    r"\b(this (image|photo|photograph|picture|render)|the (image|photo|photograph|picture|render) "
    r"(shows|depicts)|pictured|depicts|can be seen|is shown|in the foreground|in the background|"
    r"artistic impression|artist'?s impression)\b", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

ZONE_PHRASE = {
    "pool": "the pools", "clubhouse": "the clubhouse", "sports": "the sports courts",
    "kids": "the children's play areas", "landscape": "the gardens", "fitness": "the fitness spaces",
    "care": "on-site health and childcare", "security": "on-site security",
    "convenience": "everyday conveniences", "green": "the green systems", "other": "",
}


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


class Guard:
    def __init__(self, prop, content: facts.Content, positioning: strategy.Positioning):
        try:
            raw = prop.model_dump_json()
        except Exception:  # noqa: BLE001
            raw = str(prop)
        self.facts = (raw + " " + json.dumps(content.__dict__, default=lambda o: getattr(o, "__dict__", str(o)),
                                             ensure_ascii=False)).replace(",", "")
        self.positioning = positioning
        self.seen: set[str] = set()

    @staticmethod
    def key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def numbers_ok(self, text: str) -> bool:
        return all(n.replace(",", "") in self.facts for n in _NUMBER.findall(text))

    def register_ok(self, text: str) -> bool:
        low = text.lower()
        return not any(w.lower() in low for w in self.positioning.avoid)

    def ok(self, text: str) -> bool:
        if not text or DESCRIPTIVE.search(text):
            return False
        return self.numbers_ok(text) and self.register_ok(text) and self.key(text) not in self.seen

    def keep(self, text: str) -> str:
        self.seen.add(self.key(text))
        return text


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------


def short_name(c: facts.Content) -> str:
    """'Mahindra Happinest Kalyan' -> 'Happinest Kalyan' when the developer's
    brand opens the name and two words remain; otherwise the name."""
    words = c.name.split()
    dev = (c.developer or "").split()
    if dev and words and words[0].lower() == dev[0].lower() and len(words) >= 3:
        return " ".join(words[1:])
    return c.name


def developer_name(c: facts.Content) -> str:
    name = re.sub(r"\b(Pvt\.?|Private|Ltd\.?|Limited|LLP|Inc\.?|Developers?|Enterprises|Group|Realty)\b",
                  "", c.developer or "", flags=re.I)
    return facts.clean(name).strip(",. ")


def _figure(c: facts.Content, unit: str) -> str:
    f = next((f for f in c.figures if f.unit == unit), None)
    if not f:
        return ""
    noun = {"Acres": "acres", "Towers": "towers", "Residences": "residences", "Homes": "homes"}.get(unit, unit.lower())
    if f.value in ("1",):
        noun = noun.rstrip("s")
    return f"{f.value} {noun}"


def _themes(c: facts.Content) -> list[str]:
    order = []
    for claim in c.claims:
        if claim.topic not in order and claim.topic != "general":
            order.append(claim.topic)
    return order or ["general"]


def _short_item(text: str) -> str:
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    return text if len(text.split()) <= 4 else ""


def slots(c: facts.Content, positioning: strategy.Positioning, topics_on_page: tuple = ()) -> dict:
    cat = positioning.category
    themes = _themes(c)
    phrases = voice.THEME_PHRASE[cat]
    zone_phrases = []
    for z in c.zones:
        p = ZONE_PHRASE.get(z.topic, "")
        if p and p not in zone_phrases:
            zone_phrases.append(p)
    amen = []
    for z in c.zones:
        if not topics_on_page or z.topic in topics_on_page:
            for item in z.items:
                s = _short_item(item)
                if s and s.lower() not in (a.lower() for a in amen):
                    amen.append(s)
    amen_text = facts.prose_list(amen[:3])
    near = facts.prose_list([p.name for p in c.places[:3]])
    lo_hi = c.area_range.replace(" – ", " to ") if " – " in c.area_range else c.area_range
    return {
        "name": c.name, "short": short_name(c), "place": c.place_phrase, "locality": c.locality or c.city,
        "city": c.city, "config": c.config, "noun": c.noun, "noun_lc": c.noun.lower(),
        "developer": developer_name(c), "audience": c.audience,
        "acres": _figure(c, "Acres"), "towers": _figure(c, "Towers"),
        "units": _figure(c, "Residences") or _figure(c, "Homes"),
        "possession": c.possession, "price": c.price_from if positioning.show_price else "",
        "price_clause": f"; prices from {c.price_from}" if positioning.show_price and c.price_from else "",
        "theme": phrases.get(themes[0], ""), "theme2": phrases.get(themes[1], "") if len(themes) > 1 else "",
        "zone1": zone_phrases[0] if zone_phrases else "", "zone2": zone_phrases[1] if len(zone_phrases) > 1 else "",
        "zones": facts.prose_list(zone_phrases[:3]),
        "amenities": facts.capitalise(amen_text) if amen_text else "", "near": near,
        "area": lo_hi,
    }


def _condition_ok(cond: str, c: facts.Content) -> bool:
    kind, _, arg = cond.partition(":")
    if kind == "has":
        return any(z.topic == arg for z in c.zones)
    if kind == "kw":
        text = " ".join(cl.text for cl in c.claims).lower()
        return any(k in text for k in arg.split("|"))
    if kind == "claim":
        return any(cl.topic == arg for cl in c.claims)
    return True


def fill(frame: str, values: dict, c: facts.Content) -> Optional[str]:
    """A frame with its slots filled, or None when a slot or a condition it
    needs is missing."""
    frame = frame.strip()
    while frame.startswith("["):
        cond, _, rest = frame[1:].partition("]")
        if not _condition_ok(cond.strip(), c):
            return None
        frame = rest.strip()
    needed = re.findall(r"\{(\w+)\}", frame)
    if any(not values.get(k) for k in needed):
        return None
    out = frame
    for k in needed:
        out = out.replace("{" + k + "}", values[k])
    out = re.sub(r"\s+", " ", out).strip()
    return out[0].upper() + out[1:] if out else None


# ---------------------------------------------------------------------------
# the offline writer
# ---------------------------------------------------------------------------


class Writer:
    def __init__(self, prop, book: Book, positioning: strategy.Positioning):
        self.c = facts.build(prop, positioning.category)
        self.p = positioning
        self.cat = positioning.category
        self.guard = Guard(prop, self.c, positioning)
        self.rng = random.Random(zlib.crc32(self.c.name.encode("utf-8")))
        self.values = slots(self.c, positioning)
        self.book = book
        self.titles: set[str] = set()

    # -- helpers ------------------------------------------------------------
    def pick(self, options: list[str], values: Optional[dict] = None) -> str:
        values = values or self.values
        opts = list(options)
        start = self.rng.randrange(max(1, len(opts) - 1)) if opts else 0
        for frame in opts[start:] + opts[:start]:
            text = fill(frame, values, self.c)
            if text and self.guard.key(text) not in self.titles and self.guard.ok(text):
                self.titles.add(self.guard.key(text))
                return self.guard.keep(text)
        return ""

    def pool(self, frames: list, values: Optional[dict] = None, shuffle: bool = False,
             exclude: tuple = (), limit: int = 5) -> list[str]:
        """Sentences from frames, in order. A list inside `frames` is a set of
        alternatives: the first that fills is used. Frames that use a slot
        in `exclude` (already said elsewhere on the page) are skipped."""
        values = values or self.values
        out = []
        frames = list(frames)
        if shuffle:
            self.rng.shuffle(frames)
        for entry in frames:
            for frame in (entry if isinstance(entry, list) else [entry]):
                if any("{" + slot + "}" in frame for slot in exclude):
                    continue
                text = fill(frame, values, self.c)
                if text and self.guard.ok(text):
                    out.append(self.guard.keep(text if text.endswith((".", "?")) else text + "."))
                    break
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def used_slots(frame_text: str, values: dict) -> tuple:
        return tuple(k for k, v in values.items() if v and len(v) > 3 and v in frame_text)

    def title(self, key: str) -> str:
        return self.pick(voice.TITLES[key][self.cat])

    def kicker(self, kind: str) -> str:
        return voice.KICKERS.get(kind, {}).get(self.cat, "")

    # -- pages --------------------------------------------------------------
    def cover(self, page: BookPage) -> None:
        c = self.c
        w = page.words
        w.title = c.name
        w.kicker = c.place_phrase if self.cat != "luxury" else (c.locality or c.city)
        tag = c.tagline
        if tag and not tag.startswith("#"):
            w.lead = tag
        elif tag:
            w.note = tag
        if not w.lead:
            w.lead = self.pick(voice.OVERVIEW_TITLES[self.cat]["default"])
        page.data["config_line"] = f"{c.config} {c.noun}" if c.config else ""

    def overview(self, page: BookPage) -> None:
        c, w = self.c, page.words
        themes = _themes(c)
        bank = voice.OVERVIEW_TITLES[self.cat]
        options = next((bank[t] for t in themes if t in bank), bank["default"])
        w.kicker = self.kicker("overview")
        # a selling point's card headline is a title elsewhere in the book
        for claim in c.claims:
            self.titles.add(self.guard.key(claim.headline))
        w.title = self.pick(options) or self.pick(bank["default"])
        w.lead = self.pick(voice.OVERVIEW_LEAD[self.cat])
        sentences = self.pool(voice.OVERVIEW_POOL[self.cat],
                              exclude=self.used_slots(w.lead, self.values))
        claims = page.data.get("claims") or []
        w.cards = [Card(title=cl["headline"], text=cl["detail"], icon=cl["icon"], topic=cl["topic"])
                   for cl in claims]
        if c.description:
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", c.description) if self.guard.ok(s)][:2] + sentences
        page.pool = sentences
        page.paragraphs = 2 if self.cat != "luxury" else 1
        page.take = {"affordable": 3, "mid": 3, "luxury": 2}[self.cat]

    def highlights(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("highlights")
        w.title = self.title("highlights")
        claims = page.data.get("claims") or []
        transit = next((p for p in self.c.places if p.kind == "transit" and p.distance), None)
        w.cards = []
        for cl in claims:
            value = cl.get("figure", "")
            if not value and cl["topic"] == "connectivity" and transit is not None:
                value = transit.distance.split(" · ")[0]
            detail = cl["detail"]
            if value and detail:
                rest = re.sub(r"[^a-z ]+", " ", detail.lower().replace(re.sub(r"[^\w ]", "", value.lower()), " "))
                stripped = [w for w in rest.split() if w not in ("over", "up", "to", "of", "ensures", "spans",
                                                                    "features", "with", "and", "sq", "ft")]
                if len(stripped) < 2:
                    detail = ""
            w.cards.append(Card(title=cl["headline"], text=detail, icon=cl["icon"], topic=cl["topic"],
                                value=value))
        count = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(len(claims), "")
        lead = {"affordable": f"{count} good reasons to make {{short}} your family's next address.",
                "mid": "The ideas that shape {short}, and what they mean for daily life.",
                "luxury": "What distinguishes {short}, in brief."}[self.cat]
        if not count:
            lead = lead.replace(" good reasons", "Good reasons").strip()
        w.lead = self.pick([lead])

    def figures(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("figures")
        claims = page.data.get("claims") or []
        text = " ".join(cl["text"] for cl in claims).lower()
        options = list(voice.TITLES["figures"][self.cat])
        if "net zero" not in text:
            options = [o for o in options if "Net zero" not in o] or options
        w.title = self.pick(options)
        if claims:
            first = claims[0]
            lead = first["headline"] + (": " + facts.lower_first(first["detail"].rstrip(".")) + "."
                                        if first["detail"] else ".")
            w.lead = self.guard.keep(lead) if self.guard.ok(lead) else ""
        stat_sources = {s["source"] for s in page.data.get("stats", [])}
        w.cards = [Card(title=cl["headline"], text=cl["detail"], icon=cl["icon"])
                   for cl in claims[1:] if cl["text"] not in stat_sources or len(claims) <= 3][:3]

    def feature(self, page: BookPage) -> None:
        fam = page.topic or "project"
        topics_here = tuple(page.data.get("topics", ()))
        values = slots(self.c, self.p, topics_here) if fam == "leisure" else self.values
        w = page.words
        w.title = self.title(f"feature_{fam}")
        w.kicker = {"project": {"affordable": "The community", "mid": "Architecture", "luxury": "Architecture"},
                    "leisure": {"affordable": "Outdoors", "mid": "Leisure", "luxury": "Leisure"},
                    "home": {"affordable": "Your home", "mid": "Interiors", "luxury": "Interiors"}}[fam][self.cat]
        page.pool = self.pool(voice.FEATURE_POOL[(self.cat, fam)], values, limit=3)
        page.paragraphs = 1
        page.take = 2 if self.cat == "luxury" else 3

    def amenities(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("amenities")
        w.title = self.title("amenities")
        w.lead = self.pick(voice.AMENITIES_LEAD[self.cat])

    def gallery(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("gallery")
        w.title = self.title("gallery")
        w.lead = self.pick(voice.GALLERY_LEAD[self.cat] + ["{config} {noun_lc}."])
        page.pool = self.pool(voice.FEATURE_POOL[(self.cat, "home")], limit=2)
        page.take = 1 if self.cat != "affordable" else 2

    def masterplan(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("masterplan")
        w.title = self.title("masterplan")
        w.lead = self.pick(voice.MASTERPLAN_LINE[self.cat])
        w.note = voice.PLAN_NOTE

    def residences(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("residences")
        w.title = self.title("residences")
        w.lead = self.pick(voice.RESIDENCES_LEAD[self.cat] + ["{config} {noun_lc} in {locality}."])
        if any(u.get("price") for u in page.data.get("units", [])):
            w.note = voice.PRICE_NOTE
        offers = page.data.get("offers") or []
        w.cards = [Card(title=o["headline"], text=o["detail"], icon="tag", value=o.get("figure", ""))
                   for o in offers]

    def plans(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("plans")
        key = "plans_plates" if page.topic == "plates" else "plans_units"
        w.title = self.pick(voice.TITLES[key][self.cat]) or ("Floor plans" if page.topic == "plates" else "Layouts")
        w.note = voice.PLAN_NOTE

    def location(self, page: BookPage) -> None:
        w = page.words
        w.kicker = self.kicker("location")
        places = page.data.get("places") or []
        transit = any(p["kind"] in ("transit", "airport") for p in places)
        w.title = self.title("location_transit" if transit else "location")
        w.lead = self.pick(voice.LOCATION_LEAD[self.cat])
        for cl in page.data.get("claims") or []:
            w.cards.append(Card(title=cl["headline"], text=cl["detail"], icon=cl["icon"], topic=cl["topic"]))

    def closing(self, page: BookPage) -> None:
        c, w = self.c, page.words
        w.kicker = self.kicker("closing")
        cta = c.cta
        if cta and len(cta) <= 72 and self.guard.ok(cta):
            w.title = self.guard.keep(facts.capitalise(cta.rstrip(".")))
            w.lead = self.pick(voice.CLOSING_LINE[self.cat])
        else:
            w.title = self.title("closing")
            w.lead = self.guard.keep(facts.sentence(cta)) if cta and self.guard.ok(cta) else \
                self.pick(voice.CLOSING_LINE[self.cat])
        w.cta = "Call" if c.phone else ""

    def write(self, page: BookPage) -> None:
        getattr(self, page.kind)(page)
        w = page.words
        if w.kicker and self.guard.key(w.kicker) == self.guard.key(w.title):
            w.kicker = ""
        page.words.source = page.words.source or "offline"
        page.set_body()


# ---------------------------------------------------------------------------
# the model writer
# ---------------------------------------------------------------------------


def _page_brief(page: BookPage) -> dict:
    wants = {"title": "a page title of 2-7 words"}
    if page.kind in ("overview", "feature", "gallery"):
        wants["sentences"] = "4-6 short sentences of running copy, in order of importance"
    if page.kind in ("overview", "amenities", "location", "highlights", "figures", "residences"):
        wants["lead"] = "one sentence, at most 160 characters"
    if page.kind in ("highlights",):
        wants["cards"] = "for each selling point given, a 2-5 word headline and a one-sentence detail"
    return {"page": page.number, "kind": page.kind, "subject": page.topic or page.kind,
            "content_on_page": {k: v for k, v in page.data.items()
                                if k in ("claims", "zones", "places", "stats", "units", "topics")},
            "write": wants}


def _model_copy(book: Book, writer: Writer) -> dict[int, dict]:
    pages = [p for p in book.pages if p.kind not in ("cover", "plans", "closing", "masterplan")]
    try:
        template = load_yaml_config("brochure/prompts.yaml").get("book_copy")
        if not template:
            return {}
        c = writer.c
        fact_sheet = {
            "name": c.name, "developer": c.developer, "location": c.place_phrase, "address": c.address,
            "configurations": c.config, "units": [u.__dict__ for u in c.units],
            "price": c.price if writer.p.show_price else "(do not mention price)",
            "project_size": [f.__dict__ for f in c.figures], "possession": c.possession,
            "selling_points": [cl.text for cl in c.claims], "amenities": c.amenity_lines,
            "target_audience": c.audience_full, "tagline": c.tagline, "call_to_action": c.cta,
            "nearby": [p.__dict__ for p in c.places],
        }
        prompt = Template(template).substitute(
            category=writer.p.label, promise=writer.p.promise, pillars=", ".join(writer.p.pillars),
            voice=writer.p.voice, avoid=", ".join(writer.p.avoid),
            facts=json.dumps(fact_sheet, ensure_ascii=False, indent=1, default=str),
            pages=json.dumps([_page_brief(p) for p in pages], ensure_ascii=False, indent=1, default=str))
        result = with_retries(lambda: call_with_fallback(ModelRole.COPYWRITER, "generate_json", prompt),
                              max_retries=1)
    except Exception as exc:  # noqa: BLE001 - the offline writer writes the book instead
        logger.info("Copy model unavailable (%s); writing offline", type(exc).__name__)
        return {}
    items = result.get("pages") if isinstance(result, dict) else result
    out = {}
    for item in items or []:
        try:
            out[int(item.get("page"))] = item
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _apply_model(page: BookPage, item: dict, writer: Writer) -> None:
    """Take each field the model wrote that passes the guard; keep the
    offline writer's for the rest."""
    g = writer.guard
    w = page.words
    title = facts.clean(str(item.get("title") or "")).rstrip(".")
    if title and len(title) <= 60 and g.ok(title) and g.key(title) not in writer.titles:
        writer.titles.add(g.key(title))
        w.title = g.keep(title)
        w.source = "model"
    lead = facts.clean(str(item.get("lead") or ""))
    if lead and len(lead) <= 200 and g.ok(lead):
        w.lead = g.keep(lead)
        w.source = "model"
    sentences = item.get("sentences") or []
    if isinstance(sentences, str):
        sentences = re.split(r"(?<=[.!?])\s+", sentences)
    good = [g.keep(facts.sentence(s)) for s in (facts.clean(str(x)) for x in sentences)
            if s and len(s) <= 260 and g.ok(facts.sentence(s))]
    if len(good) >= 2:
        page.pool = good
        w.source = "model"
    cards = item.get("cards") or []
    if page.kind == "highlights" and isinstance(cards, list) and len(cards) == len(w.cards):
        for card, new in zip(w.cards, cards):
            if not isinstance(new, dict):
                continue
            head = facts.clean(str(new.get("headline") or new.get("title") or "")).rstrip(".")
            detail = facts.sentence(str(new.get("detail") or new.get("text") or ""))
            if head and len(head) <= 48 and g.ok(head):
                card.title = head
            if detail and len(detail) <= 180 and g.numbers_ok(detail) and g.register_ok(detail):
                card.text = detail


# ---------------------------------------------------------------------------


def write(book: Book, prop, positioning: Optional[strategy.Positioning] = None,
          seed: Optional[int] = None, use_model: bool = True) -> Book:
    """Every page's words."""
    positioning = positioning or strategy.for_category(book.category)
    writer = Writer(prop, book, positioning)
    if seed:
        writer.rng = random.Random(zlib.crc32(writer.c.name.encode("utf-8")) + seed)
    for page in book.pages:
        page.words = Copy()
        writer.write(page)
    if use_model:
        for number, item in _model_copy(book, writer).items():
            page = next((p for p in book.pages if p.number == number), None)
            if page is not None:
                _apply_model(page, item, writer)
                page.set_body()
    return book
