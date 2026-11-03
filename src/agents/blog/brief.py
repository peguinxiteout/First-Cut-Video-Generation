"""Formats the parts of a BlogInput that several agents need, so the
planner, writer, editor, SEO agent and critic all see exactly the same
brief — and derives what the brief leaves implicit (the place, the
audience, the primary keyword, the options being compared) ONCE, in one
place, instead of each agent guessing differently."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.agents.blog import blog_config
from src.data.schemas import BlogInput, BlogReference


def section_catalog() -> dict[str, dict]:
    return blog_config.sections()


def requested_sections_block(blog_input: BlogInput) -> str:
    """Every section the person asked for, catalogue ones and their own.

    Custom sections are marked as the author's own words and put last, so
    the model treats them as a requirement it was given rather than as one
    more preset it may merge away. They were being dropped when they read
    as a paraphrase of a catalogue entry."""
    catalog = section_catalog()
    lines = []
    for sid in blog_input.sections:
        entry = catalog.get(sid)
        if entry:
            lines.append(f"- {sid} ({entry['label']}): {entry['guidance']}")
    for text in custom_sections(blog_input):
        lines.append(f'- custom (the author asked for this specifically): "{text}" — '
                     "cover this in its own section, in the author's sense of it.")
    if not lines:
        return "- none specified: choose the structure that best serves this topic"
    return "\n".join(lines)


def custom_sections(blog_input: BlogInput) -> list[str]:
    """The person's own section descriptions, cleaned and de-duplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in getattr(blog_input, "custom_sections", []) or []:
        text = " ".join(str(raw).split()).strip(" -•")
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def clean_facts(blog_input: BlogInput) -> list[str]:
    return [" ".join(f.split()) for f in blog_input.local_facts if f and f.strip()]


def local_facts_block(blog_input: BlogInput) -> str:
    facts = clean_facts(blog_input)
    if not facts:
        return ("(none provided — so do NOT state any specific numbers, prices, "
                "percentages, distances, dates, or named projects)")
    return "\n".join(f"- {f}" for f in facts)


def key_points_block(blog_input: BlogInput) -> str:
    return "\n".join(f"- {p}" for p in blog_input.key_points if p.strip()) or "(none)"


def tone(blog_input: BlogInput) -> str:
    return blog_input.tone or "clear, authoritative, and advisory"


# ---------------------------------------------------------------------------
# Contact details — real ones only
# ---------------------------------------------------------------------------

_URL = re.compile(r"^https?://[^\s/]+\.[a-z]{2,}(?:[/?#]\S*)?$", re.I)
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")


def _is_placeholder(text: str) -> bool:
    from src.agents.blog.normaliser import find_placeholders

    return bool(find_placeholders(text))


def contact_details(blog_input: BlogInput) -> dict[str, str]:
    """The contact details it is safe to print: supplied in the brief AND
    not a placeholder. A form that defaults to "https://..." or "+91 XXXXX"
    must produce a CTA without contacts, never a broken one."""
    url = (blog_input.cta_url or "").strip()
    phone = (blog_input.cta_phone or "").strip()
    email = (getattr(blog_input, "cta_email", "") or "").strip()
    out = {"url": "", "phone": "", "email": ""}
    if url and _URL.match(url) and not _is_placeholder(url):
        out["url"] = url
    if phone and len(re.sub(r"\D", "", phone)) >= 8 and not re.search(r"[xX]", phone) \
            and not _is_placeholder(phone) and len(set(re.sub(r"\D", "", phone))) > 2:
        out["phone"] = phone
    if email and _EMAIL.match(email) and not _is_placeholder(email):
        out["email"] = email
    return out


def cta_instruction(blog_input: BlogInput) -> str:
    parts = []
    if blog_input.cta_text:
        parts.append(f"The author's intended action: {blog_input.cta_text}.")
    contacts = contact_details(blog_input)
    if any(contacts.values()):
        parts.append("Contact details will be shown beside your text, so do not repeat them.")
    else:
        parts.append("No contact details were provided, so invite a sensible next step "
                     "(e.g. arranging a site visit) without inventing a phone number, email, or URL.")
    return " ".join(parts)


def reference_links(blog_input: BlogInput) -> list[BlogReference]:
    """Author-supplied references ("Title | https://..." or a bare URL)."""
    out: list[BlogReference] = []
    for raw in getattr(blog_input, "reference_links", []) or []:
        text = str(raw).strip()
        if not text:
            continue
        title, _, url = text.rpartition("|") if "|" in text else ("", "", text)
        url = url.strip()
        if not _URL.match(url) or _is_placeholder(url):
            continue
        host = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
        out.append(BlogReference(title=title.strip() or host, url=url, kind="brief",
                                 note="Supplied with the brief"))
    return out


def author(blog_input: BlogInput) -> str:
    return (getattr(blog_input, "author_name", "") or getattr(blog_input, "brand_name", "")
            or "Editorial team").strip()


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

@dataclass
class Location:
    full: str = ""      # "Shela, Ahmedabad"
    short: str = ""     # "Shela"
    city: str = ""      # "Ahmedabad"
    state: str = ""     # "Gujarat"
    india: bool = False

    @property
    def known(self) -> bool:
        return bool(self.short)

    def names(self) -> list[str]:
        return [n for n in dict.fromkeys([self.short, self.city, self.state, self.full]) if n]


# Capitalised words that open a title but are not places.
_NOT_PLACES = frozenset("""is are why how what should can buying buy guide living investing invest
the a an best top homes home flats flat property properties real estate vs your first my our
complete ultimate where when which does do will tips things reasons new luxury affordable
apartments apartment villas villa plots plot rera explained understanding comparing compare in
at near around for of and or to""".split())

_PLACE_HINT = re.compile(r"\b(?:in|at|near|around|across)\s+([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3})")


def _india_signal(blog_input: BlogInput) -> bool:
    text = " ".join([blog_input.topic, blog_input.location or "", *blog_input.local_facts,
                     *blog_input.key_points]).lower()
    return any(marker in text for marker in blog_config.editorial().get("india_markers", []))


def location(blog_input: BlogInput) -> Location:
    cities: dict[str, str] = blog_config.editorial().get("cities", {}) or {}
    given = " ".join((blog_input.location or "").split()).strip(" ,")
    text = given or blog_input.topic
    lowered = text.lower()

    city_key, city_pos = "", -1
    for key in sorted(cities, key=len, reverse=True):
        match = re.search(r"\b" + re.escape(key) + r"\b", lowered)
        if match:
            city_key, city_pos = key, match.start()
            break
    city = text[city_pos:city_pos + len(city_key)] if city_key else ""
    city = city[:1].upper() + city[1:] if city else ""
    state = cities.get(city_key, "") if city_key else ""

    if given:
        parts = [p.strip() for p in given.split(",") if p.strip()]
        short = parts[0] if parts else given
        if not city and len(parts) > 1:
            city = parts[1]
        full = given
    elif city_key:
        before = text[:city_pos].rstrip(" ,")
        tokens = re.findall(r"[A-Z][\w'.-]*", before.split(":")[-1].split("?")[-1])
        locality: list[str] = []
        for token in reversed(before.split()[-4:]):
            word = token.strip(",.")
            if word and word[0].isupper() and word.lower() not in _NOT_PLACES and word in tokens:
                locality.insert(0, word)
            else:
                break
        after = re.match(r"\s*,?\s*((?:Sector|Phase|Block)\s+\d+\w*|[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?)",
                         text[city_pos + len(city_key):])
        if not locality and after and after.group(1).split()[0].lower() not in _NOT_PLACES:
            locality = [after.group(1)]
        short = " ".join(locality) or city
        if comparison_options(blog_input):
            short = city     # a comparison is about the options, set in the city
        full = f"{short}, {city}" if short != city else city
    else:
        match = _PLACE_HINT.search(blog_input.topic)
        short = match.group(1).strip(" ,.") if match else ""
        short = " ".join(w for w in short.split() if w.lower() not in _NOT_PLACES)
        full = short
    india = bool(state) or _india_signal(blog_input)
    if city_key and cities.get(city_key, None) == "":
        india = _india_signal(blog_input)
    return Location(full=full, short=short, city=city, state=state, india=india)


# ---------------------------------------------------------------------------
# Audience
# ---------------------------------------------------------------------------

def audience_profile(blog_input: BlogInput) -> dict:
    """The reader profile (configs/blog/archetypes.yaml `audiences`). The
    brief's own wording wins for the label; the profile supplies what
    matters to that reader."""
    profiles: dict = blog_config.archetypes().get("audiences", {})
    stated = (getattr(blog_input, "audience", "") or "").strip()
    for source in (stated, blog_input.topic, " ".join(blog_input.key_points)):
        if not source:
            continue
        for pid, profile in profiles.items():
            if pid == "default":
                continue
            if any(re.search(p, source, re.I) for p in profile.get("match", [])):
                return {**profile, "id": pid, "label": stated or profile["label"]}
    default = profiles.get("default", {"label": "homebuyers", "priorities": [], "why": {}, "concern": ""})
    return {**default, "id": "default", "label": stated or default["label"]}


# ---------------------------------------------------------------------------
# Subject, options, keywords
# ---------------------------------------------------------------------------

_LEAD_WORDS = re.compile(
    r"^(?:(?:is|are|should|can|could|what|how|why|which|does|do|will|would|a|an|the|your|you|i|we|"
    r"understanding|explained|guide to|a guide to|complete guide to|everything about)\s+)+", re.I)


def gerund(verb: str) -> str:
    """register -> registering, apply -> applying, get -> getting."""
    v = verb.lower()
    if v.endswith("ie"):
        return v[:-2] + "ying"
    if v.endswith("e") and not v.endswith(("ee", "ye", "oe")):
        return v[:-1] + "ing"
    if len(v) <= 4 and re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", v):
        return v + v[-1] + "ing"
    return v + "ing"


def subject(blog_input: BlogInput) -> str:
    """What the post is about, as a noun phrase: "What is RERA carpet
    area?" -> "RERA carpet area"; "How to register a flat" ->
    "Registering a flat"."""
    text = re.split(r"[:?\u2014]| - ", blog_input.topic.strip())[0].strip()
    how_to = re.match(r"^how\s+to\s+(\w+)\s*(.*)$", text, re.I)
    if how_to:
        phrase = f"{gerund(how_to.group(1))} {how_to.group(2)}".strip()
        return phrase[:1].upper() + phrase[1:]
    text = re.sub(r"^(?:what\s+(?:is|are)|how\s+(?:does|do|to)|understanding|a\s+guide\s+to)\s+",
                  "", text, flags=re.I)
    text = re.sub(r"\s+explained$", "", text, flags=re.I).strip(" ,.")
    text = _LEAD_WORDS.sub("", text).strip(" ,.") or blog_input.topic.strip()
    return text[:1].upper() + text[1:]


def comparison_options(blog_input: BlogInput) -> list[str]:
    """The options a comparison post weighs, e.g. ["Whitefield", "Sarjapur
    Road"]. Empty when the topic is not a comparison."""
    head = re.split(r"[:?—]", blog_input.topic)[0]
    parts = re.split(r"\s+(?:vs\.?|versus)\s+", head, flags=re.I)
    if len(parts) < 2 and re.search(r"\b(which|better|should)\b", blog_input.topic, re.I):
        parts = re.split(r"\s+or\s+", head, flags=re.I)
    if len(parts) < 2:
        return []
    options = []
    for i, part in enumerate(parts[:3]):
        words = part.strip(" ,.").split()
        if i == 0:
            words = _LEAD_WORDS.sub("", " ".join(words)).split()
            words = [w for w in words if w.lower() not in {"buying", "buy", "choosing", "between"}]
            words = words[-4:]
        else:
            cut = []
            for w in words:
                if w.lower() in {"for", "in", "which", "when", "with", "a", "the", "is", "are"} and cut:
                    break
                cut.append(w.strip(","))
            words = cut[:4]
        option = " ".join(words).strip(" ,.")
        for city_name in blog_config.editorial().get("cities", {}):
            option = re.sub(r"[\s,]+" + re.escape(city_name) + r"$", "", option, flags=re.I)
        if option:
            options.append(option[:1].upper() + option[1:])
    return options if len(options) >= 2 else []


_PROPERTY_NOUNS = [("apartments", r"apartments?"), ("flats", r"flats?"), ("villas", r"villas?"),
                   ("plots", r"plots?"), ("homes", r"homes?|house"), ("property", r"propert(?:y|ies)|real estate")]


def primary_keyword(blog_input: BlogInput, loc: Optional[Location] = None) -> str:
    """The brief's keyword, or a searchable phrase derived from the topic."""
    stated = (getattr(blog_input, "primary_keyword", "") or "").strip()
    if stated:
        return stated
    options = comparison_options(blog_input)
    if options:
        return " vs ".join(options[:2]).lower()
    loc = loc or location(blog_input)
    topic = blog_input.topic
    if loc.known:
        noun = next((name for name, pat in _PROPERTY_NOUNS if re.search(r"\b(?:" + pat + r")\b", topic, re.I)),
                    "property")
        place = loc.short if not loc.city or loc.city == loc.short else f"{loc.short} {loc.city}"
        if re.search(r"invest", topic, re.I):
            return f"investing in {place}".lower()
        return f"{noun} in {place}".lower()
    text = subject(blog_input).lower()
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)
    text = re.sub(r"\b(?:a good|good|really|worth it|complete|ultimate|everything you need to know)\b", "", text)
    text = re.sub(r"^\d+\s+", "", text.strip())
    words = [w for w in re.findall(r"[\w'-]+", text)][:6]
    return " ".join(words)


def secondary_keywords(blog_input: BlogInput, archetype: str, loc: Optional[Location] = None) -> list[str]:
    stated = [k.strip() for k in getattr(blog_input, "secondary_keywords", []) or [] if k.strip()]
    loc = loc or location(blog_input)
    derived: list[str] = []
    patterns = blog_config.archetypes().get("archetypes", {}).get(archetype, {}).get("keyword_patterns", [])
    for pattern in patterns:
        if "{place}" in pattern and not loc.known:
            continue
        if "{city}" in pattern and not loc.city:
            continue
        derived.append(pattern.format(place=loc.short, city=loc.city))
    primary = primary_keyword(blog_input, loc).lower()
    out = []
    for kw in stated + derived:
        if kw.lower() != primary and kw.lower() not in (o.lower() for o in out):
            out.append(kw)
    return out[:6]


@dataclass
class BriefContext:
    """Everything derived from the brief, computed once per run."""
    location: Location
    audience: dict
    primary_keyword: str
    options: list[str] = field(default_factory=list)
    subject: str = ""
    contacts: dict = field(default_factory=dict)

    topic: str = ""

    @property
    def audience_label(self) -> str:
        return self.audience.get("label", "homebuyers")

    @property
    def keyword_display(self) -> str:
        """The keyword as prose: place names keep their capitals ("homes in
        Shela Ahmedabad", not "homes in shela ahmedabad")."""
        return display_case(prose_keyword(self.primary_keyword),
                            " ".join([self.topic, self.location.full, self.location.city, *self.options]))


_VERB_LED = re.compile(r"^(buy|rent|invest|sell|book|choose|find|lease|own)\s+(?:(a|an|the|your)\s+)?(\w+)(.*)$", re.I)


def prose_keyword(phrase: str) -> str:
    """A search phrase as it reads in a sentence: "buy home in Kokapet" ->
    "buying a home in Kokapet". Search phrases drop their articles; prose
    does not, and "if you are weighing buy home in Kokapet" is not English."""
    m = _VERB_LED.match((phrase or "").strip())
    if not m:
        return phrase
    verb, article, noun, rest = m.groups()
    if not article:
        article = "a" if noun.lower() in ("home", "house", "flat", "villa", "plot", "property",
                                          "apartment", "unit") else ""
    return " ".join(x for x in (gerund(verb), article, noun) if x) + rest


def display_case(phrase: str, reference: str) -> str:
    """Restore the capitals a word has in `reference` (place names, RERA)."""
    cased = {}
    for word in re.findall(r"[\w'-]+", reference):
        if any(ch.isupper() for ch in word):
            cased.setdefault(word.lower(), word)
    return re.sub(r"[\w'-]+", lambda m: cased.get(m.group(0).lower(), m.group(0)), phrase)


def context(blog_input: BlogInput) -> BriefContext:
    loc = location(blog_input)
    return BriefContext(location=loc, audience=audience_profile(blog_input),
                        primary_keyword=primary_keyword(blog_input, loc),
                        options=comparison_options(blog_input), subject=subject(blog_input),
                        contacts=contact_details(blog_input), topic=blog_input.topic)


def brief_profile_block(blog_input: BlogInput) -> str:
    """The derived brief, for prompts: who it is for, where, what to rank
    for, and whose voice — so the model is not left to guess them."""
    ctx = context(blog_input)
    lines = [f"Audience: {ctx.audience_label} — their main concern: {ctx.audience.get('concern', '')}"]
    if ctx.location.known:
        lines.append(f"Location: {ctx.location.full}"
                     + (f" ({ctx.location.state})" if ctx.location.state else ""))
    lines.append(f"Primary keyword: {ctx.primary_keyword}")
    if blog_input.secondary_keywords:
        lines.append("Secondary keywords: " + ", ".join(blog_input.secondary_keywords))
    if blog_input.brand_name:
        lines.append(f"Publisher/brand: {blog_input.brand_name}"
                     + (f" — {blog_input.brand_description}" if blog_input.brand_description else ""))
    if blog_input.project_name:
        lines.append(f"Project that may be named: {blog_input.project_name}")
    if ctx.options:
        lines.append("Options being compared: " + " vs ".join(ctx.options))
    return "\n".join(lines)
