"""Research Agent — find out about the place before writing about it.

THE PROBLEM THIS SOLVES

Ask a language model to write about "Shela, Ahmedabad" and it will
produce confident paragraphs about schools, metro lines and travel times.
It has no idea whether any of them are there. The prose is fine; the
facts are furniture. That is the single biggest gap between an LLM blog
and one a local reader would trust, and no amount of better prompting
closes it, because the model is not lying — it does not know.

So this step goes and looks. OpenStreetMap for what is actually within a
few kilometres, a router for the journeys people care about, Wikipedia
for what the area is, news for what is changing, search for what everyone
else has already written. All of it is written down as a ResearchPack,
and the writer is given that instead of its own imagination.

WHAT THIS AGENT ACTUALLY DECIDES

Very little, and on purpose. The model's whole job here is to read the
topic and say WHAT to look up — which place, which searches, which
journeys matter. The tools do the looking. That division is what keeps
the research checkable: every fact in the pack came from a named source
with a URL, not from the model that will later write the post.

If the model call fails, a deterministic reading of the topic takes over
and the research still runs. If the tools fail, the pack comes back thin
with notes saying why, and the post is written the way it always was.
Nothing here can stop a blog being produced.
"""
from __future__ import annotations

import logging
import re
from string import Template
from typing import Optional

from src.agents.blog import archetypes, blog_config, brief, llm
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import BlogInput, JourneyTime, NearbyPlace, NewsItem, ResearchPack
from src.models.gateway import call_with_fallback
from src.tools import ToolBox, toolbox_for

logger = logging.getLogger(__name__)

# How far out to look for amenities. Three kilometres is what "nearby"
# means to someone choosing where to live; ten would return the whole city
# and prove nothing about the address.
NEARBY_RADIUS_M = 3000
MAX_NEARBY = 14
MAX_NEWS = 6
MAX_COMPETITOR_TOPICS = 10

# Journeys worth measuring properly. Deliberately short: each is a call to
# a free router that asks not to be hammered, and "18 km from the airport"
# is a sentence a reader acts on, while "4.1 km from the third-nearest
# pharmacy" is not.
JOURNEY_TARGETS = ("international airport", "railway station", "city centre")


def run(blog_input: BlogInput, box: Optional[ToolBox] = None) -> ResearchPack:
    """Everything findable about this topic, as a structure."""
    box = box if box is not None else toolbox_for("research")
    plan = _plan(blog_input)
    archetype = archetypes.classify(blog_input)
    pack = ResearchPack(subject=plan["subject"], place_query=plan["place"], archetype=archetype)

    if not box.enabled:
        pack.notes.append("External research is turned off, so this post was written "
                          "from the brief alone.")
        return pack

    # Research depth follows the archetype: a legal explainer does not need
    # the nearest pharmacy, and a locality guide is nothing without it.
    passes = set(archetypes.catalogue().get(archetype, {}).get("research")
                 or ["location", "background", "news", "official", "competitors"])
    steps = [("location", lambda: _locate(pack, box, plan)), ("background", lambda: _read_up(pack, box, plan)),
             ("news", lambda: _find_news(pack, box, plan)),
             ("official", lambda: _find_official(pack, box, plan, blog_input)),
             ("competitors", lambda: _see_what_exists(pack, box, plan))]
    attempted = 0
    for name, step in steps:
        if name not in passes:
            continue
        step()
        attempted += 1
        # Two passes in, nothing found and every failure was the network
        # itself: the rest would only add timeouts, not facts.
        if attempted >= 2 and pack.is_empty and _network_down(box.notes):
            box.notes.append(NETWORK_DOWN_NOTE)
            break

    # Only the notes worth a reader's attention; the rest is in the log.
    pack.notes = [_tidy(n) for n in dict.fromkeys(box.notes) if n]
    pack.notes = list(dict.fromkeys(pack.notes))
    if pack.is_empty and not pack.notes:
        pack.notes.append("No external research was available for this topic.")
    return pack


# ---------------------------------------------------------------------------
# What to look up
# ---------------------------------------------------------------------------

def _plan(blog_input: BlogInput) -> dict:
    """Read the topic and decide what to research.

    An LLM is good at this and a regex is not: "Is Shela a good bet for
    first-time buyers?" contains a place, a subject and an audience, and
    pulling the place out of it by pattern matching gets "Is Shela a good"
    about as often as it gets "Shela"."""
    prompts = load_yaml_config("blog/prompts.yaml")
    template = prompts.get("research_plan")
    if template:
        try:
            prompt = Template(template).substitute(
                topic=blog_input.topic,
                requested_sections=brief.requested_sections_block(blog_input),
                local_facts=brief.local_facts_block(blog_input),
            )
            result = llm.generate_json(ModelRole.BLOG_RESEARCH, prompt, call_with_fallback)
            plan = {
                "subject": str(result.get("subject") or blog_input.topic).strip(),
                "place": str(result.get("place") or "").strip(),
                "country_code": str(result.get("country_code") or "").strip().lower()[:2],
                "searches": [str(q).strip() for q in (result.get("searches") or []) if str(q).strip()],
                "news_query": str(result.get("news_query") or "").strip(),
            }
            if plan["place"]:
                return plan
        except Exception as exc:  # noqa: BLE001 - research must never block a post
            logger.warning("Research planning LLM call failed; using the topic as written: %s", exc)

    return _plan_without_a_model(blog_input)


# "in Shela, Ahmedabad", "at Whitefield", "near Sector 128 Noida"
_PLACE_HINT = re.compile(r"\b(?:in|at|near|around|across)\s+([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3})")


def _plan_without_a_model(blog_input: BlogInput) -> dict:
    """The fallback: the place the brief names (or the topic implies), as
    the shared brief helper reads it, so research and writing agree."""
    topic = blog_input.topic.strip()
    loc = brief.location(blog_input)
    if loc.known:
        place = loc.full + (", India" if loc.india and "india" not in loc.full.lower() else "")
        return {"subject": topic, "place": place, "country_code": "in" if loc.india else "",
                "searches": [brief.primary_keyword(blog_input, loc) or topic], "news_query": ""}
    match = _PLACE_HINT.search(topic)
    place = match.group(1).strip(" ,.") if match else ""
    if not place:
        # Capitalised words that are not the first word are usually the
        # place in a title like "5 reasons to buy in Shela".
        capitals = re.findall(r"\b[A-Z][a-z]{2,}\b", topic)
        place = capitals[-1] if capitals else ""
    return {"subject": topic, "place": place, "country_code": "",
            "searches": [topic], "news_query": ""}


# ---------------------------------------------------------------------------
# The four research passes
# ---------------------------------------------------------------------------

def _locate(pack: ResearchPack, box: ToolBox, plan: dict) -> None:
    """Where the place is, what is around it, and how far the journeys are."""
    if not plan["place"]:
        pack.notes.append("No location could be identified in the topic, so this post has "
                          "no local detail.")
        return

    found = box.geocode(plan["place"], country_code=plan.get("country_code", ""))
    if not found.ok or not found.data:
        return
    pack.place_name = found.data.get("display_name", "")
    pack.latitude = float(found.data.get("lat") or 0)
    pack.longitude = float(found.data.get("lon") or 0)
    pack.located = bool(pack.latitude or pack.longitude)
    if not pack.located:
        return

    nearby = box.nearby_places(lat=pack.latitude, lon=pack.longitude, radius_m=NEARBY_RADIUS_M)
    if nearby.ok:
        pack.nearby = [
            NearbyPlace(name=p["name"], kind=p.get("kind", ""),
                        distance_km=round(p["distance_m"] / 1000.0, 1))
            for p in nearby.data[:MAX_NEARBY]
        ]

    city = plan["place"].split(",")[-1].strip() or plan["place"]
    for target in JOURNEY_TARGETS:
        destination = box.geocode(f"{target} {city}")
        if not destination.ok or not destination.data:
            continue
        route = box.travel_distance(
            from_lat=pack.latitude, from_lon=pack.longitude,
            to_lat=float(destination.data.get("lat") or 0),
            to_lon=float(destination.data.get("lon") or 0),
            label=target)
        if route.ok and route.data:
            pack.journeys.append(JourneyTime(
                destination=target,
                distance_km=float(route.data.get("distance_km") or 0),
                free_flow_minutes=int(route.data.get("free_flow_minutes") or 0)))


def _read_up(pack: ResearchPack, box: ToolBox, plan: dict) -> None:
    """Neutral background on the area."""
    query = plan["place"] or plan["subject"]
    if not query:
        return
    found = box.encyclopedia(query)
    if found.ok and found.data:
        pack.background = found.data.get("summary", "")
        pack.background_source = found.data.get("url", "")
        if pack.background_source:
            pack.sources.append(pack.background_source)


def _find_news(pack: ResearchPack, box: ToolBox, plan: dict) -> None:
    """What is changing around the place right now."""
    from src.tools.news import news_query

    query = plan.get("news_query") or news_query(plan["place"] or plan["subject"])
    if not query:
        return
    found = box.search_news(query, limit=MAX_NEWS)
    if not found.ok:
        return
    for story in found.data[:MAX_NEWS]:
        pack.news.append(NewsItem(title=story.get("title", ""), source=story.get("source", ""),
                                  published=story.get("published", ""), url=story.get("url", "")))
        if story.get("url"):
            pack.sources.append(story["url"])


NETWORK_DOWN_NOTE = "The research sources could not be reached, so the remaining lookups were skipped."
_NETWORK_ERRORS = re.compile(r"proxy|connection|timed? ?out|unreachable|max retries|name resolution|"
                             r"tunnel|network|ssl|403 forbidden", re.I)


def _network_down(notes: list[str]) -> bool:
    return bool(notes) and all(_NETWORK_ERRORS.search(n or "") for n in notes)


def _tidy(note: str) -> str:
    """A tool failure, said plainly — not a stack of proxy errors."""
    note = " ".join(str(note).split())
    cut = re.search(r":\s*(?:https?://|\w+Error|RequestError)", note)
    if cut:
        note = note[:cut.start()].rstrip(" :") + " (source unreachable)"
    return note[:160]


def is_authoritative(url: str) -> bool:
    host = re.sub(r"^https?://(?:www\.)?", "", (url or "").lower()).split("/")[0]
    return any(marker in host for marker in blog_config.editorial().get("authority_domains", []))


def _find_official(pack: ResearchPack, box: ToolBox, plan: dict, blog_input: BlogInput) -> None:
    """Primary sources first: government, RERA, metro and municipal pages.

    Only results on an authoritative domain are kept, with their snippet —
    these, unlike competitors' pages, are treated as evidence (the fact
    check accepts their figures)."""
    from src.data.schemas import BlogReference

    loc = brief.location(blog_input)
    subject = brief.subject(blog_input)
    queries = []
    if loc.india:
        queries.append(f"{loc.state or loc.city} RERA {subject}".strip())
        if loc.known:
            queries.append(f"{loc.short} {loc.city} metro municipal development plan".strip())
    else:
        queries.append(f"{subject} official")
    for query in queries[:2]:
        found = box.search_web(query, limit=6)
        if not found.ok:
            continue
        for result in found.data or []:
            url = (result.get("url") or "").strip()
            if not url or not is_authoritative(url) or any(r.url == url for r in pack.official_sources):
                continue
            pack.official_sources.append(BlogReference(
                title=(result.get("title") or url)[:120], url=url, kind="official",
                note=" ".join((result.get("snippet") or "").split())[:300]))
            pack.sources.append(url)
        if len(pack.official_sources) >= 4:
            break


def _see_what_exists(pack: ResearchPack, box: ToolBox, plan: dict) -> None:
    """What everyone else writing about this already covers.

    Not source material — these are competitors' pages and nobody has
    checked them. It is the SHAPE that is useful: the subtopics every
    ranking page covers are the ones a reader expects, and the one none of
    them covers is the reason to read this post instead."""
    queries = plan.get("searches") or [plan["subject"]]
    for query in queries[:2]:
        found = box.search_web(query, limit=6)
        if not found.ok:
            continue
        for result in found.data:
            title = (result.get("title") or "").strip()
            if title and title not in pack.competitor_topics:
                pack.competitor_topics.append(title)
        if len(pack.competitor_topics) >= MAX_COMPETITOR_TOPICS:
            break
    pack.competitor_topics = pack.competitor_topics[:MAX_COMPETITOR_TOPICS]


# ---------------------------------------------------------------------------
# Rendering the pack for a prompt
# ---------------------------------------------------------------------------

def as_prompt_block(pack: ResearchPack) -> str:
    """The pack as the writer sees it.

    Every figure carries its provenance in the line itself — "measured
    straight-line", "free-flow, no traffic" — because a number handed to a
    model without its caveat comes back out as a bare assertion, and
    "35 minutes to the airport" is a very different promise from "about 35
    minutes in clear traffic"."""
    if pack.is_empty:
        return ("(no external research was available — do not state any local facts, "
                "distances, or current events that the author did not provide)")

    lines: list[str] = []
    if pack.place_name:
        lines.append(f"LOCATION: {pack.place_name}")
    if pack.background:
        lines.append(f"\nBACKGROUND (Wikipedia, neutral — paraphrase, do not quote):\n{pack.background}")
    if pack.nearby:
        lines.append("\nNEARBY PLACES (OpenStreetMap; straight-line distance, not driving):")
        lines.extend(f"- {p.name} ({p.kind}) — about {p.distance_km} km" for p in pack.nearby)
    if pack.journeys:
        lines.append("\nROAD JOURNEYS (free-flow, i.e. clear traffic — never present these "
                     "as a rush-hour commute):")
        lines.extend(f"- {j.destination}: {j.distance_km} km, about {j.free_flow_minutes} minutes"
                     for j in pack.journeys)
    if pack.news:
        lines.append("\nRECENT NEWS (headlines only — evidence that something was ANNOUNCED, "
                     "not that it has happened; attribute anything you use):")
        lines.extend(f"- \"{n.title}\" — {n.source}{', ' + n.published if n.published else ''}"
                     for n in pack.news)
    if pack.official_sources:
        lines.append("\nOFFICIAL SOURCES (government / regulator pages — the most authoritative "
                     "evidence here; cite by name):")
        lines.extend(f"- {r.title} ({r.url}): {r.note}" for r in pack.official_sources)
    if pack.competitor_topics:
        lines.append("\nWHAT OTHER ARTICLES ON THIS TOPIC COVER (for finding the angle they "
                     "miss — NOT facts, and not to be repeated as true):")
        lines.extend(f"- {t}" for t in pack.competitor_topics)
    if pack.reader_questions:
        lines.append("\nQUESTIONS REAL READERS SEARCH FOR (good candidates for headings):")
        lines.extend(f"- {q}" for q in pack.reader_questions)
    if pack.notes:
        lines.append("\nWHAT COULD NOT BE RESEARCHED (say less rather than filling these gaps):")
        lines.extend(f"- {n}" for n in pack.notes)
    return "\n".join(lines)
