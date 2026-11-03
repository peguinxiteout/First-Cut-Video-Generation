"""Article archetypes: which kind of article this is, and the section plan
that follows from it (configs/blog/archetypes.yaml).

Used by the research agent (to pick the research depth), the planner (the
deterministic plan, and the menu the LLM chooses from) and the composer.
"""
from __future__ import annotations

import re
from typing import Iterable

from src.agents.blog import blog_config, brief as brief_mod
from src.data.schemas import BlogInput, BlogSectionPlan

# Tie-break order: the more specific shape wins.
_PRIORITY = ["comparison", "how_to", "investment_analysis", "explainer", "market_update",
             "buyers_guide", "locality_guide"]


def catalogue() -> dict:
    return blog_config.archetypes().get("archetypes", {})


def classify(blog_input: BlogInput, ctx: brief_mod.BriefContext | None = None) -> str:
    """The archetype for this brief: forced by `article_type`, else the
    best regex match on the topic (key points count for less)."""
    ctx = ctx or brief_mod.context(blog_input)
    specs = catalogue()
    forced = (getattr(blog_input, "article_type", "") or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"buyer_guide": "buyers_guide", "guide": "buyers_guide", "locality": "locality_guide",
               "investment": "investment_analysis", "news": "market_update", "howto": "how_to"}
    forced = aliases.get(forced, forced)
    if forced in specs:
        return forced
    if ctx.options:
        return "comparison"
    scores: dict[str, float] = {}
    secondary = " ".join(blog_input.key_points + list(getattr(blog_input, "custom_sections", [])))
    for name, spec in specs.items():
        score = 0.0
        for pattern in spec.get("match", []):
            if re.search(pattern, blog_input.topic, re.I):
                score += 2
            if secondary and re.search(pattern, secondary, re.I):
                score += 0.5
        if spec.get("needs_location") and not ctx.location.known:
            score -= 1.5
        scores[name] = score
    best = max(_PRIORITY, key=lambda n: (scores.get(n, 0), -_PRIORITY.index(n)))
    if scores.get(best, 0) <= 0:
        return "locality_guide" if ctx.location.known else "buyers_guide"
    return best


def placeholders(ctx: brief_mod.BriefContext) -> dict[str, str]:
    loc = ctx.location
    audience = ctx.audience_label
    return {
        "place": loc.short, "place_full": loc.full or loc.short, "city": loc.city,
        "audience": audience, "Audience": audience[:1].upper() + audience[1:],
        "subject": ctx.subject,
        "Keyword": ctx.keyword_display[:1].upper() + ctx.keyword_display[1:] if ctx.primary_keyword else "",
        "a": ctx.options[0] if len(ctx.options) > 0 else "",
        "b": ctx.options[1] if len(ctx.options) > 1 else "",
    }


def resolve(patterns: Iterable[str], ctx: brief_mod.BriefContext) -> str:
    values = placeholders(ctx)
    last = ""
    for pattern in patterns:
        names = re.findall(r"\{(\w+)\}", pattern)
        last = pattern
        if all(values.get(n) for n in names):
            return pattern.format(**values)
    return re.sub(r"\s*\{\w+\}\s*", " ", last).strip()


def _when(rule: str, covers: list[str], requested: set[str], categories: set[str],
          ctx: brief_mod.BriefContext) -> bool:
    rule = rule or "always"
    if rule == "always":
        return True
    if rule == "options":
        return bool(ctx.options)
    kind, _, arg = rule.partition(":")
    wanted = set(filter(None, arg.split(",")))
    asked = bool(requested & set(covers))
    has = bool(categories & wanted)
    if kind == "requested":
        return asked
    if kind == "facts":
        return has
    if kind == "requested_or_facts":
        return asked or has
    return True


def custom_heading(text: str) -> str:
    text = " ".join(text.split()).strip(" .:-")
    words = text.split()
    if len(words) > 12:
        text = " ".join(words[:12])
    return text[:1].upper() + text[1:]


def plan_sections(blog_input: BlogInput, archetype: str, categories: set[str],
                  ctx: brief_mod.BriefContext | None = None) -> list[BlogSectionPlan]:
    ctx = ctx or brief_mod.context(blog_input)
    cfg = blog_config.archetypes()
    spec = catalogue().get(archetype) or catalogue()["buyers_guide"]
    requested = set(blog_input.sections)
    plans: list[BlogSectionPlan] = []
    covered: set[str] = set()
    for slot in spec.get("slots", []):
        covers = list(slot.get("covers", []))
        if not _when(slot.get("when", "always"), covers, requested, categories, ctx):
            continue
        plans.append(BlogSectionPlan(
            heading=resolve(slot.get("heading", [slot["id"]]), ctx), purpose=slot.get("purpose", ""),
            format=slot.get("format", "prose"), covers=[c for c in covers if c in requested],
            slot=slot["id"]))
        covered |= requested & set(covers)

    # A requested section the archetype has no slot for is still a
    # requirement: add the generic slot for it rather than drop it.
    fallback = cfg.get("fallback_slots", {})
    for sid in blog_input.sections:
        if sid in covered:
            continue
        slot_id = fallback.get(sid)
        existing = next((p for p in plans if p.slot == slot_id), None)
        if existing:
            existing.covers.append(sid)
            continue
        if not slot_id:
            continue
        plans.append(BlogSectionPlan(
            heading=resolve(cfg.get("slot_headings", {}).get(slot_id, [slot_id.replace("_", " ").title()]), ctx),
            purpose=blog_config.sections().get(sid, {}).get("guidance", ""),
            format=cfg.get("slot_formats", {}).get(slot_id, "prose"), covers=[sid], slot=slot_id))
        covered.add(sid)

    # The author's own sections, before the closing decision sections.
    closing = {"who_it_suits", "due_diligence"}
    insert_at = next((i for i, p in enumerate(plans) if p.slot in closing), len(plans))
    for text in brief_mod.custom_sections(blog_input):
        plans.insert(insert_at, BlogSectionPlan(heading=custom_heading(text), purpose=text, format="prose",
                                                covers=[text], slot="custom"))
        insert_at += 1

    # Closing sections last, in a fixed order: the decision, then the checks.
    body = [p for p in plans if p.slot not in closing]
    tail = sorted([p for p in plans if p.slot in closing], key=lambda p: ["who_it_suits", "due_diligence"].index(p.slot))
    plans = body + tail
    max_sections = blog_config.limits().get("max_sections", 10)
    if len(plans) > max_sections:
        keep_slots = {p.slot for p in tail} | {"custom"}
        required = [p for p in plans if p.covers or p.slot in keep_slots]
        optional = [p for p in plans if p not in required]
        plans = [p for p in plans if p in required or p in optional[:max(0, max_sections - len(required))]]
    # Distinct headings.
    seen: set[str] = set()
    for p in plans:
        base, n = p.heading, 2
        while p.heading.lower() in seen:
            p.heading = f"{base} ({n})"
            n += 1
        seen.add(p.heading.lower())
    return plans


_SMALL = {"a", "an", "the", "and", "or", "for", "in", "on", "at", "to", "of", "vs", "vs.", "with", "by",
          "from", "as", "per", "is"}


def title_case(text: str) -> str:
    out = []
    for i, word in enumerate(text.split()):
        if any(ch.isupper() for ch in word[1:]) or word.isupper():
            out.append(word)                      # RERA, NRI, 2BHK, iPhone
        elif i > 0 and word.lower() in _SMALL and not out[-1].endswith(":"):
            out.append(word.lower())
        else:
            out.append("-".join(p[:1].upper() + p[1:] for p in word.split("-")))
    return " ".join(out)


def working_title(blog_input: BlogInput, archetype: str, ctx: brief_mod.BriefContext | None = None) -> str:
    """The topic as a headline. It already matches what the reader typed;
    a short topic gets the archetype's promise added after a colon."""
    ctx = ctx or brief_mod.context(blog_input)
    topic = " ".join(blog_input.topic.split()).strip()
    title = title_case(topic[:1].upper() + topic[1:])
    suffix = catalogue().get(archetype, {}).get("title_suffix", "")
    if suffix and len(title) <= 46 and ":" not in title and not title.endswith("?"):
        bare = re.sub(r"\{\w+\}", "", suffix).strip()
        title = f"{title}: {resolve([suffix, bare], ctx)}".strip()
    return title
