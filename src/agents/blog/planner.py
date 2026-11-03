"""Planner Agent — the article's archetype, angle and section plan.

The outline is planned AFTER the research, so the structure can follow
what was found: a connectivity section is worth a table when there are
real distances to put in it, and worth cutting when there are none.

The deterministic plan (configs/blog/archetypes.yaml) is built first,
always. It is the fallback when no model is reachable, AND it is handed
to the model as the starting point — so the model refines a sound plan
for this archetype instead of improvising one generic template for every
topic, which is how posts ended up structurally interchangeable.
"""
from __future__ import annotations

import json
import logging
from string import Template
from typing import Optional

from src.agents.blog import archetypes, brief, facts as facts_mod, llm
from src.agents.blog import research as research_agent
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import BlogInput, BlogOutline, BlogSectionPlan, ResearchPack
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)

FORMATS = {"prose", "list", "table", "checklist", "steps", "faq", "pros_cons"}


def deterministic_outline(blog_input: BlogInput, research: Optional[ResearchPack] = None) -> BlogOutline:
    ctx = brief.context(blog_input)
    archetype = archetypes.classify(blog_input, ctx)
    categories = {f.category for f in facts_mod.collect(blog_input, research)}
    plans = archetypes.plan_sections(blog_input, archetype, categories, ctx)
    spec = archetypes.catalogue().get(archetype, {})
    return BlogOutline(
        working_title=archetypes.working_title(blog_input, archetype, ctx),
        section_headings=[p.heading for p in plans],
        section_briefs={p.heading: p.purpose for p in plans},
        archetype=archetype,
        angle=(f"Help {ctx.audience_label} decide, by separating verified facts from claims, naming the "
               f"trade-offs honestly and ending with a checklist — {ctx.audience.get('concern', '')}."),
        audience=ctx.audience_label,
        search_intent=spec.get("search_intent", ""),
        primary_keyword=ctx.primary_keyword,
        secondary_keywords=brief.secondary_keywords(blog_input, archetype, ctx.location),
        sections=plans,
        planner_mode="fallback",
    )


def _archetype_menu() -> str:
    return "\n".join(f"- {name}: {spec.get('label', name)} — {spec.get('search_intent', '')}"
                     for name, spec in archetypes.catalogue().items())


def run(blog_input: BlogInput, research: Optional[ResearchPack] = None) -> BlogOutline:
    """Turns the brief into a section-by-section outline. Every downstream
    agent treats this (plus the original brief) as the structure to
    follow — the Writer doesn't invent its own structure from scratch."""
    base = deterministic_outline(blog_input, research)
    try:
        prompt = Template(load_yaml_config("blog/prompts.yaml")["planner"]).substitute(
            topic=blog_input.topic,
            research=research_agent.as_prompt_block(research) if research else
            "(no research was gathered for this post)",
            tone=brief.tone(blog_input),
            target_word_count=blog_input.target_word_count or 1200,
            requested_sections=brief.requested_sections_block(blog_input),
            key_points=brief.key_points_block(blog_input),
            local_facts=brief.local_facts_block(blog_input),
            brief_profile=brief.brief_profile_block(blog_input),
            archetypes=_archetype_menu(),
            suggested_archetype=base.archetype,
            suggested_plan=json.dumps([p.model_dump() for p in base.sections], indent=1, ensure_ascii=False),
        )
        result = llm.generate_json(ModelRole.BLOG_PLANNER, prompt, call_with_fallback) or {}
    except Exception as exc:  # noqa: BLE001 - the deterministic plan is a complete plan
        logger.warning("Planner model unavailable; using the archetype plan: %s", exc)
        return base
    return _merge(base, result, blog_input)


def _merge(base: BlogOutline, result: dict, blog_input: BlogInput) -> BlogOutline:
    """The model's plan, checked against the deterministic one: an unknown
    archetype, a missing format, or fewer than three sections falls back
    to the base for that part rather than failing the run."""
    archetype = str(result.get("archetype") or base.archetype)
    if archetype not in archetypes.catalogue():
        archetype = base.archetype
    plans: list[BlogSectionPlan] = []
    base_slots = {p.slot for p in base.sections}
    for raw in result.get("sections") or []:
        if not isinstance(raw, dict) or not str(raw.get("heading", "")).strip():
            continue
        fmt = str(raw.get("format") or "prose").strip().lower()
        slot = str(raw.get("slot") or "custom").strip()
        plans.append(BlogSectionPlan(
            heading=" ".join(str(raw["heading"]).split()), purpose=str(raw.get("purpose", "")),
            format=fmt if fmt in FORMATS else "prose",
            covers=[str(c) for c in raw.get("covers") or []],
            slot=slot if slot in base_slots or slot == "custom" else "custom"))
    if len(plans) < 3:
        # Legacy shape: headings + briefs only.
        headings = [str(h).strip() for h in result.get("section_headings") or [] if str(h).strip()]
        briefs = result.get("section_briefs") or {}
        if len(headings) >= 2:
            plans = [BlogSectionPlan(heading=h, purpose=str(briefs.get(h, "")), slot="custom") for h in headings]
    if not plans:
        plans = base.sections
    # Any requested section the model's plan dropped comes back from the base plan.
    covered = {c for p in plans for c in p.covers}
    for p in base.sections:
        missing = [c for c in p.covers if c not in covered]
        if missing and p.slot not in {q.slot for q in plans}:
            plans.append(p)
    stated_kw = (blog_input.primary_keyword or "").strip()
    return BlogOutline(
        working_title=str(result.get("working_title") or base.working_title).strip(),
        section_headings=[p.heading for p in plans],
        section_briefs={p.heading: p.purpose for p in plans},
        archetype=archetype,
        angle=str(result.get("angle") or base.angle),
        audience=base.audience,
        search_intent=str(result.get("search_intent") or base.search_intent),
        primary_keyword=stated_kw or str(result.get("primary_keyword") or base.primary_keyword).strip(),
        secondary_keywords=[str(k) for k in (result.get("secondary_keywords") or base.secondary_keywords)][:6],
        sections=plans,
        planner_mode="llm",
    )
