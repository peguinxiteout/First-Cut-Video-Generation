"""Writer Agent — full prose from the outline, the brief and the research.

The Writer has NO tools of its own, deliberately. A writer that can
search mid-sentence wanders: it produces a different post every run and
makes the result impossible to evaluate, because a weak post can no
longer be traced to weak research or weak writing.

Two paths, one bar:
  * the model writes from configs/blog/prompts.yaml `writer`, which now
    carries the archetype, the planned FORMAT of each section, hard
    readability and markdown rules, and the revision feedback;
  * when no model answers, src/agents/blog/composer.py writes the post
    deterministically from the same plan and the same facts.
Either way the draft then goes through the Editor (normaliser +
validator) and the same rubric.

`revision_note` carries what was wrong with the previous attempt —
dimension by dimension, plus the sentences the fact check could not
support — so the revision loop is never blind.
"""
from __future__ import annotations

import logging
from datetime import date
from string import Template
from typing import Optional

from src.agents.blog import brief, composer, llm
from src.agents.blog import research as research_agent
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import (
    BlogContent, BlogFAQ, BlogImageAsset, BlogInput, BlogOutline, BlogSection, ResearchPack,
)
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)


def _available_images_block(images: list[BlogImageAsset]) -> str:
    if not images:
        return "(no images provided)"
    return "\n".join(f"- {img.image_id}: filename {img.filename}" for img in images)


def _section_plan_block(outline: BlogOutline) -> str:
    if not outline.sections:
        return outline.model_dump_json(indent=2)
    lines = []
    for i, p in enumerate(outline.sections, 1):
        covers = f" (covers: {', '.join(p.covers)})" if p.covers else ""
        lines.append(f"{i}. \"{p.heading}\" — FORMAT: {p.format}. {p.purpose}{covers}")
    return "\n".join(lines)


def _allowed_links_block(blog_input: BlogInput, research: Optional[ResearchPack]) -> str:
    refs = composer.build_references(blog_input, research)
    urls = [f"- {r.title}: {r.url}" for r in refs if r.url]
    return "\n".join(urls) or "(none — do not include any links)"


def run(blog_input: BlogInput, outline: BlogOutline, images: list[BlogImageAsset],
        research: Optional[ResearchPack] = None, revision_note: str = "") -> BlogContent:
    """Expands the outline into full prose. Images are referenced by id —
    the Writer just decides which section, if any, each one fits best."""
    try:
        content = _write_with_model(blog_input, outline, images, research, revision_note)
    except Exception as exc:  # noqa: BLE001 - the composer is a complete writer
        logger.warning("Writer model unavailable; composing deterministically: %s", exc)
        content = composer.compose(blog_input, outline, images, research)
    return finish(content, blog_input, outline, research)


def _write_with_model(blog_input: BlogInput, outline: BlogOutline, images: list[BlogImageAsset],
                      research: Optional[ResearchPack], revision_note: str) -> BlogContent:
    prompt = Template(load_yaml_config("blog/prompts.yaml")["writer"]).substitute(
        topic=blog_input.topic,
        research=research_agent.as_prompt_block(research) if research else
        "(no research was gathered — do not state local facts the author did not provide)",
        revision_note=_revision_block(revision_note),
        tone=brief.tone(blog_input),
        target_word_count=blog_input.target_word_count or "1100-1500",
        archetype=outline.archetype or "(not set)",
        angle=outline.angle or "",
        search_intent=outline.search_intent or "",
        brief_profile=brief.brief_profile_block(blog_input),
        working_title=outline.working_title,
        section_plan=_section_plan_block(outline),
        outline_json=outline.model_dump_json(indent=2),
        requested_sections=brief.requested_sections_block(blog_input),
        key_points=brief.key_points_block(blog_input),
        local_facts=brief.local_facts_block(blog_input),
        available_images=_available_images_block(images),
        allowed_links=_allowed_links_block(blog_input, research),
        cta_instruction=brief.cta_instruction(blog_input),
    )
    result = llm.generate_json(ModelRole.BLOG_WRITER, prompt, call_with_fallback)
    if not isinstance(result, dict):
        raise ValueError("writer returned no JSON object")
    sections = []
    plans = {p.heading.lower(): p for p in outline.sections}
    for raw in result.get("sections") or []:
        if not isinstance(raw, dict) or not str(raw.get("body_markdown", "")).strip():
            continue
        heading = str(raw.get("heading", "")).strip()
        plan = plans.get(heading.lower())
        image_id = raw.get("image_id")
        sections.append(BlogSection(
            heading=heading, body_markdown=str(raw["body_markdown"]),
            image_id=image_id if image_id and image_id in {i.image_id for i in images} else None,
            format=plan.format if plan else str(raw.get("format", "")), slot=plan.slot if plan else ""))
    if len(sections) < 2 or not str(result.get("intro_markdown", "")).strip():
        raise ValueError("writer returned an incomplete post")
    faqs = [BlogFAQ(question=str(f.get("question", "")).strip(), answer_markdown=str(f.get("answer_markdown")
                                                                                     or f.get("answer", "")))
            for f in result.get("faqs") or [] if isinstance(f, dict) and f.get("question")]
    return BlogContent(
        title=str(result.get("title") or outline.working_title),
        intro_markdown=str(result.get("intro_markdown", "")),
        sections=sections,
        conclusion_heading=str(result.get("conclusion_heading", "") or ""),
        conclusion_markdown=str(result.get("conclusion_markdown", "")),
        cta_heading=str(result.get("cta_heading", "") or ""),
        cta_markdown=str(result.get("cta_markdown", "") or ""),
        cover_image_id=result.get("cover_image_id") if result.get("cover_image_id") in
        {i.image_id for i in images} else None,
        key_takeaways=[str(t) for t in result.get("key_takeaways") or [] if str(t).strip()][:5],
        faqs=[f for f in faqs if f.answer_markdown.strip()][:5],
        writer_mode="llm",
    )


def _revision_block(note: str) -> str:
    if not note.strip():
        return ""
    return ("REVISION — the previous draft was scored and fell short. Fix EVERY point below; "
            "keep what was good:\n" + note.strip())


def finish(content: BlogContent, blog_input: BlogInput, outline: BlogOutline,
           research: Optional[ResearchPack]) -> BlogContent:
    """What every draft gets, whoever wrote it: references built from the
    real sources, the byline, the date and the keyword plan."""
    ctx = brief.context(blog_input)
    content.references = composer.build_references(blog_input, research, ctx)
    content.archetype = content.archetype or outline.archetype
    content.primary_keyword = content.primary_keyword or outline.primary_keyword or ctx.primary_keyword
    content.secondary_keywords = content.secondary_keywords or list(outline.secondary_keywords)
    content.author = brief.author(blog_input)
    today = date.today()
    content.published_on = f"{today.day} {today:%B %Y}"
    if not content.conclusion_heading:
        content.conclusion_heading = "The verdict"
    return content
