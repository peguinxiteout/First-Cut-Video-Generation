"""SEO Agent — meta title, meta description, slug, keywords and topic tags.

Keywords come from real demand where possible: before the model sees
anything, the toolbox asks a search-autocomplete endpoint what people
actually type around this topic. The model then picks from those — but
only the ones the post genuinely answers, because a keyword the article
does not deliver on wins a click and loses the reader.

Every field is then checked and, if needed, rebuilt deterministically:
meta title within 30-60 characters, meta description within 110-160 and
containing the primary keyword, a short hyphenated slug, 3-6 topic tags
drawn from the brief (place, archetype, audience) rather than invented.
That is also the complete fallback when no model is reachable.

It still never touches the title or the body.
"""
from __future__ import annotations

import logging
import re
from string import Template
from typing import Optional

from src.agents.blog import archetypes, article, brief, llm
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import BlogContent, BlogInput, ResearchPack
from src.models.gateway import call_with_fallback
from src.tools import ToolBox, toolbox_for

logger = logging.getLogger(__name__)

MAX_SUGGESTIONS = 18
META_TITLE = (30, 60)
META_DESCRIPTION = (110, 160)


def _full_markdown(content: BlogContent) -> str:
    return article.full_markdown(content, include_cta=False)


def search_demand(seed: str, box: Optional[ToolBox] = None) -> tuple[list[str], list[str]]:
    """(phrasings people search, question-shaped ones). Empty if unavailable."""
    if not seed.strip():
        return [], []
    box = box if box is not None else toolbox_for("seo")
    result = box.keyword_suggestions(seed, limit=MAX_SUGGESTIONS)
    if not result.ok or not result.data:
        return [], []
    return result.data.get("suggestions", []), result.data.get("questions", [])


def run(content: BlogContent, research: Optional[ResearchPack] = None,
        box: Optional[ToolBox] = None, blog_input: Optional[BlogInput] = None) -> BlogContent:
    """Adds search metadata. Title and body are left exactly as written."""
    keyword = content.primary_keyword or (brief.primary_keyword(blog_input) if blog_input else "")
    seed = keyword or (research.subject if research and research.subject else content.title) or ""
    offline = bool(research and any("could not be reached" in n for n in research.notes))
    suggestions, _questions = ([], []) if offline else search_demand(seed, box)
    demand_block = (
        "\n".join(f"- {s}" for s in suggestions) if suggestions
        else "(no search data was available — judge the keywords from the post alone)"
    )
    result: dict = {}
    try:
        prompt = Template(load_yaml_config("blog/prompts.yaml")["seo"]).substitute(
            title=content.title, post_markdown=_full_markdown(content), search_demand=demand_block,
            primary_keyword=keyword or "(derive from the post)",
            brief_profile=brief.brief_profile_block(blog_input) if blog_input else "",
        )
        result = llm.generate_json(ModelRole.BLOG_SEO, prompt, call_with_fallback) or {}
    except Exception as exc:  # noqa: BLE001 - metadata can always be built from the brief
        logger.warning("SEO model unavailable; building metadata deterministically: %s", exc)

    relevant_suggestions = [s for s in suggestions if _relevant(s, content, keyword)]
    return finalise(content, blog_input, result, relevant_suggestions)


def _relevant(phrase: str, content: BlogContent, keyword: str) -> bool:
    words = [w for w in re.findall(r"[a-z]{4,}", phrase.lower())]
    body = article.full_markdown(content).lower()
    return bool(words) and sum(1 for w in words if w in body) >= max(1, len(words) - 1)


def _trim(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + "…"


def _meta_title(content: BlogContent, keyword: str, proposed: str) -> str:
    lo, hi = META_TITLE
    for candidate in (proposed, content.title):
        candidate = " ".join((candidate or "").split())
        if lo <= len(candidate) <= hi:
            return candidate
    head = re.split(r"[:?—]", content.title)[0].strip()
    if lo <= len(head) <= hi:
        return head
    if keyword:
        kw = archetypes.title_case(keyword)
        candidate = f"{kw}: A Practical Guide"
        if len(candidate) <= hi:
            return candidate
        return _trim(kw, hi)
    return _trim(content.title, hi)


def _meta_description(content: BlogContent, keyword: str, proposed: str, audience: str) -> str:
    lo, hi = META_DESCRIPTION
    from src.evaluation.blog_metrics import keyword_present

    proposed = " ".join((proposed or "").split())
    if lo <= len(proposed) <= hi and (not keyword or keyword_present(proposed, keyword)):
        return proposed
    # Capitals come from the place names, not from the title-cased headline.
    reference = " ".join(t for t in content.tags if t[:1].isupper())
    kw = brief.display_case(brief.prose_keyword(keyword), reference) if keyword else content.title
    promise = {
        "locality_guide": "verified facts, honest trade-offs and a due-diligence checklist",
        "buyers_guide": "the steps in order, the real costs and a pre-booking checklist",
        "comparison": "a side-by-side comparison and a decision guide",
        "investment_analysis": "demand drivers, holding costs, risks and an investor checklist",
        "explainer": "what it means in practice and how to check it yourself",
        "how_to": "each step in order, the documents you need and common pitfalls",
        "market_update": "what has changed, what is only announced and what it means for you",
    }.get(content.archetype, "verified facts, trade-offs and a practical checklist")
    text = f"Considering {kw}? Get {promise} for {audience}."
    if len(text) < lo:
        text = f"Considering {kw}? Get {promise} for {audience} — every figure labelled by source."
    if len(text) > hi:
        text = f"{kw[:1].upper() + kw[1:]}: {promise}."
    if len(text) > hi:
        text = _trim(text, hi)
    return text


def _slug(keyword: str, title: str) -> str:
    base = keyword or title
    words = [w for w in re.findall(r"[a-z0-9]+", base.lower()) if w not in {"a", "an", "the", "of", "and"}]
    slug = "-".join(words[:8])
    return slug[:75].rstrip("-") or "article"


def _tags(content: BlogContent, blog_input: Optional[BlogInput], proposed: list) -> list[str]:
    tags: list[str] = []
    if blog_input is not None:
        ctx = brief.context(blog_input)
        for name in (ctx.location.short, ctx.location.city):
            if name and name.lower() not in (t.lower() for t in tags):
                tags.append(name)
        if ctx.options:
            tags += ctx.options[:2]
    if blog_input is not None and not brief.context(blog_input).location.known:
        subject = brief.subject(blog_input)
        if len(subject.split()) <= 5:
            tags.append(subject)
    label = archetypes.catalogue().get(content.archetype, {}).get("label")
    if label:
        tags.append(label)
    tags += [str(t) for t in proposed or []]
    if blog_input is not None and blog_input.audience:
        tags.append(blog_input.audience)
    elif blog_input is not None:
        tags.append(brief.audience_profile(blog_input)["label"])
    tags += list(content.secondary_keywords[:2])
    out: list[str] = []
    for t in tags:
        t = " ".join(str(t).split()).strip(" #,")
        if t and len(t) <= 40 and t.lower() not in (o.lower() for o in out):
            out.append(t[:1].upper() + t[1:])
    return out[:6]


def finalise(content: BlogContent, blog_input: Optional[BlogInput], result: dict,
             suggestions: list[str]) -> BlogContent:
    keyword = content.primary_keyword or (brief.primary_keyword(blog_input) if blog_input else "")
    audience = brief.audience_profile(blog_input)["label"] if blog_input else "buyers"
    keywords = [keyword] if keyword else []
    for k in list(result.get("seo_keywords") or []) + list(content.secondary_keywords) + suggestions:
        k = " ".join(str(k).split())
        if k and k.lower() not in (x.lower() for x in keywords):
            keywords.append(k)
    proposed_slug = str(result.get("slug") or "")
    slug = proposed_slug if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+){1,9}", proposed_slug) else _slug(keyword, content.title)
    return content.model_copy(update={
        "meta_title": _meta_title(content, keyword, str(result.get("meta_title") or "")),
        "meta_description": _meta_description(content, keyword, str(result.get("meta_description") or ""), audience),
        "seo_keywords": keywords[:8],
        "slug": slug,
        "tags": _tags(content, blog_input, result.get("tags") or []),
    })
