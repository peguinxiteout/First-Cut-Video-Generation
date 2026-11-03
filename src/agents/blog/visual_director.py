"""Visual Director Agent (blog).

Writes the brief for a blog's hero image from the article's title and
topic, then hands it to the shared Image Generation Agent. Same safety
rules as the brochure: no specific named building, no text in the image.
Falls back to a deterministic brief if the LLM call fails, so every blog
still gets a hero attempt.
"""
from __future__ import annotations

import logging
from string import Template

from src.agents.image.prompting import is_allowed
from src.agents.blog import archetypes, brief as brief_mod, llm
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.models.gateway import call_with_fallback
from src.data.schemas import BlogInput, ImageBrief

logger = logging.getLogger(__name__)


_FALLBACK_SUBJECTS = {
    "explainer": "hands holding a pen over printed paperwork on a wooden desk beside a cup of tea, soft daylight, "
                 "shallow depth of field",
    "how_to": "an organised wooden desk with paperwork, a pen and a set of keys, soft morning light",
    "investment_analysis": "a city skyline with residential towers at dusk, warm window lights, generous sky",
    "market_update": "an elevated view over residential districts at dusk, soft haze, generous sky",
    "comparison": "an elevated view over two residential neighbourhoods separated by a tree-lined road, "
                  "late afternoon light",
}


def _fallback(blog_input: BlogInput) -> ImageBrief:
    """A safe, relevant hero without a model: the archetype decides the
    scene, the city (never a specific building) sets it."""
    ctx = brief_mod.context(blog_input)
    archetype = archetypes.classify(blog_input, ctx)
    subject = _FALLBACK_SUBJECTS.get(archetype)
    candidate = ImageBrief(kind="blog_hero", aspect="wide", purpose="Hero image for the article",
                           subject=subject or "", caption="")
    if subject and not is_allowed(candidate):
        subject = None
    if not subject:
        where = f" in {ctx.location.city}" if ctx.location.city else ""
        subject = (f"a calm, tree-lined residential street{where} with contemporary mid-rise apartment buildings "
                   "at golden hour, long soft shadows, generous sky")
    return ImageBrief(kind="blog_hero", aspect="wide", purpose="Hero image for the article",
                      subject=subject, caption="")


def run(blog_input: BlogInput, working_title: str, llm_call=None) -> ImageBrief:
    try:
        if llm_call is None:
            llm_call = lambda p: llm.generate_json(ModelRole.VISUAL_DIRECTOR, p, call_with_fallback)
        prompt = Template(load_yaml_config("blog/prompts.yaml")["visual_director"]).substitute(
            title=working_title or blog_input.topic,
            topic=blog_input.topic,
            location=brief_mod.context(blog_input).location.full or "as implied by the topic",
        )
        result = llm_call(prompt) or {}
        subject = str(result.get("subject", "")).strip()
        if subject:
            brief = ImageBrief(kind="blog_hero", aspect="wide", purpose="Hero image for the article",
                               subject=subject, caption=str(result.get("caption", "")).strip())
            if is_allowed(brief):
                return brief
            logger.info("Visual director proposed a disallowed hero subject; using fallback")
    except Exception as exc:
        logger.warning("Visual director failed; using fallback hero brief: %s", exc)
    return _fallback(blog_input)
