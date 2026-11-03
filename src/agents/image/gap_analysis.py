"""Image Gap Analysis Agent (brochure).

Decides which 2-3 AI images would genuinely improve a brochure, given
what the uploaded photos already cover. An LLM proposes; deterministic
code then enforces the rules the LLM might bend:

* 2 <= count <= 3
* only ALLOWED_GENERATED_KINDS, and never a subject that depicts the
  actual property (see prompting.is_allowed)
* no duplicate kind+subject

If the LLM call fails or returns too few valid briefs, a deterministic
planner fills the gap from the coverage data, so the step never blocks
brochure generation.
"""
from __future__ import annotations

import logging
from collections import Counter
from string import Template
from typing import Optional

from src.agents.image.prompting import is_allowed
from src.agents.base import with_retries
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.models.gateway import call_with_fallback
from src.data.schemas import ImageAnalysis, ImageBrief, PropertyInput, PropertyKnowledge

logger = logging.getLogger(__name__)

MIN_IMAGES, MAX_IMAGES = 2, 3


def _category(a: ImageAnalysis) -> str:
    return a.category.value if hasattr(a.category, "value") else str(a.category)


def coverage_of(analyses: list[ImageAnalysis]) -> Counter:
    return Counter(_category(a) for a in analyses)


def _fallback_briefs(prop: PropertyInput, coverage: Counter, photo_text: str = "") -> list[ImageBrief]:
    """Deterministic plan: prefer gaps the photos genuinely leave open."""
    city = prop.location.city if prop.location else ""
    candidates: list[ImageBrief] = []
    if coverage.get("lifestyle", 0) == 0:
        candidates.append(ImageBrief(
            kind="lifestyle", aspect="landscape", placement="lifestyle section",
            purpose="The photos show spaces but no sense of daily life in them.",
            subject="a couple sharing coffee on a spacious private terrace at golden hour, seen from behind, "
                    "planters and a low parapet, city softly out of focus beyond",
            caption="Evenings that belong to you.",
        ))
    # What the photos actually show: categories PLUS the vision model's own
    # scene tags and captions, so a pool photo filed under "amenity" still
    # counts as a photographed pool.
    photographed = (" ".join(coverage) + " " + photo_text).lower()
    for amenity in prop.amenities or []:
        tokens = [t for t in amenity.lower().split() if len(t) > 3]
        if tokens and not any(t in photographed for t in tokens):
            candidates.append(ImageBrief(
                kind="amenity_concept", aspect="landscape", placement="amenities section",
                purpose=f"'{amenity}' is a listed amenity with no photograph.",
                subject=f"a serene, uncluttered {amenity} in a contemporary residential clubhouse, "
                        f"no people, morning light",
                caption=f"{amenity.strip().capitalize()}, reimagined for everyday calm.",
            ))
            break
    if city:
        candidates.append(ImageBrief(
            kind="neighbourhood", aspect="wide", placement="location section",
            purpose="Nothing situates the property in its city.",
            subject=f"a tree-lined boulevard in {city} at dusk, soft streetlights, calm traffic, "
                    f"no identifiable landmarks",
            caption=f"Connected to the rhythm of {city}.",
        ))
    candidates.append(ImageBrief(
        kind="design_element", aspect="portrait", placement="section opener",
        purpose="A quiet texture to open a section and balance text-heavy pages.",
        subject="close crop of warm brick coursing and a slim bronze reveal in raking evening light",
        caption="",
    ))
    return candidates


def _dedupe_and_limit(briefs: list[ImageBrief]) -> list[ImageBrief]:
    out, seen = [], set()
    for b in briefs:
        key = (b.kind, b.subject.strip().lower()[:60])
        if key in seen or not is_allowed(b):
            continue
        seen.add(key)
        out.append(b)
        if len(out) == MAX_IMAGES:
            break
    return out


def run(prop: PropertyInput, knowledge: Optional[PropertyKnowledge],
        analyses: list[ImageAnalysis], llm_call=None) -> list[ImageBrief]:
    coverage = coverage_of(analyses)
    proposed: list[ImageBrief] = []
    try:
        if llm_call is None:
            llm_call = lambda p: with_retries(lambda: call_with_fallback(ModelRole.IMAGE_GAP, "generate_json", p))
        prompt = Template(load_yaml_config("brochure/prompts.yaml")["image_gap"]).substitute(
            name=prop.name,
            location=", ".join(x for x in [prop.location.area, prop.location.city] if x) if prop.location else "",
            configuration=prop.configuration or "not specified",
            amenities=", ".join(prop.amenities) or "none listed",
            selling_points="\n".join(f"- {s}" for s in (knowledge.selling_points if knowledge else [])) or "- none",
            coverage="\n".join(f"{k}: {v}" for k, v in sorted(coverage.items())) or "no photos analysed",
        )
        result = llm_call(prompt) or {}
        for item in result.get("images", []):
            try:
                proposed.append(ImageBrief(**{k: item.get(k, "") for k in ImageBrief.model_fields}))
            except Exception:  # malformed entry: skip it, don't fail the step
                continue
    except Exception as exc:
        logger.warning("Image gap LLM call failed; using deterministic plan: %s", exc)

    briefs = _dedupe_and_limit(proposed)
    rejected = len(proposed) - len(briefs)
    if rejected:
        logger.info("Image gap: dropped %d brief(s) that were duplicate or depicted the property", rejected)
    if len(briefs) < MIN_IMAGES:
        photo_text = " ".join(f"{a.scene_tag} {a.suggested_title} {a.suggested_caption}" for a in analyses)
        briefs = _dedupe_and_limit(briefs + _fallback_briefs(prop, coverage, photo_text))
    return briefs
