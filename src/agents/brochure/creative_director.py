"""The Creative Director: chooses the brochure's palette.

From a CLOSED set — the category's own palettes (palettes.py), every one of
which passes the contrast check for every pairing that carries type. A
model may choose the mood that suits the property (warm and sunlit,
evening and dramatic, cool and architectural); it cannot invent a colour,
and so it cannot choose an illegible one. With no model, the category's
first palette is used.
"""
from __future__ import annotations

import json
import logging
from string import Template

from src.agents.base import with_retries
from src.agents.brochure import palettes
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import PropertyKnowledge
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)


def choose(knowledge: PropertyKnowledge | None, category: str, use_model: bool = True) -> palettes.Palette:
    options = palettes.options(category)
    if not use_model or knowledge is None:
        return options[0]
    try:
        prompt = Template(load_yaml_config("brochure/prompts.yaml")["creative_director"]).substitute(
            selling_points="\n".join(f"- {p}" for p in knowledge.selling_points) or "(none given)",
            allowed_options=json.dumps([{"style": p.name, "mood": p.mood} for p in options], indent=2),
        )
        result = with_retries(lambda: call_with_fallback(ModelRole.CREATIVE_DIRECTOR, "generate_json", prompt),
                              max_retries=1)
        return palettes.for_category(category, str(result.get("style", "")))
    except Exception as exc:  # noqa: BLE001 - the first palette is always a sound choice
        logger.info("Creative director unavailable (%s); using the category's first palette",
                    type(exc).__name__)
        return options[0]
