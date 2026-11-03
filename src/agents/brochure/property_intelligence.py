from __future__ import annotations

from string import Template
from typing import Any

from src.agents.base import with_retries
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import PropertyInput, PropertyKnowledge
from src.models.gateway import call_with_fallback


def _coerce_facts_to_strings(facts: dict[str, Any]) -> dict[str, str]:
    """Defensively flattens any non-string fact values into strings before
    validation. The prompt asks for flat string values, but models
    sometimes echo a nested object/array straight from property.json (e.g.
    {"location": {"city": "Noida", "area": "", "address": ""}} instead of
    "location": "Noida") — especially when several fields are empty, which
    seems to make it more likely to paste the raw structure back rather
    than summarize it. Rather than let that crash the whole generation
    with a Pydantic error, coerce it: flatten dicts/lists into a readable
    string, and drop facts that end up empty (an empty amenities list
    isn't a useful fact either way)."""
    coerced: dict[str, str] = {}
    for key, value in facts.items():
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            parts = [f"{k}: {v}" for k, v in value.items() if v not in (None, "", [], {})]
            text = ", ".join(parts)
        elif isinstance(value, list):
            text = ", ".join(str(item) for item in value if item not in (None, ""))
        elif value is None:
            text = ""
        else:
            text = str(value)

        if text:
            coerced[key] = text
    return coerced


def run(property_input: PropertyInput, documents_text: str = "") -> PropertyKnowledge:
    """Turns the raw property.json into a fact ledger + selling points.
    Every downstream agent that writes copy must treat this as the only
    source of truth (see the Copywriter/Critic prompts).

    Note: prompts are filled in with string.Template ($var), not str.format()
    — the prompt text itself contains literal JSON examples full of {curly
    braces}, which would collide with .format()'s placeholder syntax."""
    prompt_template = load_yaml_config("brochure/prompts.yaml")["property_intelligence"]
    prompt = Template(prompt_template).substitute(
        property_json=property_input.model_dump_json(indent=2),
        documents=documents_text or "(none provided)",
    )

    try:
        result = with_retries(
            lambda: call_with_fallback(ModelRole.PROPERTY_INTELLIGENCE, "generate_json", prompt)
        )
    except Exception:  # noqa: BLE001 - the ledger can be built from the form itself
        return ledger_from_form(property_input)
    if not isinstance(result, dict):
        return ledger_from_form(property_input)
    if isinstance(result.get("facts"), dict):
        result["facts"] = _coerce_facts_to_strings(result["facts"])
    for key in ("specifications", "legal_notes"):
        if not isinstance(result.get(key), list):
            result[key] = []
        result[key] = [str(x).strip() for x in result[key] if str(x).strip()]
    result["rera_number"] = str(result.get("rera_number") or "").strip()
    knowledge = PropertyKnowledge.model_validate(result)
    # A RERA number typed into the form always wins over one extracted
    # from a document, and is never replaced by a model's guess.
    if property_input.rera_number:
        knowledge.rera_number = property_input.rera_number.strip()
    return knowledge


def ledger_from_form(property_input: PropertyInput) -> PropertyKnowledge:
    """The fact ledger, built from exactly what the person typed — the
    fallback when no model is available, and never less truthful for it."""
    p = property_input
    facts: dict[str, str] = {"name": p.name, "city": p.location.city}
    for key, value in (("developer", p.developer), ("locality", p.location.area),
                       ("address", p.location.address), ("property_type", p.property_type),
                       ("category", p.category), ("configuration", p.configuration),
                       ("area", p.area), ("price", p.price), ("possession", p.possession),
                       ("project_size", p.project_size), ("target_audience", p.target_audience)):
        if value:
            facts[key] = str(value)
    for unit in p.unit_types:
        facts[f"unit: {unit.name}"] = ", ".join(x for x in (unit.carpet_area, unit.price) if x)
    if p.amenities:
        facts["amenities"] = ", ".join(p.amenities)
    return PropertyKnowledge(facts=facts, selling_points=list(p.usps),
                             rera_number=(p.rera_number or "").strip())
