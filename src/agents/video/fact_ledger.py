"""Node 2, fact ledger: the facts a video may state.

The same ledger the brochure uses (agents/brochure/property_intelligence.py).
With a model it also reads the property's documents; without one it is built
from exactly what was typed in the form, and is never less truthful for it
(it only knows less). An offline run, or one with no model key, takes the
form ledger straight away instead of waiting out the model's retries.
"""
from __future__ import annotations

import logging

from src.agents.brochure import property_intelligence
from src.agents.video.offline import model_blocker
from src.data.schemas import PropertyInput, PropertyKnowledge

logger = logging.getLogger(__name__)


def run(prop: PropertyInput, documents_text: str, offline: bool) -> tuple[PropertyKnowledge, list[str]]:
    """The ledger, and what to tell the person about how it was made."""
    blocker = model_blocker(offline)
    if blocker:
        notes = [f"Fact ledger built from the form ({blocker})."]
        if documents_text:
            notes.append("The property's documents were not read, so facts that only they state are not used.")
        return property_intelligence.ledger_from_form(prop), notes
    return property_intelligence.run(prop, documents_text), ["Fact ledger read from the form and documents."]
