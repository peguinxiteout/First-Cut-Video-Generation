"""Node 2, fact ledger: the facts a video may state.

The same ledger the brochure uses (agents/brochure/property_intelligence.py).
With a model it also reads the property's documents; without one it is built
from exactly what was typed in the form, and is never less truthful for it
(it only knows less). An offline run, or one with no model key, takes the
form ledger straight away instead of waiting out the model's retries.

CACHE. A model asked twice gives two ledgers, and so two scripts. A ledger a model made is
kept in data/ledger_cache/ under the property's data, the text of its documents, the model
and the prompt's version (a hash of the prompt and `CACHE_VERSION`), so the same property
gets the same ledger until one of those changes. A hit is said in the notes. The form ledger
(offline, no key, or a model that failed) is never kept: it is free to make again.
`refresh=True` (the CLI's --refresh-ledger) asks the model again and keeps the new answer.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.agents.brochure import property_intelligence
from src.agents.video.offline import model_blocker
from src.core.config import load_yaml_config, settings
from src.data.schemas import PropertyInput, PropertyKnowledge

logger = logging.getLogger(__name__)

CACHE_VERSION = 1          # raise it to have every ledger asked for again
ROLE = "property_intelligence"
CACHE_NOTE = "Fact ledger from the cache"
REFRESH_NOTE = "Fact ledger refreshed"


def cache_dir() -> Path:
    return settings.data_dir_resolved / "ledger_cache"


def model_name() -> str:
    """The ledger role's model, as the gateway would resolve it."""
    role = load_yaml_config("models.yaml")["roles"][ROLE]
    name = getattr(settings, role["model_env"].lower(), "") or role["model_env"]
    return f"{role['provider']}:{name}"


def prompt_version() -> str:
    template = load_yaml_config("brochure/prompts.yaml")["property_intelligence"]
    return f"{CACHE_VERSION}-{hashlib.sha256(template.encode('utf-8')).hexdigest()[:12]}"


def cache_key(prop: PropertyInput, documents_text: str, model: str, version: str) -> str:
    payload = json.dumps({"property": prop.model_dump_json(), "documents": documents_text or "", "model": model,
                          "prompt": version}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _load(path: Path) -> Optional[tuple[PropertyKnowledge, str]]:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        return PropertyKnowledge.model_validate(entry["knowledge"]), str(entry.get("created_at", ""))
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring an unreadable ledger cache entry: %s", path.name)
        return None


def _store(path: Path, knowledge: PropertyKnowledge, model: str, version: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": model, "prompt_version": version, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "knowledge": knowledge.model_dump(mode="json")}), encoding="utf-8")
    except OSError as exc:       # a cache that cannot be written is not worth a failed video
        logger.warning("The ledger could not be cached: %s", exc)


def cached(prop: PropertyInput, documents_text: str, directory: Optional[Path] = None) -> Optional[PropertyKnowledge]:
    """The ledger kept for this property, documents, model and prompt, or None. Never asks a model."""
    path = (directory if directory is not None else cache_dir()) /         f"{cache_key(prop, documents_text, model_name(), prompt_version())}.json"
    hit = _load(path) if path.exists() else None
    return hit[0] if hit else None


def run(prop: PropertyInput, documents_text: str, offline: bool, *, refresh: bool = False,
        directory: Optional[Path] = None) -> tuple[PropertyKnowledge, list[str]]:
    """The ledger, and what to tell the person about how it was made."""
    blocker = model_blocker(offline, ROLE)
    if blocker:
        notes = [f"Fact ledger built from the form ({blocker})."]
        if documents_text:
            notes.append("The property's documents were not read, so facts that only they state are not used.")
        return property_intelligence.ledger_from_form(prop), notes

    model, version = model_name(), prompt_version()
    path = (directory if directory is not None else cache_dir()) / f"{cache_key(prop, documents_text, model, version)}.json"
    if not refresh and path.exists():
        hit = _load(path)
        if hit is not None:
            knowledge, made = hit
            return knowledge, [f"{CACHE_NOTE} (made {made[:10] or 'earlier'} by {model}): the same property, documents, "
                               "model and prompt as then. --refresh-ledger asks the model again."]

    knowledge = property_intelligence.run(prop, documents_text)
    kept = knowledge != property_intelligence.ledger_from_form(prop)       # the model's answer, not the fallback it falls to
    if kept:
        _store(path, knowledge, model, version)
    if not refresh:
        return knowledge, ["Fact ledger read from the form and documents."]
    return knowledge, [f"{REFRESH_NOTE}: the model was asked again"
                       + (" and its answer kept." if kept else ", but it gave nothing the form does not; the ledger is the form's.")]
