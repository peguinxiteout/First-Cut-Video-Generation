"""What a photograph visibly shows, in one vocabulary shared with the script.

The vision step tags each picture with what is clearly in it (pool, lawn, lounge, bbq,
tower exterior ...). The script has amenities, selling points and chips in words
("Private jacuzzi", "BBQ and dining", "Swimming pool"). Both are turned into the names of
configs/video/vision.yaml (`tags` for amenities, `scenes` for kinds of view) so they can be
compared: a line is about a pool when its text says pool, and a picture shows one when it
is tagged pool.

    names()          the amenity names, which the vision prompt offers the model
    clean(tags)      a picture's tags as names: lower case, aliases merged, no duplicates
    needs(text)      the amenity names a text mentions (empty: it names no amenity)
    supports(text, tags)   does a picture with these tags show what the text names?

A word is matched whole (a plural too), longest phrase first, and a phrase that has been
read is not read again, so "pool table" is a lounge and not a pool, and "courtyard" is not a
court.

EVIDENCE. A picture shows an amenity only when the vision model tagged it AND the caption says
it, both read through this vocabulary (`evidence`). A tag the caption does not back is "weak":
the amenity may be in the frame (a pool in a corner of a tower picture) but a viewer would not
see it, so such a picture is never the picture for a line or a chip about it. It still serves as
a general, scene or exterior picture. The caption is read with a tighter list than a brochure's
amenity strings (`caption_ignore`, `caption_consume` in vision.yaml): "green trees" is not a
lawn and a home's "dining room" is not a dining amenity.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from src.core.config import load_yaml_config

logger = logging.getLogger(__name__)

CONFIG_FILE = "video/vision.yaml"


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[tuple[tuple[str, str, re.Pattern], ...], tuple[str, ...], tuple[str, ...]]:
    config = load_yaml_config(CONFIG_FILE)
    amenities = {name: list(words or []) for name, words in (config.get("tags") or {}).items()}
    scenes = {name: list(words or []) for name, words in (config.get("scenes") or {}).items()}
    for name, words in (config.get("synonyms") or {}).items():       # more words for a name, from what a camera sees
        if name in amenities or name in scenes:
            (amenities if name in amenities else scenes)[name] += list(words or [])
        else:
            logger.warning("vision.yaml: synonyms for '%s', which is not a tag or a scene; ignored", name)
    entries = []
    for group in (amenities, scenes):
        for name, words in group.items():
            for word in words or []:
                text = _norm(word)
                entries.append((name, text, re.compile(rf"(?<![a-z]){re.escape(text)}(?:s|es)?(?![a-z])")))
    for phrase in config.get("consume") or []:                # read as nothing, but not as a shorter phrase
        text = _norm(phrase)
        entries.append(("", text, re.compile(rf"(?<![a-z]){re.escape(text)}(?:s|es)?(?![a-z])")))
    entries.sort(key=lambda e: -len(e[1]))                    # the longest phrase first
    return tuple(entries), tuple(amenities), tuple(scenes)


@lru_cache(maxsize=1)
def _caption_vocabulary() -> tuple[tuple[str, str, re.Pattern], ...]:
    """The entries a caption is read with: the vocabulary without the words that read too much into
    a caption, and with the phrases a caption uses for something that is not an amenity."""
    config = load_yaml_config(CONFIG_FILE)
    ignored = {_norm(w) for w in config.get("caption_ignore") or []}
    consumed = list(config.get("caption_consume") or [])
    for phrases in (config.get("rooms") or {}).values():          # a home's "dining area" is a room, not the amenity
        consumed += list(phrases or [])
    hidden = ignored | {_norm(p) for p in consumed}
    entries = [e for e in _vocabulary()[0] if e[1] not in hidden]
    for phrase in consumed:
        text = _norm(phrase)
        entries.append(("", text, re.compile(rf"(?<![a-z]){re.escape(text)}(?:s|es)?(?![a-z])")))
    entries.sort(key=lambda e: -len(e[1]))
    return tuple(entries)


@lru_cache(maxsize=1)
def _rooms() -> tuple[tuple[tuple[str, re.Pattern], ...], dict, frozenset, tuple[re.Pattern, ...]]:
    """(the room phrases, longest first; the room of a stored category; the categories whose captions are
    read for a room; the phrases that are not a room of a home)."""
    config = load_yaml_config(CONFIG_FILE)
    phrases = [(room, _norm(p)) for room, words in (config.get("rooms") or {}).items() for p in words or []]
    phrases.sort(key=lambda e: -len(e[1]))
    entries = tuple((room, re.compile(rf"(?<![a-z]){re.escape(p)}(?:s|es)?(?![a-z])")) for room, p in phrases)
    nots = tuple(re.compile(rf"(?<![a-z]){re.escape(_norm(p))}(?:s|es)?(?![a-z])")
                 for p in config.get("not_a_room") or [])
    return (entries, dict(config.get("room_category") or {}), frozenset(config.get("room_caption_categories") or []),
            nots)


def room_order() -> list[str]:
    """The rooms of a home, in the order a script would show them (the order of the table)."""
    return list(load_yaml_config(CONFIG_FILE).get("rooms") or {})


def room(category: str, caption: str = "") -> str:
    """The room of a home a picture shows ("living", "bedroom" ...), or "". Read from the stored category
    and the caption (the vision prompt is not asked for it): a caption's first room phrase names the room
    in a picture of a home or of people; a picture of a home interior whose caption names none takes the
    room of its category; a picture of anything else (an exterior, an amenity) has none."""
    entries, by_category, caption_categories, nots = _rooms()
    category = (category or "").strip().lower()
    text = f" {_norm(caption)} "
    for pattern in nots:
        text = pattern.sub(" ", text)
    if category in caption_categories:
        found = [(m.start(), name) for name, pattern in entries for m in [pattern.search(text)] if m]
        if found:
            return min(found)[1]
    return by_category.get(category, "")


def is_view(topic: str, caption: str = "") -> bool:
    """Whether a topic or caption speaks of the whole place seen from above or afar (an aerial view)."""
    words = [_norm(w) for w in load_yaml_config(CONFIG_FILE).get("views") or []]
    text = f" {_norm(topic)} {_norm(caption)} "
    return any(re.search(rf"(?<![a-z]){re.escape(w)}", text) for w in words)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


@lru_cache(maxsize=1)
def _non_visual_patterns() -> tuple[re.Pattern, ...]:
    words = (load_yaml_config(CONFIG_FILE).get("not_visual") or [])
    return tuple(re.compile(rf"(?<![a-z]){re.escape(_norm(w))}(?:s|es)?(?![a-z])") for w in words if _norm(w))


def non_visual(text: str) -> bool:
    """True when a text names something no photograph can show (a utility, a safety system, a
    service) and no amenity of the vocabulary. Such a text needs no picture."""
    if needs(text):
        return False
    padded = f" {_norm(text)} "
    return any(p.search(padded) for p in _non_visual_patterns())


def names() -> list[str]:
    """The amenity names the vision model is asked to prefer."""
    return list(_vocabulary()[1])


def scene_names() -> list[str]:
    return list(_vocabulary()[2])


def _read(text: str, entries=None) -> list[str]:
    """The vocabulary names (amenities and scenes) in a text, in order of the phrases found."""
    entries = _vocabulary()[0] if entries is None else entries
    left, found = f" {_norm(text)} ", []
    for name, word, pattern in entries:
        if pattern.search(left):
            if name and name not in found:
                found.append(name)
            left = pattern.sub(" ", left)
    return found


def needs(text: str) -> set[str]:
    """The amenities a text names. A text that names none (the property's design, its
    location) returns an empty set: it needs no particular picture."""
    amenities = set(_vocabulary()[1])
    return {name for name in _read(text) if name in amenities}


def clean(tags, limit: int = 8) -> list[str]:
    """A picture's tags as vocabulary names where one fits, else the tag as plain lower
    case words, without duplicates, at most `limit`."""
    out: list[str] = []
    for tag in tags or []:
        if not isinstance(tag, str) or not _norm(tag):
            continue
        for name in _read(tag) or [_norm(tag)]:
            if name not in out:
                out.append(name)
    return out[:limit]


def from_caption(caption: str) -> list[str]:
    """The amenities a picture's caption speaks of, read with the caption's own list (see EVIDENCE).
    Views (exterior, interior) are not taken from a caption: a rooftop "outdoor kitchen" would read
    as an interior."""
    amenities = set(_vocabulary()[1])
    return [name for name in _read(caption or "", _caption_vocabulary()) if name in amenities]


def evidence(tags, caption: str = "", limit: int = 12) -> tuple[list[str], list[str]]:
    """(the tags a picture is used with, its weak tags). The model's tags come first, cleaned. A view
    (exterior, interior ...) or any tag that is not an amenity stands as it is; an amenity tag stands
    only when the caption says it too. The weak tags are the amenities the model tagged and the
    caption does not say. An amenity only the caption says is not a tag: the model did not see it."""
    amenities = set(_vocabulary()[1])
    said = set(from_caption(caption))
    cleaned = clean(tags, limit)
    shown = [t for t in cleaned if t not in amenities or t in said]
    weak = [t for t in cleaned if t in amenities and t not in said]
    return shown, weak


def shows(text: str, tags) -> list[str]:
    """The tags of a picture (cleaned) that show what `text` names. A text naming a
    vocabulary amenity needs that tag; a text naming something outside the vocabulary is
    shown only by a tag of the picture that appears in the text; a text that names nothing
    a picture could show is shown by nothing."""
    tags = list(tags or [])
    wanted = needs(text)
    if wanted:
        return sorted(wanted & set(tags))
    words = f" {_norm(text)} "
    scenes = set(_vocabulary()[2])
    return [t for t in tags if t not in scenes and f" {t} " in words]


def supports(text: str, tags) -> bool:
    return bool(shows(text, tags))


def is_scene(tags, scene: str) -> bool:
    return scene in set(tags or [])
