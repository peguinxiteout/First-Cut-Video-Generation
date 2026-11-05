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


def _read(text: str) -> list[str]:
    """The vocabulary names (amenities and scenes) in a text, in order of the phrases found."""
    entries, _, _ = _vocabulary()
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
    """The amenities a picture's caption speaks of: text matching on the reading the vision model
    already made, no second call. Views (exterior, interior) are not taken from a caption: a rooftop
    "outdoor kitchen" would read as an interior."""
    amenities = set(_vocabulary()[1])
    return [name for name in _read(caption or "") if name in amenities]


def merge(tags, caption: str = "", limit: int = 12) -> list[str]:
    """A picture's cleaned tags, then what its caption adds, without duplicates. The model's own
    tags come first; `limit` is higher than the model's own cap because captions add to it."""
    out = clean(tags, limit)
    for name in from_caption(caption):
        if name not in out:
            out.append(name)
    return out[:limit]


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
