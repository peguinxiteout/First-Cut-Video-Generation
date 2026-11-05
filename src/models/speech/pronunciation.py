"""Respelling words for the voice.

A speech model reads a name the way its spelling suggests, which is not always
the way it is said. `respell` swaps the words a map names for the spelling
that makes the voice say them well. It is applied to the text handed to the
voice only: captions, titles and the closing card keep the real spelling.

Whole words, any case, longest key first (so "S.G. Highway" wins over
"Highway"). The map is configs/video/speech.yaml's `pronunciations`, with a
property's own (video_input.json) laid over it.
"""
from __future__ import annotations

import re


def merge(*maps: dict[str, str]) -> dict[str, str]:
    """Maps laid over one another, the later winning; keys compared without case."""
    merged: dict[str, tuple[str, str]] = {}
    for mapping in maps:
        for word, spoken in (mapping or {}).items():
            if word.strip() and str(spoken).strip():
                merged[word.strip().lower()] = (word.strip(), str(spoken).strip())
    return dict(merged.values())


def respell(text: str, mapping: dict[str, str]) -> str:
    """`text` with every word of `mapping` replaced by what it should be spoken as."""
    if not mapping:
        return text
    spoken = {word.lower(): value for word, value in mapping.items()}
    words = "|".join(re.escape(w) for w in sorted(spoken, key=len, reverse=True))
    # one pass, so a respelling is never itself respelled
    return re.sub(rf"(?<!\w)(?:{words})(?!\w)", lambda m: spoken[m.group(0).lower()], text, flags=re.IGNORECASE)
