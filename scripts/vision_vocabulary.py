#!/usr/bin/env python
"""List the amenity names of every property and what the photo tag vocabulary does not yet name.

    python scripts/vision_vocabulary.py

Reads data/properties/*/property.json (nothing is written) and prints the union of the
amenity strings, how many of them name an amenity of configs/video/vision.yaml, which are
things no photograph can show (`not_visual`), and which are neither. Extend `tags`,
`synonyms` or `not_visual` in that file with what is left; a new tag NAME changes the
vision prompt, so each property's pictures are read again on its next render.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.video import photo_tags  # noqa: E402
from src.core.config import settings  # noqa: E402


def amenity_names(properties_dir: Path) -> dict[str, list[str]]:
    """{amenity string: the properties that list it}, from every property.json under the folder."""
    found: dict[str, list[str]] = {}
    for path in sorted(properties_dir.glob("*/property.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        amenities = data.get("amenities") if isinstance(data, dict) else None
        if isinstance(amenities, str):                      # the form stores the list as JSON text
            try:
                amenities = json.loads(amenities)
            except ValueError:
                amenities = [amenities]
        for item in amenities or []:
            if isinstance(item, str) and item.strip():
                found.setdefault(item.strip(), []).append(path.parent.name)
    return found


def classify(names: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(named by the vocabulary, not visual, neither)."""
    named = [n for n in names if photo_tags.needs(n)]
    hidden = [n for n in names if n not in named and photo_tags.non_visual(n)]
    rest = [n for n in names if n not in named and n not in hidden]
    return named, hidden, rest


def main() -> None:
    folder = settings.data_dir_resolved / "properties"
    found = amenity_names(folder)
    named, hidden, rest = classify(sorted(found))
    print(f"{len(found)} distinct amenity strings in {len(list(folder.glob('*/property.json')))} properties")
    print(f"  {len(named)} name an amenity of the vocabulary, {len(hidden)} are not visual, {len(rest)} are neither")
    for title, items in (("Not visual", hidden), ("Neither (extend vision.yaml)", rest)):
        if items:
            print(f"\n{title}:")
            for item in items:
                print(f"  {item}   [{', '.join(found[item])}]")


if __name__ == "__main__":
    main()
