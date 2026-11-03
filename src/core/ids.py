"""Turning what a person typed into a name a filesystem will accept.

A property name and a blog topic both become DIRECTORY NAMES — the brief
is written to `data/properties/<id>/` or `data/blogs/<id>/` and read back
from there. So the length of the slug is a filesystem constraint rather
than a matter of taste, and it is one the slug did not use to have.

Windows resolves a path against a 260-character limit. A blog topic is a
sentence, and a real one —

    "Whitefield vs Sarjapur Road vs Electronic City Bengaluru: a practical
     comparison for homebuyers and property investors covering
     connectivity, pricing, infrastructure, lifestyle, rental potential
     and future development"

— slugs to 197 characters. Under a project folder 75 characters deep that
is 272, and `mkdir()` fails with

    FileNotFoundError: [WinError 3] The system cannot find the path
    specified: 'D:\\...\\data\\blogs\\WHITEFIELD-VS-SARJAPUR-ROAD-VS-...'

which is the confusing part: Windows reports a path too long to create as
a path that cannot be found, naming a directory that looks perfectly
valid. Nothing is written, and the failure lands in the UI before the
workflow starts.

So a slug is cut to length here. The digest that goes on the end when it
is cut is not decoration — see `slugify`.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Long enough to stay readable in a file browser, short enough that a
# deep project folder, this directory and the file names inside it all
# fit inside 260 characters with room to spare.
SLUG_MAX_CHARS = 56

_NOT_SLUG = re.compile(r"[^a-zA-Z0-9]+")


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8].upper()


def slugify(text: str, fallback: str = "ITEM") -> str:
    """A filesystem-safe, stable id for a property name or a blog topic.

    The same input always gives the same id, which is what lets a person
    regenerate a property and have it land in the folder it already has.

    Two cases need a digest on the end, and both are collisions that would
    silently overwrite one brief with another:

    * a name cut to length — comparison posts about the same three
      localities open identically and would truncate to the same id;
    * a name this strips to nothing, such as a topic written in
      Devanagari, where otherwise every one of them is called "BLOG".
    """
    original = text.strip()
    cleaned = _NOT_SLUG.sub("-", original).strip("-").upper()
    if not cleaned:
        return f"{fallback}-{_digest(original)}" if original else fallback
    if len(cleaned) <= SLUG_MAX_CHARS:
        return cleaned
    # Cut at a word boundary where there is one, so the id still reads.
    head = cleaned[:SLUG_MAX_CHARS].rsplit("-", 1)[0] or cleaned[:SLUG_MAX_CHARS]
    return f"{head}-{_digest(original)}"


def brief_dir(root: Path, slug: str) -> Path:
    """`root / slug`, but only when `slug` really is one.

    The HTTP API takes a property_id or a blog_id from the caller and joins
    it onto the data directory to find the brief. An id is produced by
    `slugify` and is therefore letters, digits and hyphens — but nothing
    was checking that, so "../../.." resolved to a directory outside the
    data root and the API would happily read a brief from it.

    What it checks is that the id names something directly INSIDE `root`,
    which is the actual boundary. It deliberately does not insist the id
    look like `slugify` output: a folder put there by hand may well be
    called "Sky Villas 128" or "my_property", and those are perfectly good
    directories that the person has every right to generate from. A guard
    that rejected them would trade a traversal bug for a broken feature.

    Raises ValueError, which the caller turns into a 400."""
    name = (slug or "").strip()
    if not name or name.startswith(".") or any(c in name for c in "/\\\0"):
        raise ValueError(f"not a valid id: {slug!r}")
    root = root.resolve()
    resolved = (root / name).resolve()
    if resolved.parent != root or resolved == root:
        raise ValueError(f"not a valid id: {slug!r}")
    return resolved
