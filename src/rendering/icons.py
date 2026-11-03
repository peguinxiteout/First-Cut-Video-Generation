"""A small line-icon set, drawn inline.

A brochure generated offline has to look like one generated online, so the
icons are inline SVG strokes: no icon font, nothing fetched at render time,
no file to go missing. They take their colour from the text around them
(`currentColor`) and their size from the CSS, so one set serves all six
templates.

An unknown name draws a small dot — never the wrong picture.
"""
from __future__ import annotations

from markupsafe import Markup

_PATHS: dict[str, str] = {
    "pool": '<path d="M3 17c2 0 2-1.5 4-1.5S9 17 11 17s2-1.5 4-1.5 2 1.5 4 1.5 2-1 2-1"/>'
            '<path d="M3 21c2 0 2-1.5 4-1.5S9 21 11 21s2-1.5 4-1.5 2 1.5 4 1.5 2-1 2-1"/>'
            '<path d="M8 14V5a2 2 0 0 1 4 0M16 14V5a2 2 0 0 0-4 0M8 9h8"/>',
    "club": '<path d="M4 20V9l8-5 8 5v11"/><path d="M9 20v-6h6v6M3 20h18"/>',
    "fitness": '<path d="M4 9v6M7 7v10M17 7v10M20 9v6M7 12h10"/>',
    "yoga": '<circle cx="12" cy="5" r="2"/><path d="M12 7v6M6 11l6 2 6-2M8 20l4-7 4 7"/>',
    "spa": '<path d="M12 21c-5 0-8-3-8-7 3 0 6 1 8 4 2-3 5-4 8-4 0 4-3 7-8 7z"/><path d="M12 18c0-4 0-8 0-12M9 8c1-2 2-3 3-4 1 1 2 2 3 4"/>',
    "racquet": '<ellipse cx="14" cy="9" rx="6" ry="6"/><path d="M10 13l-6 7M11 6l6 6M9 9l6 6"/>',
    "ball": '<circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c3 3 3 13 0 16M12 4c-3 3-3 13 0 16"/>',
    "run": '<circle cx="15" cy="4.5" r="2"/><path d="M8 21l3-6 3 2v4M7 12l3-4 4 1 3 4h3M11 15l-1-4"/>',
    "cycle": '<circle cx="6" cy="16" r="4"/><circle cx="18" cy="16" r="4"/><path d="M6 16l5-8h4l3 8M11 8l2 8"/>',
    "kids": '<circle cx="9" cy="5" r="2"/><circle cx="16" cy="7" r="1.6"/><path d="M9 7v7l-3 6M9 14l3 6M5 10h8M16 9v5l-2 5M16 14l2 5"/>',
    "garden": '<path d="M12 21V11"/><path d="M12 11c0-4 3-7 7-7 0 4-3 7-7 7zM12 14c0-3-2.5-5-5.5-5 0 3 2.5 5 5.5 5z"/><path d="M5 21h14"/>',
    "leaf": '<path d="M5 19c0-8 6-14 14-14 0 8-6 14-14 14z"/><path d="M5 19L14 10"/>',
    "fire": '<path d="M12 3c1 4 5 5 5 10a5 5 0 0 1-10 0c0-3 2-4 2-7 1 1 2 2 3 4 0-3 0-5 0-7z"/><path d="M5 21h14"/>',
    "film": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18M3 15h18M8 5v14M16 5v14"/>',
    "book": '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5M9 7h6"/>',
    "cup": '<path d="M5 8h11v6a5 5 0 0 1-5 5h-1a5 5 0 0 1-5-5z"/><path d="M16 10h2a2 2 0 0 1 0 4h-2M8 3v2M12 3v2"/>',
    "dining": '<path d="M7 3v8M5 3v5a2 2 0 0 0 4 0V3M7 11v10M16 3c-2 1-3 4-3 7h3v11"/>',
    "security": '<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "parking": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M10 16V8h3a2.5 2.5 0 0 1 0 5h-3"/>',
    "plug": '<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4"/>',
    "lift": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 9l3-3 3 3M9 15l3 3 3-3"/>',
    "shop": '<path d="M4 8h16l-1.2 11a2 2 0 0 1-2 1.8H7.2a2 2 0 0 1-2-1.8z"/><path d="M9 8V6a3 3 0 0 1 6 0v2"/>',
    "work": '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M3 12h18"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M16 7l3 3M14 9l2 2"/>',
    "bench": '<path d="M3 11h18M5 11v8M19 11v8M4 7h16M3 15h18"/>',
    "paw": '<circle cx="7" cy="10" r="1.6"/><circle cx="10.5" cy="6.5" r="1.6"/><circle cx="14.5" cy="6.5" r="1.6"/><circle cx="18" cy="10" r="1.6"/><path d="M8 17c0-3 2-5 4-5s4 2 4 5c0 2-2 2-4 2s-4 0-4-2z"/>',
    "stage": '<path d="M3 20h18M5 20V9a7 4 0 0 1 14 0v11"/><path d="M9 20v-5h6v5"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2"/>',
    "moon": '<path d="M20 15A8 8 0 1 1 9 4a6.5 6.5 0 0 0 11 11z"/>',
    "water": '<path d="M12 3s5 5.5 5 9a5 5 0 0 1-10 0c0-3.5 5-9 5-9z"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
    "star": '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
    "step": '<circle cx="12" cy="12" r="9"/><path d="M10 8l4 4-4 4"/>',
    "phone": '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2z"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/>',
    "web": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>',
    "pin": '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "transit": '<rect x="5" y="3" width="14" height="14" rx="2"/><path d="M5 11h14"/><circle cx="8.5" cy="14" r="1"/><circle cx="15.5" cy="14" r="1"/><path d="M8 17l-2 4M16 17l2 4"/>',
    "airport": '<path d="M3 13l8-2 2-7 2 7 6 1.5-6 1.5-2 7-2-7-8-1.5z"/>',
    "school": '<path d="M12 4L2 9l10 5 10-5-10-5z"/><path d="M6 11.5V16c0 1.5 3 3 6 3s6-1.5 6-3v-4.5"/>',
    "hospital": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M12 8v8M8 12h8"/>',
    "road": '<path d="M7 21L10 3M17 21L14 3M12 5v2M12 11v2M12 17v2"/>',
    "landmark": '<path d="M3 21h18M5 21V11M19 21V11M9 21v-6h6v6M3 11l9-7 9 7z"/>',
    "map": '<path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2z"/><path d="M9 4v14M15 6v14"/>',
    "sofa": '<path d="M4 11V8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v3"/><path d="M3 11a2 2 0 0 1 2 2v2h14v-2a2 2 0 0 1 4 0v5H1v-5a2 2 0 0 1 2-2zM5 18v2M19 18v2"/>',
    "kitchen": '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M4 10h16M8 6h.01M12 6h.01M9 14v3"/>',
    "bed": '<path d="M3 18V7M3 13h18v5M21 18v-3a3 3 0 0 0-3-3h-8v1"/><circle cx="7" cy="10" r="2"/>',
    "bath": '<path d="M4 12h16v3a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5zM6 12V5a2 2 0 0 1 4 0M7 20l-1 2M17 20l1 2"/>',
    "people": '<circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.4"/><path d="M3 20c0-3.5 2.7-6 6-6s6 2.5 6 6M14 14.5c3.5-.5 7 1.5 7 5.5"/>',
    "building": '<path d="M4 21V5l8-2v18M12 7l8 2v12M3 21h18M7 8h2M7 12h2M7 16h2M15 12h2M15 16h2"/>',
    "price": '<path d="M7 5h10M7 9h10M9 5c4 0 5 4 0 4h-2l8 10"/>',
    "area": '<rect x="4" y="4" width="16" height="16"/><path d="M4 9h3M4 14h2M9 4v3M14 4v2M20 15h-3M20 10h-2"/>',
    "home": '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10M10 20v-6h4v6"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
    "sports": '<circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c3 3 3 13 0 16M12 4c-3 3-3 13 0 16"/>',
    "convenience": '<path d="M4 8h16l-1.2 11a2 2 0 0 1-2 1.8H7.2a2 2 0 0 1-2-1.8z"/><path d="M9 8V6a3 3 0 0 1 6 0v2"/>',
    "people2": '<circle cx="12" cy="7" r="3"/><path d="M5 21c0-4 3-7 7-7s7 3 7 7"/>',
    "dot": '<circle cx="12" cy="12" r="3"/>',
    "tag": '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.5"/>',
    "leaf2": '<path d="M5 19c0-8 6-14 14-14 0 8-6 14-14 14z"/>',
    "recycle": '<path d="M7 19H4l3-5M17 19h3l-3-5M9 5l3-3 3 3M12 2v7M5 14l3-5M19 14l-3-5M8 19h8"/>',
    "train": '<rect x="6" y="3" width="12" height="13" rx="3"/><path d="M6 10h12M9 20l-2 2M15 20l2 2M9 13h.01M15 13h.01"/>',
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z"/>',
    "bed2": '<path d="M3 18V7M3 13h18v5"/>',
}
_ALIASES = {"gym": "fitness", "kid": "kids", "green": "garden", "retail": "shop",
            "leisure": "stage", "configuration": "home", "possession": "calendar",
            "location": "pin", "property_type": "building", "project_size": "map",
            "rera": "security", "transport": "transit"}


def svg(name: str, cls: str = "ic") -> Markup:
    key = _ALIASES.get(name, name)
    paths = _PATHS.get(key, _PATHS["dot"])
    return Markup(f'<svg class="{cls}" viewBox="0 0 24 24" aria-hidden="true" fill="none" '
                  f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
                  f'stroke-linejoin="round">{paths}</svg>')


def names() -> list[str]:
    return sorted(_PATHS)
