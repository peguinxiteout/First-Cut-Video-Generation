"""Turns an ImageBrief into a model prompt.

Two problems this exists to solve, both visible in the team's first test
outputs (three near-identical white-marble American kitchens):

1. Style mismatch. A generated image placed beside warm, dusk-lit brick
   renders looks pasted in. We derive a style profile from what the
   Vision agent already saw in the uploaded photos (materials, light,
   palette) and condition every prompt on it.
2. The "AI look". Keyword soup like "8k, ultra-detailed, masterpiece"
   pushes models toward an over-sharpened, synthetic finish. We describe
   the shot the way a photographer or editorial art director would:
   lens, light, composition, restraint.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from src.data.schemas import ALLOWED_GENERATED_KINDS, ImageAnalysis, ImageBrief

# Vocabulary we look for in Vision captions/notes. Grounded: only words
# the vision model actually used about THIS property's photos.
_MATERIALS = ["brick", "oak", "walnut", "teak", "wood", "timber", "marble", "stone", "travertine",
              "concrete", "glass", "brass", "bronze", "linen", "terrazzo", "granite", "leather", "rattan"]
_LIGHT = ["dusk", "sunset", "golden hour", "evening", "morning", "daylight", "twilight", "warm light",
          "ambient", "soft light", "night"]
_PALETTE = ["warm", "neutral", "earthy", "terracotta", "beige", "cream", "muted", "white", "grey",
            "green", "sage", "charcoal"]

# Subjects that would depict the actual property. Rejected outright.
_FORBIDDEN = re.compile(
    r"\b(floor\s*plan|site\s*plan|master\s*plan|layout\s*plan|blueprint|facade of|the (?:tower|building|"
    r"property|project)\b|this (?:tower|building|property|villa|apartment)|our (?:tower|building|villa)|"
    r"logo|brochure cover|text|signage|lettering)\b",
    re.IGNORECASE,
)

NEGATIVE_PROMPT = (
    "text, lettering, watermark, logo, signature, frame, border, collage, split screen, "
    "cartoon, illustration, 3d render look, plastic, oversaturated, HDR halos, over-sharpened, "
    "fisheye distortion, warped lines, deformed hands, distorted faces, crowd, clutter, low quality"
)

_ASPECT_SIZES = {
    "landscape": (1536, 1024),
    "wide": (1600, 900),      # 16:9 — blog hero
    "portrait": (1024, 1365),  # 3:4
    "square": (1152, 1152),
    # The brochure page's own proportions (1240x1754). A generated image at
    # this aspect fills a page edge to edge with no cropping.
    "page": (1024, 1448),
}


@dataclass
class StyleProfile:
    medium: str = "photograph"
    materials: list[str] = field(default_factory=list)
    light: list[str] = field(default_factory=list)
    palette: list[str] = field(default_factory=list)
    region: str = ""

    def phrase(self) -> str:
        parts = []
        if self.palette:
            parts.append(f"{' and '.join(self.palette[:2])} palette")
        if self.materials:
            parts.append(f"{', '.join(self.materials[:3])} textures")
        if self.light:
            parts.append(f"{self.light[0]}")
        return ", ".join(parts)


def _top_terms(text: str, vocab: list[str], n: int) -> list[str]:
    counts = Counter()
    for term in vocab:
        hits = len(re.findall(r"\b" + re.escape(term) + r"\b", text))
        if hits:
            counts[term] = hits
    return [t for t, _ in counts.most_common(n)]


def dominant_medium(analyses: list[ImageAnalysis]) -> str:
    """The medium most of the property's own images use. An AI image in a
    different medium is what made the review's 'Everyday Living' page look
    like stock among CGI renders."""
    counts = Counter(a.medium for a in analyses if getattr(a, "medium", ""))
    return counts.most_common(1)[0][0] if counts else "photograph"


def derive_style(analyses: list[ImageAnalysis], region: str = "", medium_override: str = "auto") -> StyleProfile:
    corpus = " ".join(
        f"{a.suggested_caption} {a.notes} {a.scene_tag}".lower() for a in analyses
    )
    return StyleProfile(
        medium=medium_override if medium_override in ("cgi_render", "photograph", "drawing")
        else dominant_medium(analyses),
        materials=_top_terms(corpus, _MATERIALS, 3),
        light=_top_terms(corpus, _LIGHT, 1),
        palette=_top_terms(corpus, _PALETTE, 2),
        region=region,
    )


def is_allowed(brief: ImageBrief) -> bool:
    if brief.kind not in ALLOWED_GENERATED_KINDS:
        return False
    return not _FORBIDDEN.search(f"{brief.subject} {brief.purpose}")


def size_for(aspect: str) -> tuple[int, int]:
    return _ASPECT_SIZES.get(aspect, _ASPECT_SIZES["landscape"])


_KIND_DIRECTION = {
    "lifestyle": "editorial lifestyle photograph, natural candid moment, figures small or seen from behind, "
                 "shot on a 35mm lens at eye level",
    "amenity_concept": "architectural interior photograph, calm and uncluttered, shot on a 24mm tilt-shift lens, "
                       "straight verticals",
    "neighbourhood": "wide cityscape photograph, atmospheric and restrained, shot on a 50mm lens",
    "design_element": "abstract architectural detail photograph, close crop of surface texture and shadow, "
                      "minimal composition with generous negative space, no identifiable building",
    "blog_hero": "editorial magazine cover-style photograph, strong single focal point, shot on a 35mm lens, "
                 "generous negative space for a headline",
}


# Rendering language per medium. Applied so a generated image matches how
# the property's own images were made.
_MEDIUM_DIRECTION = {
    "cgi_render": ("architectural visualisation render, physically based rendering, clean CGI lighting, "
                   "immaculate surfaces, precise straight lines"),
    "photograph": "photograph, soft natural light, true-to-life colour, subtle film grain",
    "drawing": "photograph, soft natural light, true-to-life colour",
}


def build_prompt(brief: ImageBrief, style: StyleProfile) -> str:
    direction = _KIND_DIRECTION.get(brief.kind, _KIND_DIRECTION["lifestyle"])
    if style.medium == "cgi_render":
        # Drop photographic cues that fight the CGI look.
        direction = direction.replace("editorial lifestyle photograph", "architectural visualisation")
        direction = direction.replace("photograph", "visualisation")
    region = f", set in {style.region}" if style.region and brief.kind in ("lifestyle", "neighbourhood",
                                                                        "blog_hero") else ""
    style_phrase = style.phrase()
    pieces = [
        f"{brief.subject.strip().rstrip('.')}{region}",
        direction,
        style_phrase,
        _MEDIUM_DIRECTION.get(style.medium, _MEDIUM_DIRECTION["photograph"]),
        "high-end real estate editorial",
    ]
    return ", ".join(p for p in pieces if p)
