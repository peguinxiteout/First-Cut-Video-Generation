"""What each category of brochure SAYS, and in which order — its positioning.

    Affordable  value: the price and the payment benefits lead, then the
                homes and their plans, the everyday amenities, connectivity.
                Clear, warm, practical; short sentences.
    Mid-Tier    lifestyle: the project's story and its distinctions lead
                (sustainability, privacy, design), then the amenities, the
                interiors, the plan, and the residences; price is a
                supporting fact.
    Luxury      architecture and exclusivity: the narrative, the
                distinctions, the architecture and the leisure it holds, the
                interiors; residences without prices; few, chosen words.

A Positioning is read by the planner (which pages, in what order, how many
photographs a page carries), the copywriter (voice, register, what to leave
unsaid) and the templates (whether a price is printed at all). Nothing here
is a fact about a property.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Positioning:
    category: str
    label: str
    promise: str                       # the one thing the book sells
    pillars: tuple[str, ...]
    voice: str                         # brief for the copy model
    avoid: tuple[str, ...]             # words that break the register
    show_price: bool                   # prices on the cover and in the schedule
    book: tuple[str, ...]              # the page kinds, in order
    max_pages: int
    max_feature_images: int            # photographs on one feature page
    fill: float                        # share of a text slot the copy aims to fill
    paragraphs: int                    # running-copy paragraphs on a narrative page


AFFORDABLE = Positioning(
    category="affordable", label="Affordable",
    promise="a well-planned home you can own sooner, with everything a family needs close by",
    pillars=("value for money", "practical layouts", "everyday convenience", "connectivity",
             "a secure community"),
    voice=("Clear, warm and practical. Lead with the benefit to the family and the value of "
           "the home. Short sentences, concrete nouns, no jargon. Name prices, savings and "
           "subsidies exactly as given. Speak to first-time buyers and growing families."),
    avoid=("opulent", "palatial", "exclusive enclave", "bespoke", "ultra-luxury", "elite",
           "rarefied", "world-class"),
    show_price=True,
    book=("cover", "overview", "highlights", "residences", "plans", "masterplan", "amenities",
          "feature", "location", "closing"),
    max_pages=11, max_feature_images=3, fill=0.86, paragraphs=2,
)

MID = Positioning(
    category="mid", label="Mid-Tier",
    promise="a balanced, considered life — comfort at home, everything for leisure downstairs",
    pillars=("lifestyle", "comfort", "sustainability", "convenience", "balanced value"),
    voice=("Confident, warm and lifestyle-led. Describe the day a resident has here and the "
           "thinking behind it — sustainability, privacy, light. Balanced, never boastful; one "
           "idea per sentence. Price is a supporting fact, not the headline."),
    avoid=("cheap", "budget", "palatial", "ultra-luxury", "world-class", "best-in-class"),
    show_price=True,
    book=("cover", "overview", "figures", "highlights", "feature", "amenities", "gallery",
          "masterplan", "residences", "plans", "location", "closing"),
    max_pages=12, max_feature_images=3, fill=0.78, paragraphs=2,
)

LUXURY = Positioning(
    category="luxury", label="Luxury",
    promise="an address few will share",
    pillars=("architecture", "exclusivity", "privacy", "craft", "experience"),
    voice=("Restrained, assured and sensory. Few words, each chosen. Speak of architecture, "
           "light, privacy, craft and the experience of arriving home — never of price, "
           "discounts or value-for-money. No exclamation marks, no stacked superlatives, no "
           "clichés such as 'world-class' or 'luxurious lifestyle'."),
    avoid=("affordable", "budget", "value for money", "cheap", "deal", "offer", "discount",
           "world-class", "best-in-class", "luxurious lifestyle", "!", "massive", "huge"),
    show_price=False,
    book=("cover", "overview", "highlights", "feature", "amenities", "gallery", "masterplan",
          "residences", "plans", "location", "closing"),
    max_pages=12, max_feature_images=2, fill=0.66, paragraphs=2,
)

POSITIONINGS = {p.category: p for p in (AFFORDABLE, MID, LUXURY)}


def for_category(category: Optional[str]) -> Positioning:
    from src.agents.brochure.catalog import normalise_category

    return POSITIONINGS.get(normalise_category(category) or "mid", MID)
