from enum import Enum


class ImageCategory(str, Enum):
    EXTERIOR = "exterior"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    AMENITY = "amenity"
    LIFESTYLE = "lifestyle"
    FLOORPLAN = "floorplan"
    LOGO = "logo"
    # These three are NOT photographs — they're already-composed pieces of
    # information (an icon list, a scanned spec sheet, a location map), and
    # the planner never sets them in a grid of photographs.
    INFOGRAPHIC = "infographic"  # an icon list / diagram that's already a complete visual unit (e.g. "6 amenity icons in one graphic")
    DOCUMENT = "document"        # a scanned/screenshotted text-heavy page (specifications, terms, disclaimers)
    MAP = "map"                  # a location/site map
    OTHER = "other"


class ImageRole(str, Enum):
    HERO = "hero"
    SECONDARY_HERO = "secondary_hero"
    GALLERY = "gallery"
    AMENITIES = "amenities"
    LIFESTYLE = "lifestyle"
    FLOORPLAN = "floorplan"
    UNUSED = "unused"


class QualityGateResult(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    FAIL = "fail"


class ModelRole(str, Enum):
    PROPERTY_INTELLIGENCE = "property_intelligence"
    VISION = "vision"
    COPYWRITER = "copywriter"
    CREATIVE_DIRECTOR = "creative_director"
    CRITIC = "critic"
    FALLBACK = "fallback"
    BLOG_PLANNER = "blog_planner"
    BLOG_WRITER = "blog_writer"
    BLOG_SEO = "blog_seo"
    BLOG_CRITIC = "blog_critic"
    IMAGE_GAP = "image_gap"
    VISUAL_DIRECTOR = "visual_director"
    # Narrative Director: decides the brochure's story arc and what
    # each page is for, before any layout is chosen.
    NARRATIVE = "narrative"
    # Blog Research Agent: reads the topic and decides WHAT to look up —
    # which place, which searches. The tools then do the looking.
    BLOG_RESEARCH = "blog_research"
    # Blog Fact Check Agent: judges whether each claim in the draft is
    # supported by the author's facts or by what research actually found.
    BLOG_FACT_CHECK = "blog_fact_check"
