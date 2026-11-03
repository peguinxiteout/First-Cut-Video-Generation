"""What a photograph is OF, and which amenities belong beside it.

    "Use the information contextually throughout the brochure. For example,
    place relevant clubhouse amenities alongside clubhouse images rather
    than creating a separate amenities page."

So photographs and amenities are sorted into the same small set of
TOPICS — pool, clubhouse, fitness, sports, kids, landscape, … — and a page
built around a clubhouse photograph carries the clubhouse's own amenities
as its supporting list. What no photograph claims is gathered onto the
"Everything on site" page, grouped by the same topics, with the same icons.

A topic is read from what the Vision Agent said about a photograph (its
category, scene tag and suggested title) and from the words of an amenity
the person typed. Both are matched against one lexicon, so the two can
never disagree about what "squash court" is.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# topic -> (display name, icon, keywords)
TOPICS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "pool":      ("Pool & water", "pool", ("pool", "swim", "aqua", "jacuzzi", "water body",
                                           "splash", "deck", "lap pool", "infinity")),
    "clubhouse": ("Clubhouse", "club", ("club", "lounge", "billiard", "snooker", "indoor game",
                                        "party", "banquet", "multipurpose hall", "multi-purpose hall",
                                        "library", "theatre", "theater", "cafe", "café", "coffee",
                                        "restaurant", "business centre", "business center",
                                        "co-working", "coworking", "guest room", "card room",
                                        "chess", "carrom", "table tennis", "pool table", "mini theatre",
                                        "hobby", "games room", "game room")),
    "fitness":   ("Health & fitness", "fitness", ("gym", "fitness", "yoga", "spa", "sauna", "steam",
                                                  "aerobic", "meditation", "zumba", "pilates",
                                                  "wellness", "salon", "massage", "reflexology",
                                                  "crossfit")),
    "sports":    ("Sports", "sports", ("court", "tennis", "badminton", "basketball", "squash",
                                       "cricket", "football", "futsal", "volleyball", "skating",
                                       "jogging", "running track", "cycling", "cycle track",
                                       "sports", "golf", "putting", "pickleball", "box cricket",
                                       "activity field")),
    "kids":      ("For children", "kids", ("kid", "child", "toddler", "creche", "crèche", "play area",
                                           "play zone", "playground", "tot lot", "sand pit",
                                           "day care", "daycare")),
    "landscape": ("Gardens & open spaces", "garden", ("garden", "park", "landscape", "lawn",
                                                      "courtyard", "trail", "walkway", "promenade",
                                                      "plaza", "pergola", "gazebo", "amphitheatre",
                                                      "amphitheater", "orchard", "green", "tree",
                                                      "seating", "bbq", "barbecue", "picnic",
                                                      "bonfire", "fire pit", "senior", "pet",
                                                      "open space", "boulevard", "water feature",
                                                      "fountain")),
    "care":      ("Health & childcare", "hospital", ("health center", "health centre", "clinic",
                                                     "daycare by", "medical", "pharmacy", "doctor")),
    "security":  ("Safety & security", "security", ("security", "cctv", "surveillance", "gated",
                                                    "intercom", "video door", "fire", "guard",
                                                    "access control", "boom barrier")),
    "convenience": ("Everyday convenience", "convenience", ("parking", "ev charging", "ev charger",
                                                            "power backup", "power back-up",
                                                            "backup", "lift", "elevator",
                                                            "store", "shop", "retail", "atm",
                                                            "laundry", "rainwater", "solar",
                                                            "water treatment", "stp", "wtp",
                                                            "piped gas", "wifi", "wi-fi",
                                                            "concierge", "drop-off", "drop off",
                                                            "visitor", "maintenance")),
    # Residences and architecture: photographs, rarely amenities.
    "exterior":  ("Architecture", "building", ("exterior", "facade", "façade", "elevation", "tower",
                                               "building", "development", "skyline")),
    "aerial":    ("The masterplan", "map", ("aerial", "bird", "site view", "drone", "overview")),
    "night":     ("After dark", "moon", ("night", "dusk", "evening", "twilight")),
    "arrival":   ("Arrival", "key", ("entrance", "entry", "gate", "arrival", "lobby", "porte",
                                     "foyer", "reception", "drop")),
    "living":    ("Living spaces", "sofa", ("living", "family room", "drawing room", "interior")),
    "dining":    ("Dining", "dining", ("dining",)),
    "kitchen":   ("Kitchen", "kitchen", ("kitchen", "pantry", "modular")),
    "bedroom":   ("Bedrooms", "bed", ("bedroom", "master suite", "bed room", "wardrobe")),
    "bathroom":  ("Bathrooms", "bath", ("bath", "toilet", "shower", "vanity", "washroom")),
    "balcony":   ("Balconies & views", "sun", ("balcony", "terrace", "sit-out", "sitout", "view",
                                               "veranda", "deck")),
    "lifestyle": ("Community life", "people", ("family", "couple", "people", "community", "friends",
                                               "children playing", "lifestyle", "celebration",
                                               "festival")),
}

# Topics an AMENITY may belong to. The residence topics are for photographs.
AMENITY_TOPICS = ("pool", "clubhouse", "fitness", "sports", "kids", "landscape",
                  "care", "security", "convenience")
RESIDENCE_TOPICS = ("living", "dining", "kitchen", "bedroom", "bathroom", "balcony")

# Vision categories that are not photographs to be arranged with others.
DRAWINGS = ("floorplan", "map", "document", "infographic", "logo")


def _norm(text: str) -> str:
    return re.sub(r"[_\-]+", " ", (text or "").lower())


def match(text: str, allowed: Optional[Iterable[str]] = None) -> str:
    """The topic whose keywords appear in this text, or ''. Longer keywords
    win, so 'table tennis' is a clubhouse game and not a sports court."""
    words = _norm(text)
    if not words:
        return ""
    best, best_len = "", 0
    for topic, (_label, _icon, keys) in TOPICS.items():
        if allowed is not None and topic not in allowed:
            continue
        for key in keys:
            if len(key) > best_len and re.search(r"(?<![a-z])" + re.escape(key), words):
                best, best_len = topic, len(key)
    return best


def of_image(analysis, filename: str = "") -> str:
    """A photograph's topic, from what the Vision Agent saw in it."""
    if analysis is None:
        return match(filename) or "other"
    category = (getattr(analysis.category, "value", None) or str(analysis.category or "")).lower()
    if category in DRAWINGS:
        return category
    text = " ".join(filter(None, (getattr(analysis, "scene_tag", ""),
                                  getattr(analysis, "suggested_title", ""),
                                  filename)))
    by_category = {"living_room": "living", "bedroom": "bedroom", "kitchen": "kitchen",
                   "bathroom": "bathroom"}
    if category in by_category:
        # A living-room photograph that is plainly a dining room is still a
        # residence, but the scene tag is the better word for it.
        found = match(text, RESIDENCE_TOPICS)
        return found or by_category[category]
    found = match(text)
    if found:
        return found
    if category == "exterior":
        return "exterior"
    if category == "lifestyle":
        return "lifestyle"
    if category == "amenity":
        return "landscape"
    return "other"


def of_amenity(name: str) -> str:
    return match(name, AMENITY_TOPICS) or "other"


def label(topic: str) -> str:
    return TOPICS.get(topic, ("Amenities", "", ()))[0]


def icon(topic: str) -> str:
    return TOPICS.get(topic, ("", "dot", ()))[1]


def icon_for_amenity(name: str) -> str:
    """The icon for one amenity: by its own words first (a 'squash court'
    is a racquet, not a generic sports mark), then by its topic."""
    words = _norm(name)
    specific = (("pool", "pool"), ("swim", "pool"), ("gym", "fitness"), ("yoga", "yoga"),
                ("spa", "spa"), ("tennis", "racquet"), ("badminton", "racquet"),
                ("squash", "racquet"), ("basketball", "ball"), ("football", "ball"),
                ("cricket", "ball"), ("jog", "run"), ("track", "run"), ("cycl", "cycle"),
                ("kid", "kids"), ("child", "kids"), ("play", "kids"), ("garden", "garden"),
                ("park", "garden"), ("lawn", "garden"), ("bbq", "fire"), ("barbecue", "fire"),
                ("bonfire", "fire"), ("theatre", "film"), ("theater", "film"),
                ("library", "book"), ("cafe", "cup"), ("café", "cup"), ("coffee", "cup"),
                ("restaurant", "dining"), ("banquet", "dining"), ("party", "dining"),
                ("cctv", "security"), ("security", "security"), ("parking", "parking"),
                ("ev ", "plug"), ("power", "plug"), ("lift", "lift"), ("elevator", "lift"),
                ("shop", "shop"), ("store", "shop"), ("retail", "shop"), ("billiard", "club"),
                ("game", "club"), ("work", "work"), ("guest", "key"), ("senior", "bench"),
                ("pet", "paw"), ("amphithe", "stage"), ("solar", "sun"), ("rain", "water"))
    for key, mark in specific:
        if key in words:
            return mark
    return icon(of_amenity(name))


def landmark_icon(category: str, name: str = "") -> str:
    words = " " + _norm(f"{category} {name}") + " "
    for key, mark in (("metro", "transit"), ("rail", "transit"), ("station", "transit"),
                      (" bus ", "transit"), ("hotel", "bed"), ("tech", "work"), ("transit", "transit"), ("airport", "airport"),
                      ("school", "school"), ("college", "school"), ("university", "school"),
                      ("hospital", "hospital"), ("clinic", "hospital"), ("health", "hospital"),
                      ("mall", "shop"), ("retail", "shop"), ("market", "shop"),
                      ("shop", "shop"), ("it park", "work"), ("tech", "work"),
                      ("office", "work"), ("work", "work"), ("business", "work"),
                      ("highway", "road"), ("road", "road"), ("expressway", "road"),
                      ("park", "garden"), ("lake", "water"), ("temple", "landmark"),
                      ("leisure", "stage")):
        if key in words:
            return mark
    return "pin"
