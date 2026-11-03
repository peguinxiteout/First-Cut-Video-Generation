"""The three voices — titles, framing sentences and transitions per category.

Every line here is a FRAME: it carries tone, never a fact. Facts (names,
numbers, places, amenities, prices) enter only through the slots the
copywriter fills from the content model (facts.py), and a frame whose slot
is empty is not used. Several frames exist for each job so two projects in
the same category do not read alike; which one a project gets is decided by
a hash of its name, so a project's copy is stable from run to run.

    affordable  practical, warm, family-first; value, savings, access
    mid         lifestyle-led, considered; the day a resident has here
    luxury      restrained, sensory; architecture, privacy, craft
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# themes: what a project leads with, read from its selling points
# ---------------------------------------------------------------------------

THEME_PHRASE = {
    "affordable": {
        "green": "lower running costs from greener design", "connectivity": "an easy daily commute",
        "offer": "real savings on the way in", "space": "layouts that waste nothing",
        "design": "smart, practical design", "leisure": "everyday leisure inside the gates",
        "quality": "construction you can trust", "privacy": "a safe, secure community",
        "general": "everything a family needs close by",
    },
    "mid": {
        "green": "a lighter footprint", "connectivity": "an easy commute in every direction",
        "offer": "thoughtful value", "space": "generous open space",
        "design": "architecture with real character", "leisure": "leisure at the doorstep",
        "quality": "lasting quality", "privacy": "a more private way of life",
        "general": "a considered way of living",
    },
    "luxury": {
        "green": "a quiet respect for the land", "connectivity": "the city within easy reach",
        "offer": "rare privileges", "space": "space held generously in reserve",
        "design": "architecture with a sense of place", "leisure": "leisure composed as an art",
        "quality": "uncompromising craft", "privacy": "privacy by design",
        "general": "an address apart",
    },
}

OVERVIEW_TITLES = {
    "affordable": {
        "offer": ["A smarter start for your family", "Your first home, made simpler"],
        "connectivity": ["Your home on the growth corridor", "Well placed for every day"],
        "green": ["Greener homes, lower bills", "A healthier home for less"],
        "default": ["Homes that make sense", "A home your family will love"],
    },
    "mid": {
        "green": ["Living lighter, living better", "A greener way to come home"],
        "privacy": ["Room to breathe, and to be", "Designed for quiet"],
        "design": ["Considered from the ground up", "Design you can live in"],
        "default": ["A considered way to live", "Everyday life, elevated"],
    },
    "luxury": {
        "design": ["Architecture with a sense of place", "A classical idea, reimagined"],
        "space": ["Space, held in reserve", "Room for the extraordinary"],
        "green": ["Quiet by nature", "In step with the land"],
        "default": ["An address apart", "Few will share it"],
    },
}

KICKERS = {
    "overview": {"affordable": "The project", "mid": "Overview", "luxury": "The vision"},
    "highlights": {"affordable": "Why buy here", "mid": "Distinctions", "luxury": "Distinctions"},
    "figures": {"affordable": "Green living", "mid": "Sustainability", "luxury": "Sustainability"},
    "amenities": {"affordable": "Amenities", "mid": "Lifestyle", "luxury": "Leisure"},
    "gallery": {"affordable": "Inside your home", "mid": "Interiors", "luxury": "Interiors"},
    "masterplan": {"affordable": "Masterplan", "mid": "Masterplan", "luxury": "The masterplan"},
    "residences": {"affordable": "Homes & prices", "mid": "Residences", "luxury": "Residences"},
    "plans": {"affordable": "Floor plans", "mid": "Floor plans", "luxury": "Plans"},
    "location": {"affordable": "Location", "mid": "Location", "luxury": "Location"},
    "closing": {"affordable": "Get in touch", "mid": "Visit", "luxury": "Private viewing"},
}

TITLES = {
    "highlights": {
        "affordable": ["Why families choose {short}", "Good reasons to say yes", "More home for your money"],
        "mid": ["What sets it apart", "Thoughtful by design", "The difference, in detail"],
        "luxury": ["Distinctions", "What makes it rare", "Singular by design"],
    },
    "figures": {
        "affordable": ["Green homes that save", "Good for you, good for the planet"],
        "mid": ["Net zero, by design", "Greener by design"],
        "luxury": ["Quietly sustainable", "Built in step with nature"],
    },
    "amenities": {
        "affordable": ["Everything your family needs, inside", "More to do, every day"],
        "mid": ["Leisure, a few steps away", "Weekends begin downstairs"],
        "luxury": ["The art of leisure", "Leisure, composed"],
    },
    "gallery": {
        "affordable": ["Smart spaces for everyday life", "Room for the whole family"],
        "mid": ["Rooms that work as hard as you do", "Light, space and calm"],
        "luxury": ["Interiors of quiet assurance", "Light, proportion, calm"],
    },
    "masterplan": {
        "affordable": ["A well-planned community", "Planned around families"],
        "mid": ["The masterplan", "Planned with purpose"],
        "luxury": ["The plan", "Composed as a whole"],
    },
    "residences": {
        "affordable": ["Homes & prices", "Choose your home"],
        "mid": ["The residences", "Homes for every stage"],
        "luxury": ["The residences", "Residences"],
    },
    "plans_units": {
        "affordable": ["Home layouts", "Unit plans"],
        "mid": ["Unit plans", "Layouts"],
        "luxury": ["Residence plans", "Plans"],
    },
    "plans_plates": {
        "affordable": ["Typical floor plans", "How each floor works"],
        "mid": ["Typical floor plans", "Floor plates"],
        "luxury": ["Typical floor plans", "Floor plates"],
    },
    "location_transit": {
        "affordable": ["Well connected, every day", "Metro, schools and shops, nearby"],
        "mid": ["Close to what matters", "Connected in every direction"],
        "luxury": ["The address", "At the centre of things"],
    },
    "location": {
        "affordable": ["Everything close to home", "A neighbourhood that works"],
        "mid": ["Close to what matters", "A neighbourhood that has it all"],
        "luxury": ["The address", "A considered location"],
    },
    "feature_project": {
        "affordable": ["A community built around you", "Planned for the way families live"],
        "mid": ["Architecture with an easy grace", "Presence, without excess"],
        "luxury": ["Architecture, considered", "A composition in light", "The arrival"],
    },
    "feature_leisure": {
        "affordable": ["Play, stroll, unwind", "Fresh air at your doorstep", "Room to play"],
        "mid": ["Weekends, sorted", "Time well spent", "Space to slow down"],
        "luxury": ["Leisure without leaving", "Evenings, unhurried", "Water, light and calm"],
    },
    "feature_home": {
        "affordable": ["A home that grows with you", "Comfort, room by room"],
        "mid": ["Comfort, room by room", "Made for real life"],
        "luxury": ["Rooms that breathe", "The art of living well"],
    },
    "closing": {
        "affordable": ["Book your site visit", "Your new home is waiting"],
        "mid": ["Come and see it for yourself", "Visit us"],
        "luxury": ["An invitation", "By appointment"],
    },
}

# Narrative frames. A frame may open with conditions in brackets; it is used
# only when they hold: [has:pool] the project lists a pool among its
# amenities, [kw:arch|column] its selling points mention arches or columns,
# [claim:offer] it has a selling point on that topic.
# Slots: {name} {short} {place} {locality} {city} {config}
# {noun} {noun_lc} {developer} {audience} {acres} {towers} {units}
# {possession} {price} {theme} {theme2} {zone1} {zone2} {zones} {places}
# {near} {amenities}
OVERVIEW_LEAD = {
    "affordable": [
        "{name} brings {config} {noun_lc} to {locality}, starting at {price} — with {theme} built in.",
        "Own a {config} home in {locality} from {price}, in a community designed for {theme}.",
        "{name} offers {config} {noun_lc} in {locality} from {price}, planned for {audience}.",
        "{name} brings {config} {noun_lc} to {locality}, planned for {audience}.",
    ],
    "mid": [
        "{name} brings {config} {noun_lc} to {place} — a community shaped around {theme} and {theme2}.",
        "On {locality}, {name} offers {config} {noun_lc} designed for {theme} and {theme2}.",
        "{name} brings {config} {noun_lc} to {place}, designed for {theme}.",
    ],
    "luxury": [
        "In {locality}, {name} offers {config} {noun_lc} defined by {theme} and {theme2}.",
        "{name} brings {config} {noun_lc} to {locality}, shaped by {theme} and {theme2}.",
        "{name} offers {config} {noun_lc} in {locality}, defined by {theme}.",
    ],
}

OVERVIEW_POOL = {
    "affordable": [
        "[has:landscape] Spread across {acres}, the community keeps green spaces within a short walk of every home.",
        "It is planned for {audience} who want a well-built home without stretching their budget.",
        "{developer} has designed every home to make the most of each square foot.",
        "Inside the gates, {zones} keep every day busy and close to home.",
        "Daily needs are close by too, with {near} a short trip away.",
        "[claim:offer] Book early to make the most of the savings and benefits open to buyers today.",
    ],
    "mid": [
        ["Set across {acres} with {units}, it keeps the easy scale of a neighbourhood.",
         "[has:landscape] Spread across {acres}, it balances homes with open, landscaped space."],
        "It is designed for {audience} who want their home to reflect how they live.",
        "{developer} has shaped the community around {theme2}, so the benefits are felt every day.",
        "Downstairs, {zones} turn ordinary evenings into something to look forward to.",
        "{near} are all within easy reach.",
        "Residences are scheduled for possession in {possession}.",
    ],
    "luxury": [
        ["Across {acres}, {towers} and {units} are composed as a single, unhurried whole.",
         "Across {acres}, the residences are composed as a single, unhurried whole."],
        "It is conceived for {audience} — those who value what is rare.",
        "{developer} brings its signature attention to detail to every surface and every threshold.",
        "Beyond the residences, {zones} complete a world of its own.",
        "Residences are scheduled for handover in {possession}.",
    ],
}

AMENITIES_LEAD = {
    "affordable": ["From {zone1} to {zone2}, everything your family enjoys is inside the community.",
                   "{zone1}, {zone2} and more — all a short walk from your door."],
    "mid": ["From {zone1} to {zone2}, leisure here is part of the everyday, not an occasion.",
            "{zone1} and {zone2}, a few steps from home."],
    "luxury": ["{zone1}, {zone2} and more, held within the grounds.",
               "From {zone1} to {zone2}, leisure is composed with the same care as the residences."],
}

FEATURE_POOL = {
    ("affordable", "leisure"): [
        "[has:sports] Courts and tracks keep everyone active, whatever their age.",
        "[has:landscape] Green walks and shaded lawns turn an ordinary evening into family time.",
        "{amenities} give children room to play and parents room to unwind.",
        "Fresh air, open lawns and safe paths make evenings outdoors part of family life.",
        "Everything is inside the gates, so weekends cost less and mean more.",
        "Neighbours become friends over a game, a walk or a picnic.",
    ],
    ("affordable", "project"): [
        "Wide internal roads and planned open spaces make the community easy to live in.",
        "{name} is planned around the way families actually live.",
        "[has:landscape] Homes look onto green, open spaces, so children always have room to play.",
        "A secure, gated community gives your family peace of mind every day.",
    ],
    ("affordable", "home"): [
        "Every room has a purpose, and every corner is put to use.",
        "Smartly planned rooms make the most of every square foot.",
        "Bright, practical spaces adapt as your family grows.",
    ],
    ("mid", "leisure"): [
        "[has:sports] Courts for a quick game after work keep the week in balance.",
        "[has:kids] Children have spaces of their own, close enough to keep an eye on.",
        "[has:fitness] Fitness is part of the routine when the gym is a lift ride away.",
        "{amenities} are a few steps from your front door.",
        "[has:pool] Mornings can begin with a swim and evenings can end outdoors.",
        "Leisure here is part of the everyday rather than a weekend plan.",
        "Children play within sight, and neighbours become friends.",
    ],
    ("mid", "project"): [
        "Towers rise with a quiet confidence, letting light and greenery lead.",
        "The details are considered, from the way the buildings meet the ground to the way they meet the sky.",
        "The architecture is calm and contemporary, with homes placed to make the most of light and outlook.",
        "Every approach is designed to feel like an arrival.",
        "{name} is designed around {theme} — visible in the details, felt every day.",
    ],
    ("mid", "home"): [
        "Kitchens and living spaces open to one another, so the family stays together.",
        "Rooms are planned for real life — gatherings, quiet evenings and everything between.",
        "Daylight and an easy flow between spaces make every home feel larger than its plan.",
    ],
    ("luxury", "leisure"): [
        "[has:sports] Courts and lawns invite a game in the cool of the evening.",
        "[has:kids] Children are given spaces of their own, imagined with the same care as everything else.",
        "[has:landscape] Gardens and plazas offer quiet corners for a book, a conversation or nothing at all.",
        "{amenities} are held within the grounds, reserved for residents.",
        "[has:pool] Leisure here is unhurried — an afternoon by the water, an evening among friends.",
        "Each space is composed with the same care as the residences themselves.",
    ],
    ("luxury", "project"): [
        "By night the architecture glows softly, a landmark that never shouts.",
        "Every elevation is resolved with the patience of a single hand.",
        "Thresholds are generous, and the transition from city to home is unhurried.",
        "[kw:column|arch] Columns and arches give the architecture a timeless, classical composure.",
        "Arrival is choreographed: the city falls away before you reach the door.",
        "Light moves across the façades through the day, changing their character hour by hour.",
    ],
    ("luxury", "home"): [
        "Materials are chosen for how they age, and how they feel underfoot.",
        "Proportion and natural light give each room a calm, assured character.",
        "Spaces flow into one another, so entertaining feels effortless and evenings feel private.",
    ],
}

GALLERY_LEAD = {
    "affordable": ["Inside {short}, every square foot of a {config} home is put to work.",
                   "{config} homes at {short}, planned so every room earns its place.",
                   "The {config} homes at {short} keep family life practical and bright."],
    "mid": ["Inside {short}: {config} {noun_lc} planned for light, flow and the way you live.",
            "The {config} {noun_lc} at {short} are made for real days, not show days.",
            "At {short}, {config} {noun_lc} balance open living with quiet corners."],
    "luxury": ["Inside {short}, {config} {noun_lc} composed around light and proportion.",
               "The {config} {noun_lc} at {short} are measured in light, not square feet.",
               "At {short}, each of the {config} {noun_lc} is an essay in calm."],
}

RESIDENCES_LEAD = {
    "affordable": ["Choose from {config} homes at {short} starting at {price}, with carpet areas from {area}.",
                   "{config} homes at {short} from {price}, with carpet areas from {area}.",
                   "At {short}, {config} homes start at {price}, with carpet areas from {area}."],
    "mid": ["{config} {noun_lc} at {short}, with carpet areas from {area}{price_clause}.",
            "At {short}, {config} {noun_lc} range in carpet area from {area}{price_clause}.",
            "{short} offers {config} {noun_lc} with carpet areas from {area}{price_clause}."],
    "luxury": ["{config} {noun_lc} at {short}, with carpet areas from {area}.",
               "At {short}, {config} {noun_lc} range from {area} of carpet area.",
               "{short} offers {config} {noun_lc}, from {area} of carpet area."],
}

LOCATION_LEAD = {
    "affordable": ["{near} — the places your family needs every week are close to home.",
                   "Close to {near}, {short} keeps everyday travel short."],
    "mid": ["{near} are all close at hand.",
            "{short} keeps {near} within easy reach."],
    "luxury": ["{near} lie within easy reach.",
               "The address places {near} close at hand."],
}

CLOSING_LINE = {
    "affordable": ["Call us or visit {short} — our team will help you choose the right home and plan your payments.",
                   "Visit {short} with your family; our team will walk you through homes, prices and payments.",
                   "See {short} for yourself — we will help you pick the home that fits your budget."],
    "mid": ["Walk the grounds of {short}, see the residences and find the one that fits your life.",
            "Visit {short} to see the homes, the amenities and the thinking behind them.",
            "Come to {short} and spend an hour in the place you could call home."],
    "luxury": ["A private viewing of {short} can be arranged at your convenience.",
               "{short} is best experienced in person, by appointment.",
               "We would be pleased to welcome you to {short}."],
}

MASTERPLAN_LINE = {
    "affordable": ["The plan of {short} shows how homes, open spaces and amenities fit together, so you can see how close everything is.",
                   "Homes, open spaces and amenities at {short}, all on one page.",
                   "See at a glance how {short} keeps everything within a short walk."],
    "mid": ["The plan shows how the residences, open spaces and amenities of {short} come together across the site.",
            "How {short} is laid out: residences, open spaces and amenities in one view.",
            "The site plan of {short}, with its residences, open spaces and amenities."],
    "luxury": ["The plan reveals how the residences, gardens and leisure of {short} are composed as one considered whole.",
               "{short}, seen whole: residences, gardens and leisure in one composition.",
               "The composition of {short}, from the arrival to the furthest garden."],
}

PLAN_NOTE = "Plans are indicative and not to scale."
PRICE_NOTE = "Prices are as supplied and subject to change; please confirm current pricing with our team."

CAPTIONS = {"living": "Living", "dining": "Dining", "kitchen": "Kitchen", "bedroom": "Bedroom",
            "bathroom": "Bath", "balcony": "Terrace", "pool": "Pool", "clubhouse": "Clubhouse",
            "sports": "Sports", "kids": "Play", "landscape": "Gardens", "fitness": "Fitness",
            "convenience": "Retail", "exterior": "", "aerial": "", "night": "", "arrival": "Arrival"}
