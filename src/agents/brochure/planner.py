"""The art director: the whole book is planned before a word is written.

    1. INVENTORY  every upload as photo, floor plan, master plan, map or
                  logo; what a photograph shows (the subject the person gave
                  it, else the Vision Agent's reading, else its file name);
                  its pixel size; which uploads are the same picture twice
                  (image_plan.py)
    2. PAGES      only the pages the project's material supports, in the
                  category's order (strategy.py) — a project with no master
                  plan has no master-plan page, one with five photographs
                  is not stretched to sixteen pages; no contents page and no
                  generic "how to book" page: the closing page carries the
                  call to action and every way to act on it
    3. PICTURES   each photograph on one page only, chosen for the page by
                  subject and by SHAPE: a photograph goes to the slot whose
                  proportions crop it least, the cover takes the strongest
                  view of the building, plans and maps are shown whole and
                  large, and nothing is printed far below its resolution
    4. CONTENT    the structured content each page sets (facts.py): the
                  figures, the claims, the amenity zones, the unit schedule,
                  the places — each on the one page it belongs to

Then copywriter.py writes the words.
"""
from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from src.agents.brochure import facts, focal, image_plan, layouts, strategy, topics
from src.agents.brochure.book import Book, BookPage, Box, Placed
from src.agents.brochure.catalog import TemplateSpec

logger = logging.getLogger(__name__)

LEISURE = ("pool", "clubhouse", "sports", "kids", "fitness", "landscape", "convenience", "care")
INTERIOR = ("living", "dining", "kitchen", "bedroom", "bathroom", "balcony")
PROJECT = ("exterior", "aerial", "night", "arrival")

PREFS = {
    "cover": ("exterior", "night", "aerial", "arrival", "clubhouse", "landscape", "pool"),
    "closing": ("night", "arrival", "exterior", "aerial", "clubhouse", "landscape", "balcony"),
    "overview": ("aerial", "exterior", "landscape", "arrival", "clubhouse", "night", "pool"),
    "highlights": ("landscape", "clubhouse", "balcony", "pool", "exterior", "arrival", "lifestyle",
                   "night", "sports", "kids", "convenience"),
    "figures": ("landscape", "aerial", "night", "exterior", "pool", "balcony", "kids"),
    "residences": ("living", "dining", "bedroom", "balcony", "kitchen"),
    "location": ("aerial", "exterior", "convenience", "arrival", "night"),
    "amenities": ("pool", "clubhouse", "sports", "kids", "fitness", "landscape", "convenience", "care"),
    "gallery": ("living", "dining", "kitchen", "bedroom", "bathroom", "balcony"),
}

SUBJECT_SYNONYMS = {
    "facade": "exterior", "elevation": "exterior", "tower": "exterior", "building": "exterior",
    "entrance": "arrival", "gate": "arrival", "lobby": "arrival", "dropoff": "arrival",
    "garden": "landscape", "park": "landscape", "court": "sports", "gym": "fitness",
    "club": "clubhouse", "swimming": "pool", "retail": "convenience", "shopping": "convenience",
    "bath": "bathroom", "bed": "bedroom", "terrace": "balcony", "dusk": "night",
    "play": "kids", "children": "kids",
}

MIN_DPI = 150          # below this a photograph prints visibly soft


# ---------------------------------------------------------------------------
# 1. inventory
# ---------------------------------------------------------------------------


@dataclass
class Asset:
    id: str
    kind: str                  # photo plan masterplan map logo twin
    topic: str
    aspect: float
    px_w: int
    px_h: int
    score: float
    filename: str
    ai: bool = False
    cluster: int = -1
    visible_text: str = ""
    path: str = ""


def _size(path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - an unreadable image is left out
        return 0, 0


def _kind(image, analysis) -> str:
    role = (getattr(image, "role", "") or "").lower()
    mapping = {"floorplan": "plan", "floor_plan": "plan", "plan": "plan", "masterplan": "masterplan",
               "master_plan": "masterplan", "siteplan": "masterplan", "site_plan": "masterplan",
               "map": "map", "location_map": "map", "logo": "logo", "photo": "photo"}
    if role in mapping:
        return mapping[role]
    name = (image.filename or "").lower()
    if analysis is None:
        if "logo" in name:
            return "logo"
        if re.search(r"master|site[_ -]?plan|layout", name):
            return "masterplan"
        if "plan" in name:
            return "plan"
        if "map" in name or "location" in name:
            return "map"
        return "photo"
    category = (getattr(analysis.category, "value", None) or str(analysis.category)).lower()
    words = " ".join((analysis.scene_tag or "", analysis.suggested_title or "", name)).lower()
    if category == "logo":
        return "logo"
    if category in ("floorplan", "map") and re.search(r"master|site|layout|campus|township", words):
        return "masterplan"
    if category == "floorplan":
        return "plan"
    if category == "map":
        return "map"
    if category in ("document", "infographic"):
        return "skip"
    return "photo"


def _topic(image, analysis) -> str:
    subject = (getattr(image, "subject", "") or "").strip().lower()
    if subject:
        if subject in topics.TOPICS:
            return subject
        return SUBJECT_SYNONYMS.get(subject, topics.match(subject) or "other")
    return topics.of_image(analysis, image.filename)


def inventory(images: Sequence, analyses: dict) -> list[Asset]:
    out = []
    for image in images:
        w, h = _size(image.path)
        if not w or not h:
            continue
        analysis = analyses.get(image.image_id)
        kind = _kind(image, analysis)
        if kind == "skip":
            continue
        topic = _topic(image, analysis) if kind == "photo" else kind
        score = 0.0
        if analysis is not None:
            score = analysis.hero_score * 1.5 + analysis.quality + analysis.composition
        score += min(3.0, (w * h) / 2_000_000)          # resolution counts
        out.append(Asset(id=image.image_id, kind=kind, topic=topic, aspect=w / h, px_w=w, px_h=h,
                         score=score, filename=image.filename,
                         ai=bool(getattr(image, "is_ai_generated", False)), path=image.path,
                         visible_text=(getattr(analysis, "visible_text", "") or "") if analysis else ""))
    return out


def plan_label(filename: str) -> str:
    """A drawing's name from its file name, tidied: 'HK-Floor Plan-Typical
    Tower 2 3 4-Template (1).webp' -> 'Typical floor · Towers 2, 3 & 4'."""
    stem = re.sub(r"\.[a-z0-9]+$", "", filename, flags=re.I)
    stem = re.sub(r"^(floorplan|masterplan|plan|map)_", "", stem, flags=re.I)
    stem = re.sub(r"\(\d+\)|_\d+$|\btemplate\b|\bcopy\b|\bfinal\b", " ", stem, flags=re.I)
    stem = re.sub(r"^[A-Z]{2,3}[-_ ]+", "", stem.strip())
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"(\d)\s*BHK", r"\1 BHK", stem, flags=re.I)
    stem = re.sub(r"\b(\d)\s+5 BHK", r"\1.5 BHK", stem)
    if re.fullmatch(r"\d+(?:\.\d)?\s*BHK(?:\s+\w+)?", stem, re.I):
        return re.sub(r"(?i)bhk", "BHK", stem)
    stem = facts.clean(stem)
    tower = re.search(r"typical.*?tower\s*([\d ]+)", stem, re.I)
    if tower:
        nums = tower.group(1).split()
        towers = facts.prose_list(nums, "&")
        return f"Typical floor plan · Tower{'s' if len(nums) > 1 else ''} {towers}"
    stem = re.sub(r"\bfloor\s*plan\b", "", stem, flags=re.I).strip()
    if re.fullmatch(r"[\w ]{1,40}", stem) and stem:
        words = [w if w.isupper() or w.upper() == "BHK" else w.capitalize() for w in stem.split()]
        return " ".join(w.upper() if w.lower() == "bhk" else w for w in words)
    return ""


# ---------------------------------------------------------------------------
# 2-3. picking pictures
# ---------------------------------------------------------------------------


class Pool:
    def __init__(self, assets: list[Asset]):
        self.assets = assets
        self.used: set[str] = set()

    def photos(self, allow_ai: bool = True) -> list[Asset]:
        return [a for a in self.assets if a.kind == "photo" and a.id not in self.used
                and (allow_ai or not a.ai)]

    def count(self, topic_set: Sequence[str]) -> int:
        return sum(1 for a in self.photos() if a.topic in topic_set)

    def pick(self, prefs: Sequence[str], aspect: Optional[float] = None, *, strict: bool = False,
             allow_ai: bool = True, slot_mm: Optional[tuple[float, float]] = None,
             avoid: Sequence[str] = (), max_crop: Optional[float] = None) -> Optional[Asset]:
        best, best_cost = None, 1e9
        for a in self.photos(allow_ai):
            if max_crop is not None and aspect and layouts.crop_loss(a.aspect, aspect) > max_crop:
                continue
            if a.topic in avoid:
                continue
            if a.topic in prefs:
                rank = prefs.index(a.topic) * 0.6
            elif strict:
                continue
            else:
                rank = len(prefs) * 0.6 + 2
            cost = rank - a.score * 0.08
            if aspect:
                cost += 6.0 * layouts.crop_loss(a.aspect, aspect)
            if slot_mm:
                dpi = min(a.px_w * 25.4 / slot_mm[0], a.px_h * 25.4 / slot_mm[1])
                if dpi < MIN_DPI:
                    cost += (MIN_DPI - dpi) / 25.0
            if cost < best_cost:
                best, best_cost = a, cost
        if best is not None:
            self.used.add(best.id)
        return best


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def plan(prop, images: Sequence, analyses: dict, spec: TemplateSpec,
         positioning: Optional[strategy.Positioning] = None) -> Book:
    positioning = positioning or strategy.for_category(spec.category)
    content = facts.build(prop, spec.category)
    analyses = analyses or {}
    assets = inventory(images, analyses)
    by_id = {a.id: a for a in assets}
    notes: list[str] = []

    # the same picture twice is printed once
    images_by_id = {i.image_id: i for i in images if i.image_id in by_id}
    clusters = image_plan.build(images_by_id, analyses)
    best_of: dict[int, str] = {}
    for a in sorted(assets, key=lambda x: -x.score):
        a.cluster = clusters.cluster(a.id)
        if a.kind == "photo" and a.cluster >= 0:
            if a.cluster in best_of:
                a.kind = "twin"
            else:
                best_of[a.cluster] = a.id
    twins = [a for a in assets if a.kind == "twin"]
    if twins:
        notes.append(f"{len(twins)} photograph(s) repeated another and were printed once")

    pool = Pool(assets)
    plans_ = [a for a in assets if a.kind == "plan"]
    masterplans = [a for a in assets if a.kind == "masterplan"]
    maps = [a for a in assets if a.kind == "map"]
    logo = next((a for a in assets if a.kind == "logo"), None)

    def slot_of(kind: str, name: str = "img", **opt) -> layouts.Slot:
        return layouts.build(kind, spec, **opt).slots[name]

    def mm(slot: layouts.Slot) -> tuple[float, float]:
        return slot.rect.w, slot.rect.h

    # -- the pages the material supports ------------------------------------
    book_kinds = list(positioning.book)
    green = content.claims_on("green")
    stats = content.stats
    has_figures = "figures" in book_kinds and len(stats) >= 2 and len(green) >= 2
    highlight_claims = [c for c in content.claims if not (has_figures and c.topic == "green")]
    if spec.category == "affordable" and content.units:
        # payment benefits sit with the prices, on the homes page
        highlight_claims = [c for c in highlight_claims if c.topic != "offer"]
    has_highlights = len(highlight_claims) >= 3
    interiors = pool.count(INTERIOR)
    gallery_n = min(4, interiors) if "gallery" in book_kinds and interiors >= 2 else 0
    wide_rooms = [a for a in pool.photos() if a.topic in INTERIOR]
    if spec.portrait and wide_rooms and sum(a.aspect for a in wide_rooms) / len(wide_rooms) > 1.45:
        gallery_n = min(gallery_n, 3)
    leisure_n = pool.count(LEISURE)
    amen_slots = 3 if spec.landscape else 2

    pages: dict[str, list[BookPage]] = {k: [] for k in book_kinds}

    # cover — the strongest view of the building, never an AI image
    cover_slot = slot_of("cover")
    # The cover fills its sheet, so it is never reshaped: it takes a view of
    # the building that fits the sheet within the crop cap when there is one.
    cover_aspect, cap = cover_slot.rect.aspect, layouts.CROP_CAP
    cover_img = (pool.pick(PREFS["cover"], cover_aspect, allow_ai=False, slot_mm=mm(cover_slot), strict=True,
                           max_crop=cap)
                 or pool.pick(PREFS["cover"], cover_aspect, allow_ai=False, max_crop=cap)
                 or pool.pick(PREFS["cover"], cover_aspect, allow_ai=False, slot_mm=mm(cover_slot), strict=True)
                 or pool.pick(PREFS["cover"], cover_aspect, allow_ai=False))
    cover = BookPage(kind="cover", data={"image": [cover_img.id] if cover_img else []})

    # interiors for the gallery first: they are the scarcest subject
    gallery_imgs: list[Asset] = []
    for _ in range(gallery_n):
        a = pool.pick(PREFS["gallery"], 1.5, strict=True, avoid=[x.topic for x in gallery_imgs]) \
            or pool.pick(PREFS["gallery"], 1.5, strict=True)
        if a:
            gallery_imgs.append(a)

    amen_imgs: list[Asset] = []
    if content.zones:
        zone_topics = [z.topic for z in content.zones]
        prefs = tuple(dict.fromkeys([t for t in zone_topics if t in LEISURE] + list(PREFS["amenities"])))
        for _ in range(min(amen_slots, leisure_n)):
            a = pool.pick(prefs, 1.3, strict=True, avoid=[x.topic for x in amen_imgs]) \
                or pool.pick(prefs, 1.3, strict=True)
            if a:
                amen_imgs.append(a)

    ov_slot = slot_of("overview")
    overview_img = pool.pick(PREFS["overview"], ov_slot.rect.aspect, allow_ai=False, slot_mm=mm(ov_slot))
    closing_slot = slot_of("closing")
    closing_img = pool.pick(PREFS["closing"], closing_slot.rect.aspect, allow_ai=False, slot_mm=mm(closing_slot))

    res_img = None
    if content.units:
        rs = slot_of("residences")
        res_img = pool.pick(PREFS["residences"], rs.rect.aspect, strict=True)

    hi_img = None
    if has_highlights:
        hs = layouts.build("highlights", spec).slots.get("img")
        hi_img = pool.pick(PREFS["highlights"], hs.rect.aspect if hs else 1.0)

    fig_img = None
    if has_figures:
        fs = slot_of("figures")
        fig_img = pool.pick(PREFS["figures"], fs.rect.aspect)

    loc_img = None
    if (content.places or maps) and not maps:
        ls = slot_of("location", has_map=False)
        loc_img = pool.pick(PREFS["location"], ls.rect.aspect)

    if content.units and res_img is None and len(pool.photos()) > 1:
        res_img = pool.pick(PREFS["residences"], slot_of("residences").rect.aspect)

    # -- feature pages from what is left, grouped by subject ----------------
    fixed = 2 + 1 + int(has_highlights) + int(has_figures) + int(bool(content.zones)) \
        + int(bool(gallery_imgs)) + int(bool(masterplans)) + int(bool(content.units)) \
        + int(bool(content.places or maps)) + _plan_pages(plans_, spec)
    room = max(0, positioning.max_pages - fixed)
    order = {"luxury": PROJECT + LEISURE + INTERIOR, "mid": PROJECT + LEISURE + INTERIOR,
             "affordable": LEISURE + PROJECT + INTERIOR}[spec.category]
    left = sorted(pool.photos(), key=lambda a: (order.index(a.topic) if a.topic in order else 99, -a.score))
    features: list[tuple[str, list[Asset]]] = []
    family = lambda t: "project" if t in PROJECT else ("home" if t in INTERIOR else "leisure")  # noqa: E731
    groups: dict[str, list[Asset]] = {}
    for a in left:
        groups.setdefault(family(a.topic), []).append(a)
    per = positioning.max_feature_images
    for fam in ("project", "leisure", "home"):
        items = groups.get(fam, [])
        while items and len(features) < room:
            take = len(items) if len(items) <= per else (per if len(items) - per != 1 else per - 1)
            take = max(1, min(take, per))
            chunk, items = items[:take], items[take:]
            features.append((fam, chunk))
            for a in chunk:
                pool.used.add(a.id)
    unused = pool.photos()
    if unused:
        notes.append(f"{len(unused)} photograph(s) were not needed for a book of this length "
                     f"and were left out")

    # -- every selling point on exactly one page ----------------------------
    shown = {id(cl) for cl in (highlight_claims if has_highlights else [])}
    if spec.category == "affordable" and content.units:
        shown |= {id(cl) for cl in content.claims_on("offer")}
    shown |= {id(cl) for cl in (green if has_figures else [])}
    leftover = [cl for cl in content.claims if id(cl) not in shown]
    location_claims = [cl for cl in leftover if cl.topic == "connectivity"] \
        if (content.places or maps) else []
    overview_claims = [cl for cl in leftover if cl not in location_claims]

    # -- assemble -----------------------------------------------------------
    seq: list[BookPage] = [cover]
    overview = BookPage(kind="overview", data={"image": [overview_img.id] if overview_img else [],
                                               "claims": [cl.__dict__ for cl in overview_claims]})
    highlights = BookPage(kind="highlights", data={"image": [hi_img.id] if hi_img else [],
                                                   "claims": [cl.__dict__ for cl in highlight_claims]}) \
        if has_highlights else None
    figures_pg = BookPage(kind="figures", data={"image": [fig_img.id] if fig_img else []}) \
        if has_figures else None
    amen = BookPage(kind="amenities", data={"image": [a.id for a in amen_imgs]}) if content.zones else None
    gallery = BookPage(kind="gallery", data={"image": [a.id for a in gallery_imgs],
                                             "topics": [a.topic for a in gallery_imgs]}) if gallery_imgs else None
    master = BookPage(kind="masterplan", data={"plan": [masterplans[0].id]}) if masterplans else None
    if len(masterplans) > 1:
        notes.append(f"{len(masterplans) - 1} further master plan(s) were not printed")
    residences = BookPage(kind="residences", data={"image": [res_img.id] if res_img else []}) \
        if content.units else None
    plan_pages = _plan_book(plans_, content, spec, notes)
    # Homes and their layouts together: where the homes page has no
    # photograph of the interiors, the unit plans take its place, and their
    # own page goes.
    if residences is not None and res_img is None and spec.category == "affordable":
        units_page = next((p for p in plan_pages if p.topic == "units" and len(p.data["plan"]) <= 2), None)
        if units_page is not None:
            residences.data["plan"] = list(units_page.data["plan"])
            residences.data["labels"] = units_page.data.get("labels", {})
            residences.data["areas"] = units_page.data.get("areas", {})
            plan_pages.remove(units_page)
    location = BookPage(kind="location", data={"plan": [maps[0].id] if maps else [],
                                               "image": [loc_img.id] if loc_img else [],
                                               "claims": [cl.__dict__ for cl in location_claims[:1]]}) \
        if (content.places or maps) else None
    closing = BookPage(kind="closing", data={"image": [closing_img.id] if closing_img else []})
    feature_pages = [BookPage(kind="feature", topic=fam, data={"image": [a.id for a in chunk],
                                                               "topics": [a.topic for a in chunk]})
                     for fam, chunk in features]
    after = {"project": "highlights", "leisure": "amenities", "home": "gallery"}
    placed_feature: set[int] = set()

    singles = {"overview": overview, "highlights": highlights, "figures": figures_pg, "amenities": amen,
               "gallery": gallery, "masterplan": master, "residences": residences, "location": location,
               "closing": closing}
    for kind in book_kinds[1:]:
        if kind == "plans":
            seq.extend(plan_pages)
            continue
        if kind == "feature":
            for i, fp in enumerate(feature_pages):
                if fp.topic == "project" and i not in placed_feature:
                    seq.append(fp)
                    placed_feature.add(i)
            continue
        page = singles.get(kind)
        if page is not None:
            seq.append(page)
        for i, fp in enumerate(feature_pages):
            if i not in placed_feature and after.get(fp.topic) == kind:
                seq.append(fp)
                placed_feature.add(i)
    for i, fp in enumerate(feature_pages):            # whatever had no anchor
        if i not in placed_feature:
            seq.insert(len(seq) - 1, fp)

    # -- lay out and place ----------------------------------------------------
    mirror = False
    for number, page in enumerate(seq, start=1):
        page.number = number
        _compose(page, spec, by_id, content, mirror)
        if page.kind == "feature":
            mirror = not mirror

    for page in seq:
        _attach_data(page, content, positioning, spec)

    book = Book(template=spec.id, template_name=spec.name, category=spec.category,
                orientation=spec.orientation, width_mm=spec.sheet.w, height_mm=spec.sheet.h,
                pages=seq, notes=notes, logo=logo.id if logo else None,
                roles={**{a.id: a.kind for a in assets}, **{f"topic:{a.id}": a.topic for a in assets}})
    for page in seq:
        for p in page.images:
            if p.kind == "image" and p.dpi and p.dpi < MIN_DPI:
                book.notes.append(f"Page {page.number}: a photograph prints at about {p.dpi:.0f} dpi "
                                  f"(supply a larger original for print)")
            if p.kind == "plan" and p.dpi and p.dpi < 120:
                book.notes.append(f"Page {page.number}: {p.label or 'a plan'} is a small file "
                                  f"(about {p.dpi:.0f} dpi at this size) — supply a larger original "
                                  f"for print")
    return book


def _plan_pages(plans_: list[Asset], spec: TemplateSpec) -> int:
    units = [a for a in plans_ if re.search(r"bhk|unit|type|villa", a.filename, re.I)]
    plates = [a for a in plans_ if a not in units]
    return -(-len(units) // 2) + -(-len(plates) // 4)


def _plan_book(plans_: list[Asset], content: facts.Content, spec: TemplateSpec,
               notes: list[str]) -> list[BookPage]:
    """Unit plans two to a page; typical floor plates up to four."""
    linked = {u.plan_image: u for u in content.units if u.plan_image}
    units, plates = [], []
    for a in plans_:
        (units if a.filename in linked or re.search(r"bhk|unit|type|villa", a.filename, re.I)
         else plates).append(a)
    pages = []
    for group, size, sub in ((units, 2, "units"), (plates, 4, "plates")):
        for i in range(0, len(group), size):
            chunk = group[i:i + size]
            pages.append(BookPage(kind="plans", topic=sub, data={"plan": [a.id for a in chunk]}))
    labels = {a.id: (linked[a.filename].label if a.filename in linked else plan_label(a.filename))
              for a in plans_}
    known = {re.sub(r"\s+", "", u.label.lower()) for u in content.units}
    for a in units:
        lab = re.sub(r"\s+", "", (labels.get(a.id) or "").lower())
        if lab and known and not any(lab.startswith(k) or k.startswith(lab) for k in known):
            notes.append(f"Plan '{labels[a.id]}' ({a.filename}) does not match a configuration in "
                         f"the unit schedule — check its label")
    for page in pages:
        page.data["labels"] = {i: labels.get(i, "") for i in page.data["plan"]}
        page.data["areas"] = {a.id: linked[a.filename].area for a in plans_
                              if a.id in page.data["plan"] and a.filename in linked}
    return pages


def _place(slot: layouts.Slot, a: Asset, kind: str) -> Placed:
    r = slot.rect
    if kind == "plan":
        dpi = max(a.px_w * 25.4 / r.w, a.px_h * 25.4 / r.h)
        crop = 0.0
        position = "50% 50%"
    else:
        dpi = min(a.px_w * 25.4 / r.w, a.px_h * 25.4 / r.h)
        crop = layouts.crop_loss(a.aspect, r.aspect)
        position = focal.focal_point(a.path, a.aspect, r.aspect, a.topic)
    return Placed(slot=slot.name, image_id=a.id, x=r.x, y=r.y, w=r.w, h=r.h, kind=kind,
                  position=position, bleed=slot.bleed, dpi=round(dpi), crop=round(crop, 3),
                  aspect=round(a.aspect, 4))


def _assign(slots: list[layouts.Slot], assets: list[Asset]) -> list[tuple[layouts.Slot, Asset]]:
    """Photographs to slots so the total crop is least (and the best
    photograph takes the largest slot when crops tie)."""
    if not slots or not assets:
        return []
    assets = assets[:len(slots)]
    slots_ = slots[:len(assets)]
    best, best_cost = None, 1e9
    for perm in itertools.permutations(assets):
        cost = 0.0
        for s, a in zip(slots_, perm):
            cost += layouts.crop_loss(a.aspect, s.rect.aspect) ** 1.5 * 10
            cost -= a.score * s.rect.area / 1e5
        if cost < best_cost:
            best, best_cost = list(zip(slots_, perm)), cost
    return best or []


WIDE_CAPABLE = ("overview", "figures", "feature", "residences", "closing", "highlights", "location",
                "gallery", "amenities")
# Pages with a column composition for a tall photograph on a portrait sheet.
TALL_CAPABLE = ("overview", "figures", "feature", "residences", "closing", "location")


def _crop_total(layout: layouts.Layout, imgs: list[Asset]) -> float:
    pairs = _assign(layout.images(), imgs)
    return sum(layouts.crop_loss(a.aspect, s.rect.aspect) for s, a in pairs) if pairs else 1.0


def _compose(page: BookPage, spec: TemplateSpec, by_id: dict, content: facts.Content, mirror: bool) -> None:
    imgs = [by_id[i] for i in page.data.get("image", []) if i in by_id]
    drawings = [by_id[i] for i in page.data.get("plan", []) if i in by_id]
    k = page.kind
    opts: dict = {}
    if k == "closing":
        opts = {"with_image": bool(imgs)}
    elif k == "highlights":
        opts = {"with_image": bool(imgs)}
    elif k == "residences":
        opts = {"with_image": bool(imgs), "plans": len(drawings)}
    elif k == "feature":
        opts = {"n": max(1, len(imgs)), "mirror": mirror}
    elif k in ("gallery", "amenities"):
        opts = {"n": len(imgs)}
    elif k == "plans":
        opts = {"n": max(1, len(drawings))}
    elif k == "location":
        opts = {"has_map": bool(drawings), "with_image": bool(imgs)}
    layout = layouts.build(k, spec, **opts)
    # A wide photograph on a landscape sheet: the band composition, when it
    # crops the picture clearly less than the column one.
    single = not (k == "feature" and len(imgs) > 1) and not (k == "location" and drawings)
    either = k in ("gallery", "amenities")          # these have wide forms in both orientations
    if (spec.landscape or either) and imgs and k in WIDE_CAPABLE and single:
        alt = layouts.build(k, spec, **opts, wide=True)
        if _crop_total(alt, imgs) + 0.12 < _crop_total(layout, imgs):
            layout = alt
    # A tall photograph on a portrait sheet: a column beside the copy rather
    # than a band that keeps a strip of it.
    if spec.portrait and imgs and k in TALL_CAPABLE and single and len(imgs) == 1:
        alt = layouts.build(k, spec, **opts, tall=True)
        if alt.images() and _crop_total(alt, imgs) + 0.12 < _crop_total(layout, imgs):
            layout = alt
    page.layout, page.variant = layout.name, layout.variant
    for slot, a in _assign(layout.images(), imgs):
        placed = _place(slot, a, "image")
        if k != "cover":            # the cover fills its sheet; its focal point keeps the subject
            _cap_crop(placed, a)
        page.images.append(placed)
    for slot, a in zip(layout.plans(), drawings):
        placed = _place(slot, a, "plan")
        placed.label = (page.data.get("labels") or {}).get(a.id, "")
        placed.sublabel = (page.data.get("areas") or {}).get(a.id, "")
        page.images.append(placed)
    _fit_drawings(page, spec, by_id)
    for s in layout.text():
        page.boxes[s.name] = Box(slot=s.name, x=s.rect.x, y=s.rect.y, w=s.rect.w, h=s.rect.h)
    _close_up(page, spec)
    # the slots the page has no picture for are dropped (the layout was
    # chosen for the pictures it has, so this is only a short page)


def _cap_crop(p: Placed, a: Asset) -> None:
    """No photograph loses more than layouts.CROP_CAP of itself. Where its
    slot would cut deeper, the slot takes the picture's shape instead —
    narrowed (or shortened) about its centre, and no longer bleeding on the
    sides it pulled back from. The space it gives up stays paper; no copy
    moves into it, so nothing can overlap."""
    if p.crop <= layouts.CROP_CAP or not a.aspect:
        return
    cap = layouts.CROP_CAP
    t, r, b, l = p.bleed
    if a.aspect < p.w / p.h:                 # picture taller than the slot: narrow the slot
        new_w = p.h * a.aspect / (1 - cap)
        p.x += (p.w - new_w) / 2
        p.w = new_w
        p.bleed = (t, False, b, False)
    else:                                    # picture wider: shorten the slot
        new_h = p.w / (a.aspect * (1 - cap))
        p.y += (p.h - new_h) / 2
        p.h = new_h
        p.bleed = (False, r, False, l)
    p.crop = round(layouts.crop_loss(a.aspect, p.w / p.h), 3)
    p.dpi = round(min(a.px_w * 25.4 / p.w, a.px_h * 25.4 / p.h))
    p.position = focal.focal_point(a.path, a.aspect, p.w / p.h, a.topic)


PLAN_PAD = {"luxury": 7.0, "mid": 5.0, "affordable": 5.0}
CAPTION_MM = 8.0


def _fit_drawings(page: BookPage, spec: TemplateSpec, by_id: dict) -> None:
    """A drawing's card takes the drawing's own proportions, so a plan is
    not a small picture in a large empty card. Cards in one row share a
    height; the rows are centred in the space the layout gave them."""
    plans = [p for p in page.images if p.kind == "plan"]
    if not plans:
        return
    pad = PLAN_PAD.get(spec.category, 5.0)
    top = min(p.y for p in plans)
    bottom = max(p.y + p.h for p in plans)
    rows: dict[float, list[Placed]] = {}
    for p in plans:
        rows.setdefault(round(p.y, 1), []).append(p)
    needed = {}
    for y, row in rows.items():
        h_row = 0.0
        for p in row:
            a = by_id.get(p.image_id)
            cap = CAPTION_MM if (p.label or p.sublabel) else 0.0
            inner_w, inner_h = p.w - 2 * pad, p.h - 2 * pad - cap
            if not a or inner_w <= 0 or inner_h <= 0:
                h_row = max(h_row, p.h)
                continue
            draw_h = min(inner_h, inner_w / a.aspect)
            h_row = max(h_row, draw_h + 2 * pad + cap)
        needed[y] = min(h_row, max(p.h for p in row))
    gap = spec.design.gutter
    total = sum(needed.values()) + gap * (len(rows) - 1)
    cursor = top + max(0.0, (bottom - top - total) / 2)
    for y in sorted(rows):
        for p in rows[y]:
            p.y, p.h = cursor, needed[y]
            a = by_id.get(p.image_id)
            if a:
                p.dpi = round(max(a.px_w * 25.4 / max(1, p.w - 2 * pad), a.px_h * 25.4 / max(1, p.h - 2 * pad)))
        cursor += needed[y] + gap


def _close_up(page: BookPage, spec: TemplateSpec) -> None:
    """On a stacked page (a drawing above the copy), the drawing sits at the
    top of its area and the copy starts where the drawing ends, rather than
    where the layout's grid row ended."""
    text = page.boxes.get("text")
    plans = [p for p in page.images if p.kind == "plan"]
    if text is None or not plans or page.variant != "stack":
        return
    up = min(p.y for p in plans) - spec.frame().y
    for p in plans:
        p.y -= up
    text.y = max(p.y + p.h for p in plans) + 3 * spec.design.gutter
    text.h = spec.frame().bottom - text.y


def _attach_data(page: BookPage, c: facts.Content, positioning: strategy.Positioning,
                 spec: TemplateSpec) -> None:
    d = page.data
    k = page.kind
    if k == "cover":
        d.update(config=c.config, noun=c.noun, place=c.place_phrase, tagline=c.tagline,
                 price=c.price_from if positioning.show_price else "",
                 price_note="onwards" if c.price_from else "",
                 figures=[f.__dict__ for f in c.figures[:3]], possession=c.possession)
    elif k == "overview":
        figs = [f.__dict__ for f in c.figures]
        if c.possession:
            figs.append({"value": _short_date(c.possession), "unit": "Possession", "label": ""})
        if len(figs) < 3:
            figs += [f.__dict__ for f in facts.count_figures(c.amenity_lines)][:3 - len(figs)]
        d.update(figures=figs[:4])
    elif k == "highlights":
        d.setdefault("claims", [cl.__dict__ for cl in c.claims])
    elif k == "figures":
        d.update(stats=[s.__dict__ for s in c.stats[:4]],
                 claims=[cl.__dict__ for cl in c.claims_on("green")])
    elif k == "amenities":
        d.update(zones=[z.__dict__ for z in c.zones])
    elif k == "residences":
        d.update(units=[{"label": u.label, "detail": u.detail, "area": u.area,
                         "price": u.price if positioning.show_price else ""} for u in c.units],
                 price_from=c.price_from if positioning.show_price else "",
                 offers=[cl.__dict__ for cl in c.claims_on("offer")] if spec.category == "affordable" else [],
                 area_range=c.area_range,
                 summary=[x for x in (
                     {"value": c.price_from, "unit": "Starting price"} if positioning.show_price and c.price_from else None,
                     {"value": c.area_range, "unit": "Carpet areas"} if c.area_range else None,
                     {"value": str(len(c.units)), "unit": "Layouts"} if len(c.units) > 1 else None) if x])
    elif k == "masterplan":
        figs = [f.__dict__ for f in c.figures]
        d.update(figures=figs[:3])
    elif k == "location":
        d.update(places=[p.__dict__ for p in c.places[:8]], address=c.address or c.place_phrase)
    elif k == "closing":
        d.update(phone=c.phone, email=c.email, website=c.website, address=c.visit_address,
                 rera=c.rera, developer=c.developer, cta=c.cta)


def _short_date(text: str) -> str:
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", text.strip())
    if m and len(m.group(1)) > 3:
        return f"{m.group(1)[:3]} {m.group(2)}"
    return text
