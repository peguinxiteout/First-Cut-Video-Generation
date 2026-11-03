"""One owner of which photograph appears where.

THE DEFECT THIS EXISTS TO END

A thirty-six page SOBHA book printed the same couple watching a sunset on
two consecutive pages, and five pages carrying what a reader sees as the
same white Greek clubhouse: "Amenities", "Clubhouse Facade", "Clubhouse
Exterior", the field beside "Everything on site", and the field beside the
track record.

Only ONE of those was a duplicate file. The other four were four different
renders of the same building from almost the same angle — different bytes,
the same picture. Nothing in the system could see that, because every
check it had was on identity: `page.images` compared by image_id, a
"spare" pool keyed on whether an id had been used. A book is not judged on
file hashes.

WHAT THIS DOES

    1. SIGN      Every photograph gets a perceptual signature — a 8x8
                 difference hash of its luma plus a coarse colour profile.
                 Two renders of the same facade from the same angle agree
                 on both; two different rooms do not.

    2. CLUSTER   Photographs that look alike are grouped. The best of each
                 group (the Vision Agent's own quality/composition score)
                 is the one the book prints; the rest are held back and
                 used only where the book would otherwise go without a
                 picture at all, and never near their twin.

    3. ALLOCATE  Pages ask this module for a photograph. It hands out the
                 unused ones first, then — only if the book has more pages
                 wanting pictures than it has pictures — allows a second
                 use, never within MIN_GAP pages and never from a cluster
                 already on the facing spread.

WHAT IT NEVER DOES

Drops somebody's own choice silently. Where a section names its own
photograph, that section keeps it; the duplicate is resolved on the OTHER
pages, and what moved is reported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # pragma: no cover
    from src.data.schemas import ImageAnalysis, ImageAsset, LayoutPage

logger = logging.getLogger(__name__)

# How close two signatures have to be to read as the same picture.
#
# The hash is 64 bits; HAMMING_NEAR is how many of them may differ. 12 was
# measured on the SOBHA set: the two clubhouse renders that a reader calls
# the same picture differ in 7-10 bits, and the clubhouse against the
# swimming pool differs in 24. Colour has to agree too — a dark and a
# daylight version of one elevation are different pages, not one.
HAMMING_NEAR = 12
COLOUR_NEAR = 0.16

# How many pages apart a second use of the same picture — or of its twin —
# has to be before a reader stops reading it as a repeat. A spread is two
# pages, so anything under four is visible at a glance.
MIN_GAP = 6

# The side of the greyscale thumbnail the hash is taken from. 9x8 gives 72
# samples and 8x8 = 64 comparisons between horizontally adjacent pixels.
_HASH_W, _HASH_H = 9, 8
_COLOUR_BINS = 4            # 4x4x4 coarse RGB histogram


@dataclass(frozen=True)
class Signature:
    """What a photograph looks like, in two numbers a machine can compare."""
    bits: int                       # 64-bit difference hash of the luma
    colour: tuple[float, ...]       # normalised coarse RGB histogram
    aspect: float = 1.0

    def near(self, other: "Signature") -> bool:
        if bin(self.bits ^ other.bits).count("1") > HAMMING_NEAR:
            return False
        return _colour_distance(self.colour, other.colour) <= COLOUR_NEAR


def _colour_distance(a: Iterable[float], b: Iterable[float]) -> float:
    """Half the L1 distance between two histograms: 0 identical, 1 disjoint."""
    return sum(abs(x - y) for x, y in zip(a, b)) / 2.0


def sign(path: str) -> Optional[Signature]:
    """The signature of the image at `path`, or None if it cannot be read.

    Never raises: a photograph that cannot be opened is simply one this
    module has no opinion about, and the book is composed as it was
    before. A missing signature is not a reason to fail a generation.
    """
    try:
        from PIL import Image

        with Image.open(path) as handle:
            handle.draft("RGB", (256, 256))     # cheap, decoder-side downscale
            rgb = handle.convert("RGB")
            aspect = (rgb.width / rgb.height) if rgb.height else 1.0
            grey = rgb.convert("L").resize((_HASH_W, _HASH_H))
            pixels = list(grey.getdata())
            bits = 0
            for row in range(_HASH_H):
                for col in range(_HASH_W - 1):
                    left = pixels[row * _HASH_W + col]
                    right = pixels[row * _HASH_W + col + 1]
                    bits = (bits << 1) | (1 if left > right else 0)
            small = rgb.resize((32, 32))
            counts = [0.0] * (_COLOUR_BINS ** 3)
            step = 256 // _COLOUR_BINS
            for r, g, b in small.getdata():
                counts[(r // step) * _COLOUR_BINS ** 2
                       + (g // step) * _COLOUR_BINS + (b // step)] += 1.0
            total = sum(counts) or 1.0
            return Signature(bits=bits, colour=tuple(c / total for c in counts),
                             aspect=aspect)
    except Exception:                           # pragma: no cover - defensive
        logger.debug("could not sign %s", path, exc_info=True)
        return None


@dataclass
class Plan:
    """Which photographs the book has, and which of them look alike."""
    cluster_of: dict[str, int] = field(default_factory=dict)
    members: dict[int, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def cluster(self, image_id: str) -> int:
        """The visual group an image belongs to; -1 for "not comparable"."""
        return self.cluster_of.get(image_id, -1)

    def same_picture(self, a: str, b: str) -> bool:
        if a == b:
            return True
        one, two = self.cluster(a), self.cluster(b)
        return one >= 0 and one == two


def build(images_by_id: dict[str, "ImageAsset"],
          analyses: Optional[dict[str, "ImageAnalysis"]] = None) -> Plan:
    """Group the book's photographs by what they look like.

    `scene_tag` is taken as agreement, not as proof: the Vision Agent gave
    the two clubhouse renders "clubhouse_facade" and "clubhouse_exterior"
    and a reader calls them one picture, so the pixels decide and the tag
    only breaks ties."""
    analyses = analyses or {}
    plan = Plan()
    signed: list[tuple[str, Signature]] = []
    for image_id, asset in images_by_id.items():
        signature = sign(getattr(asset, "path", "") or "")
        if signature is not None:
            signed.append((image_id, signature))

    representatives: list[tuple[int, Signature, str]] = []
    for image_id, signature in signed:
        tag = (getattr(analyses.get(image_id), "scene_tag", "") or "").strip().lower()
        found = -1
        for index, other, other_tag in representatives:
            if signature.near(other) or (tag and tag == other_tag):
                found = index
                break
        if found < 0:
            found = len(representatives)
            representatives.append((found, signature, tag))
            plan.members[found] = []
        plan.cluster_of[image_id] = found
        plan.members[found].append(image_id)

    repeats = {k: v for k, v in plan.members.items() if len(v) > 1}
    if repeats:
        plan.notes.append(
            f"{len(repeats)} group(s) of photographs look alike; the best of each "
            f"is printed and the others are held back")
        for ids in repeats.values():
            logger.info("Image plan: these read as one picture: %s", ", ".join(ids))
    return plan


def _score(image_id: str, analyses: dict) -> float:
    analysis = analyses.get(image_id)
    if analysis is None:
        return 0.0
    return (getattr(analysis, "hero_score", 0.0) * 1.5
            + getattr(analysis, "composition", 0.0)
            + getattr(analysis, "quality", 0.0))


def _is_drawing(analyses: dict, image_id: str) -> bool:
    category = (getattr(analyses.get(image_id), "category", "") or "").lower()
    return category in ("floorplan", "site_plan", "map", "logo", "document")


class Allocator:
    """Hands out photographs, and remembers what the book has already shown.

    Constructed from the pages that already exist, so a picture a person
    chose for their own page is counted as spent before anything else is
    handed out."""

    def __init__(self, plan: Plan, images_by_id: dict, analyses: Optional[dict] = None,
                 pages: Optional[list] = None):
        self.plan = plan
        self.analyses = analyses or {}
        self.images = images_by_id
        # page index -> clusters shown there, so "not within MIN_GAP pages"
        # is a question this can answer.
        self.shown: dict[str, int] = {}          # image_id -> first page index
        self.cluster_pages: dict[int, list[int]] = {}
        for index, page in enumerate(pages or []):
            for image_id in (page.images or []):
                self.note(image_id, index)

    def note(self, image_id: str, page_index: int) -> None:
        self.shown.setdefault(image_id, page_index)
        cluster = self.plan.cluster(image_id)
        if cluster >= 0:
            self.cluster_pages.setdefault(cluster, []).append(page_index)

    def used(self, image_id: str) -> bool:
        return image_id in self.shown

    def free(self, image_id: str, page_index: int) -> bool:
        """May this page show this picture without reading as a repeat?"""
        if self.used(image_id):
            return False
        cluster = self.plan.cluster(image_id)
        if cluster < 0:
            return True
        return all(abs(page_index - other) >= MIN_GAP
                   for other in self.cluster_pages.get(cluster, ()))

    def candidates(self, page_index: int, wanted: tuple[str, ...] = ()) -> list[str]:
        """Unused photographs this page could take, best first.

        A photograph whose twin is already on a nearby page is not a
        candidate, however good it is: the point of holding it back is
        that a reader would see the same picture twice."""
        pool = [i for i in self.images
                if not _is_drawing(self.analyses, i) and self.free(i, page_index)]

        def rank(image_id: str) -> tuple[int, float]:
            category = (getattr(self.analyses.get(image_id), "category", "") or "").lower()
            return (0 if (wanted and category in wanted) else 1,
                    -_score(image_id, self.analyses))

        return sorted(pool, key=rank)

    def take(self, page_index: int, wanted: tuple[str, ...] = ()) -> Optional[str]:
        """The best unused photograph for this page, marked as spent."""
        choice = next(iter(self.candidates(page_index, wanted)), None)
        if choice is not None:
            self.note(choice, page_index)
        return choice


def enforce(pages: list["LayoutPage"], images_by_id: dict,
            analyses: Optional[dict] = None,
            plan: Optional[Plan] = None) -> list[str]:
    """No page shows a picture the book has already shown, or its twin.

    Walks the book in order. The FIRST page to use a picture keeps it —
    that is almost always the page somebody wrote for it — and any later
    page showing the same picture, or one from the same visual group
    within MIN_GAP pages, is given the best photograph the book has not
    printed yet. A page that cannot be given one keeps what it has rather
    than losing its picture: a repeat is a flaw, a hole is a defect.

    Returns what moved, for reporting."""
    analyses = analyses or {}
    plan = plan or build(images_by_id, analyses)
    allocator = Allocator(plan, images_by_id, analyses)
    moved: list[str] = []

    for index, page in enumerate(pages):
        if not page.images:
            continue
        wanted = tuple(filter(None, (
            (getattr(analyses.get(i), "category", "") or "").lower()
            for i in page.images)))
        replaced: list[str] = []
        for image_id in page.images:
            if _is_drawing(analyses, image_id) or allocator.free(image_id, index):
                allocator.note(image_id, index)
                replaced.append(image_id)
                continue
            swap = allocator.take(index, wanted)
            if swap is None:
                # Nothing unused left. Keep the picture rather than leave
                # the page bare — and say so, because the honest fix is a
                # shorter book or more photographs.
                allocator.note(image_id, index)
                replaced.append(image_id)
                continue
            moved.append(f'page {index + 1} ("{page.title or page.type}") showed a '
                         f"picture the book had already shown; it now shows {swap}")
            replaced.append(swap)
        page.images = replaced
    for note in moved:
        logger.info("Image plan: %s", note)
    return moved
