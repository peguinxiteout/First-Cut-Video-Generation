"""One grade on every photograph in a book.

    "The photographs are shot by different people on different days and
     printed as they came. Grade them as a set."

A property's pictures arrive from three or four places — the architect's
renders, a photographer's shoot, a drone operator, and whatever the
developer had on file — and each has its own white balance and its own
contrast. Printed side by side they read as a scrapbook. One light grade
across all of them is what makes them read as one set, and it is the first
thing a picture editor does.

WHERE IT HAPPENS, AND WHY NOT IN CSS

The obvious place is a CSS filter on the image, and that was the first
version. It works and it costs 45MB: a filtered image cannot be embedded
in the PDF as the JPEG it already is, so Chromium rasterises every
photograph in the book. The brochures went from 8MB to 52MB, which is a
file nobody can email.

So it happens once, on the normalised copy, in Pillow — the same pass that
already flattens transparency and downsamples to 300dpi — and the PDF
embeds ordinary JPEGs again.

WHAT IT MAY AND MAY NOT DO

It is deliberately slight: a point of contrast, a point of saturation, a
degree of warmth. A brochure may style a photograph; it may not alter what
the photograph shows, and anything strong enough to be noticed as a grade
is strong enough to be a claim about the building.

It is applied to PHOTOGRAPHS only. A floor plan, a site plan, a document
or a logo is not a photograph: a filter on one of those is a change to
what it states, and asset_prep keeps them out by category.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grade:
    """A tier's grade. 1.0 is "leave it alone" on all three enhancers."""
    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 1.0
    # Positive is warmer (red up, blue down), as a fraction of full scale.
    # ±0.06 is about as far as this may go before it reads as a filter.
    warmth: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (self.saturation == 1.0 and self.contrast == 1.0
                and self.brightness == 1.0 and self.warmth == 0.0)


# Each tier's grade sits with its palette, because that is what it is for:
# the photography has to look like it belongs on the paper it is printed on.
GRADES = {
    # Espresso and ivory are a warm pair, and the tier's photography is
    # lit to match. A touch down on saturation, a touch up on contrast.
    "luxury_modern": Grade(saturation=0.96, contrast=1.03, warmth=0.035),
    # Teal and gold are a cool pair, so the photography is pulled a
    # half-step cooler to sit on them.
    "meridian": Grade(saturation=1.02, contrast=1.04, warmth=-0.025),
    # A warm white sheet and a bright green. This tier's pictures are
    # often the weakest of the three, and a heavy grade makes a weak
    # photograph look worse rather than better — so it is barely a grade.
    "compass": Grade(saturation=1.04, contrast=1.02, brightness=1.02, warmth=0.015),
}

DEFAULT = Grade()


def for_template(template: str) -> Grade:
    """The grade for a template id, an old template name, or a category."""
    if template in GRADES:
        return GRADES[template]
    from src.agents.brochure.catalog import ALIASES, TEMPLATES

    spec = TEMPLATES.get(ALIASES.get(template, template))
    category = spec.category if spec else (template or "")
    legacy = {"luxury": "luxury_modern", "mid": "meridian", "affordable": "compass"}.get(category)
    return GRADES.get(legacy or "", DEFAULT)


def apply(image, grade: Grade):
    """Return a graded copy. Never mutates the image it is given."""
    if grade.is_identity:
        return image
    from PIL import Image, ImageEnhance

    out = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    if grade.saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(grade.saturation)
    if grade.contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(grade.contrast)
    if grade.brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(grade.brightness)
    if grade.warmth and out.mode == "RGB":
        # A straight per-channel gain rather than a curve: at this
        # strength the difference is invisible and a gain cannot clip a
        # channel into a colour cast the way a curve can.
        shift = grade.warmth
        red, green, blue = out.split()
        red = red.point(lambda v: min(255, int(v * (1 + shift))))
        blue = blue.point(lambda v: min(255, int(v * (1 - shift))))
        out = Image.merge("RGB", (red, green, blue))
    return out
