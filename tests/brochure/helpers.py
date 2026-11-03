"""Synthetic properties for the brochure tests: a real property folder, as
the Streamlit form writes it, with photographs that are genuinely different
pictures (so near-duplicate detection does not merge them) and a drawing
that looks like a drawing."""
from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from src.data import property_form

PHOTOS = (  # filename, size, base colour
    ("exterior_facade.jpg", (1600, 1000), (150, 120, 90)),
    ("swimming_pool.jpg", (1600, 900), (40, 130, 170)),
    ("living_room.jpg", (1500, 1000), (190, 170, 140)),
    ("master_bedroom.jpg", (1000, 1400), (120, 90, 110)),
    ("kitchen.jpg", (1400, 1000), (200, 200, 190)),
    ("clubhouse.jpg", (1600, 1000), (90, 70, 50)),
    ("gym.jpg", (1500, 1000), (60, 60, 70)),
    ("garden_landscape.jpg", (1000, 1500), (60, 140, 70)),
)


def photo_bytes(size, colour, seed: int) -> bytes:
    rng = random.Random(seed)
    image = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(image)
    w, h = size
    for _ in range(40):
        x, y = rng.randrange(w), rng.randrange(h)
        r = rng.randrange(20, max(21, w // 4))
        tone = tuple(max(0, min(255, c + rng.randrange(-90, 90))) for c in colour)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tone)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def plan_bytes(size=(1400, 1000)) -> bytes:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle((60, 60, w - 60, h - 60), outline="black", width=6)
    draw.line((w // 2, 60, w // 2, h - 60), fill="black", width=4)
    draw.line((60, h // 2, w // 2, h // 2), fill="black", width=4)
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def form(category: str = "Mid-Tier", orientation: str = "Landscape", **extra) -> property_form.BrochureForm:
    values = dict(
        name="Aurelia Heights", city="Pune", locality="Baner", category=category,
        orientation=orientation, property_type="Apartments", price="Rs 85 Lakh onwards",
        configurations=[property_form.ConfigRow(configuration="2 BHK", carpet_area="720 sq ft",
                                                price="Rs 85 Lakh onwards"),
                        property_form.ConfigRow(configuration="3 BHK", carpet_area="1,040 sq ft")],
        usps="Two minutes from the Baner metro station\nSixty percent open green space\nClubhouse of 25,000 sq ft",
        amenities="Swimming pool\nClubhouse\nGymnasium\nKids' play area\nJogging track\nYoga lawn\n"
                  "Indoor games room\nParty hall\nCCTV security",
        target_audience="young families", developer="Northlight Developers",
        phone="+91 98765 43210", email="sales@northlight.example", website="https://northlight.example",
        cta="Book a site visit", rera_number="P52100012345", possession="December 2027",
        nearby="Baner Metro | 1.2 km | transit\nCity Hospital | 2.5 km | hospital\nWestend Mall | 3 km",
    )
    values.update(extra)
    return property_form.BrochureForm(**values)


def make_property(root: Path, category: str = "Mid-Tier", orientation: str = "Landscape",
                  photos: int = len(PHOTOS), plans: int = 1, **extra) -> Path:
    uploads = {"photo": [(name, photo_bytes(size, colour, i))
                         for i, (name, size, colour) in enumerate(PHOTOS[:photos])],
               "floorplan": [(f"plan_{i}.png", plan_bytes()) for i in range(plans)]}
    return property_form.write_property(form(category, orientation, **extra), uploads, root)
