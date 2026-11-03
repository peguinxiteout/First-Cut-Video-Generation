"""The brochure form: what the person types, and the property it becomes.

    "Completely remove the Excel-upload workflow for entering property
    details. Collect information exclusively through Streamlit using the
    simplest practical form."

The form asks for the eleven things a brochure is made from, in this order:

     1. property / project name          *
     2. location (city, locality)        * city
     3. property type (apartment, villa)
     4. category (Affordable / Mid-Tier / Luxury)   *
     5. starting price / price range
     6. configurations and carpet areas  * at least one configuration
     7. key selling points (USPs)
     8. amenities and facilities
     9. target audience
    10. developer name and logo
    11. contact details and call to action   * a phone number or an email

    * mandatory. Everything else is optional and used only when given.

plus the pictures, each in its own slot — project photographs (at least
one), floor plans, a master plan, a location map, the developer's logo —
and a few optional extras a brochure prints when they exist (possession,
RERA registration, project size, nearby places, a tagline).

`write_property` turns a completed form into the property folder every
workflow reads: property.json, images/, and images/roles.json recording
which slot each picture came from.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.core.ids import slugify


class ConfigRow(BaseModel):
    configuration: str = ""         # "2 BHK"
    carpet_area: str = ""           # "650 – 720 sq ft"
    price: str = ""                 # "Rs 52 Lakh onwards"


class BrochureForm(BaseModel):
    name: str = ""
    city: str = ""
    locality: str = ""
    address: str = ""
    property_type: str = "Apartments"
    category: str = ""              # Affordable | Mid-Tier | Luxury
    orientation: str = "Landscape"  # Landscape | Portrait | Digital
    price: str = ""
    configurations: list[ConfigRow] = Field(default_factory=list)
    usps: str = ""                  # one per line
    amenities: str = ""             # one per line, or comma separated
    target_audience: str = ""
    developer: str = ""
    phone: str = ""
    email: str = ""
    website: str = ""
    cta: str = ""
    # optional extras
    tagline: str = ""
    possession: str = ""
    rera_number: str = ""
    project_size: str = ""
    nearby: str = ""                # "Name | 2 km | school", one per line
    description: str = ""
    brand_color: str = ""           # "#E31837"


CATEGORIES = ("Affordable", "Mid-Tier", "Luxury")
PROPERTY_TYPES = ("Apartments", "Villas", "Row houses", "Penthouses", "Plots", "Townhouses",
                  "Studio apartments", "Mixed-use")
LANDMARK_CATEGORIES = ("transit", "school", "hospital", "retail", "work", "airport", "leisure", "road")


def lines(text: str) -> list[str]:
    """One item per line — or, if typed on one line, comma separated."""
    raw = [x.strip(" •-\t") for x in re.split(r"[\r\n]+", text or "")]
    raw = [x for x in raw if x]
    if len(raw) == 1 and raw[0].count(",") >= 2:
        raw = [x.strip() for x in raw[0].split(",") if x.strip()]
    seen, out = set(), []
    for item in raw:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def missing(form: BrochureForm, photo_count: int) -> list[str]:
    """The mandatory fields still empty, in plain words."""
    out = []
    if not form.name.strip():
        out.append("the project name")
    if not form.city.strip():
        out.append("the city")
    if not form.category.strip():
        out.append("the category")
    if not any(r.configuration.strip() for r in form.configurations):
        out.append("at least one configuration (for example 2 BHK)")
    if photo_count < 1:
        out.append("at least one project photograph")
    if not (form.phone.strip() or form.email.strip()):
        out.append("a phone number or an email for enquiries")
    return out


def _landmarks(text: str) -> list[dict]:
    out = []
    for line in lines(text):
        parts = [p.strip() for p in re.split(r"\s*[|;]\s*", line)]
        name = parts[0]
        distance = parts[1] if len(parts) > 1 else ""
        category = parts[2].lower() if len(parts) > 2 else ""
        if not category:
            low = name.lower()
            category = next((c for c, keys in (
                ("transit", ("metro", "station", "railway", "bus")), ("airport", ("airport",)),
                ("school", ("school", "college", "university")), ("hospital", ("hospital", "clinic")),
                ("retail", ("mall", "market", "store")), ("work", ("tech park", "it park", "office", "sez")),
                ("road", ("highway", "road", "expressway"))) if any(k in low for k in keys)), "")
        out.append({"name": name, "distance": distance, "category": category})
    return out


def to_property(form: BrochureForm, property_id: Optional[str] = None) -> dict:
    """The property.json the workflows read."""
    from src.agents.brochure.catalog import normalise_category

    configs = [r for r in form.configurations if r.configuration.strip()]
    data = {
        "property_id": property_id or slugify(form.name) or "property",
        "name": form.name.strip(),
        "developer": form.developer.strip() or None,
        "location": {"city": form.city.strip(), "area": form.locality.strip() or None,
                     "address": form.address.strip() or None},
        "property_type": form.property_type.strip() or None,
        "category": normalise_category(form.category) or None,
        "price": form.price.strip() or None,
        "unit_types": [{"name": r.configuration.strip(), "carpet_area": r.carpet_area.strip(),
                        "price": r.price.strip()} for r in configs],
        "usps": lines(form.usps),
        "amenities": lines(form.amenities),
        "target_audience": form.target_audience.strip() or None,
        "contact": {"phone": form.phone.strip() or None, "email": form.email.strip() or None,
                    "website": form.website.strip() or None},
        "cta": form.cta.strip() or None,
        "tagline": form.tagline.strip() or None,
        "possession": form.possession.strip() or None,
        "rera_number": form.rera_number.strip() or None,
        "project_size": form.project_size.strip() or None,
        "landmarks": _landmarks(form.nearby),
        "description": form.description.strip() or None,
        "brand_color": form.brand_color.strip() or None,
    }
    return data


def _safe_name(name: str, used: set) -> str:
    stem, dot, ext = name.rpartition(".")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem or name)[:60] or "image"
    ext = (ext or "jpg").lower()
    candidate = f"{stem}.{ext}"
    n = 2
    while candidate in used:
        candidate = f"{stem}_{n}.{ext}"
        n += 1
    used.add(candidate)
    return candidate


def write_property(form: BrochureForm, uploads: dict[str, list[tuple[str, bytes]]],
                   root: Path, property_id: Optional[str] = None) -> Path:
    """Write the property folder. `uploads` maps a slot — photo, floorplan,
    masterplan, map, logo — to (filename, bytes) pairs."""
    data = to_property(form, property_id)
    folder = root / data["property_id"]
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    for old in images.iterdir():
        if old.is_file():
            old.unlink()
    roles: dict[str, str] = {}
    used: set = set()
    for slot in ("photo", "floorplan", "masterplan", "map", "logo"):
        for filename, payload in uploads.get(slot, []) or []:
            name = _safe_name(f"{slot}_{filename}" if slot != "photo" else filename, used)
            (images / name).write_bytes(payload)
            if slot != "photo":
                roles[name] = slot
    (images / "roles.json").write_text(json.dumps(roles, indent=1), encoding="utf-8")
    (folder / "property.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return folder


def template_for(form: BrochureForm) -> str:
    """The catalog id for this form's category and orientation."""
    from src.agents.brochure.catalog import resolve

    return resolve(None, form.category, form.orientation).id
