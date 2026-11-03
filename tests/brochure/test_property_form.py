"""The Streamlit form and the property folder it writes (no spreadsheet)."""
from __future__ import annotations

import json

from src.data import property_form
from src.data.ingestion import discover_images, load_property_input
from tests.brochure.helpers import form, make_property


def test_lines_accepts_one_per_line_or_commas_and_drops_repeats():
    assert property_form.lines("Pool\n Gym \n\npool") == ["Pool", "Gym"]
    assert property_form.lines("Pool, Gym, Clubhouse") == ["Pool", "Gym", "Clubhouse"]
    assert property_form.lines("• Pool\n- Gym") == ["Pool", "Gym"]


def test_only_the_essentials_are_mandatory():
    empty = property_form.BrochureForm()
    missing = property_form.missing(empty, photo_count=0)
    assert len(missing) == 6
    complete = property_form.BrochureForm(
        name="A", city="Pune", category="Luxury", phone="1",
        configurations=[property_form.ConfigRow(configuration="3 BHK")])
    assert property_form.missing(complete, photo_count=1) == []
    # email instead of phone is enough
    complete.phone, complete.email = "", "a@b.example"
    assert property_form.missing(complete, photo_count=1) == []


def test_the_form_becomes_a_property(tmp_path):
    folder = make_property(tmp_path, photos=3, plans=2)
    prop = load_property_input(folder)
    assert prop.name == "Aurelia Heights" and prop.location.city == "Pune"
    assert prop.category == "mid"
    assert [u.name for u in prop.unit_types] == ["2 BHK", "3 BHK"]
    assert prop.unit_types[1].carpet_area == "1,040 sq ft"
    assert len(prop.usps) == 3 and "Clubhouse" in prop.amenities
    assert prop.contact.phone and prop.cta == "Book a site visit"
    assert prop.landmarks[0].category == "transit" and prop.landmarks[0].distance == "1.2 km"
    assert prop.landmarks[2].category == "retail"           # inferred from "Mall"


def test_each_upload_keeps_the_slot_it_came_from(tmp_path):
    folder = make_property(tmp_path, photos=2, plans=2)
    roles = json.loads((folder / "images" / "roles.json").read_text())
    assert sorted(roles.values()) == ["floorplan", "floorplan"]
    images = discover_images(folder)
    assert sorted(i.role or "photo" for i in images) == ["floorplan", "floorplan", "photo", "photo"]


def test_rewriting_a_property_replaces_its_pictures(tmp_path):
    make_property(tmp_path, photos=5)
    folder = make_property(tmp_path, photos=2, plans=0)
    assert len(discover_images(folder)) == 2


def test_category_and_orientation_pick_the_template():
    assert property_form.template_for(form("Luxury", "Portrait")) == "luxury_portrait"
    assert property_form.template_for(form("Affordable", "Landscape")) == "affordable_landscape"
