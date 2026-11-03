"""The content model: raw form fields made ready to print — and nothing
added that the person did not give."""
from __future__ import annotations

import pytest

from src.agents.brochure import facts
from src.data.schemas import Landmark, PropertyContact, PropertyInput, PropertyLocation, UnitType


@pytest.mark.parametrize("raw,label,detail", [
    ("1 Bed + 1 Toilet", "1 BHK", "1 bath"),
    ("3 Bed + 3 Toilet + Study", "3 BHK + Study", "3 baths"),
    ("1 BHK-A", "1 BHK", "Type A"),
    ("2 BHK-B", "2 BHK", "Type B"),
    ("3 BHK + Study", "3 BHK + Study", ""),
    ("4 BHK", "4 BHK", ""),
])
def test_configurations_read_as_marketing_terms(raw, label, detail):
    unit = facts.parse_unit(raw)
    assert unit.label == label and unit.detail == detail


def test_the_configuration_summary():
    units = [facts.parse_unit(n) for n in ("1 Bed + 1 Toilet", "2 Bed + 2 Toilet", "3 Bed + 2 Toilet",
                                           "3 Bed + 3 Toilet", "3 Bed + 3 Toilet + Study", "4 Bed + 4 Toilet")]
    assert facts.config_summary(units) == "1, 2, 3 & 4 BHK"
    assert facts.config_summary([facts.parse_unit("1 BHK-A"), facts.parse_unit("2 BHK-B")]) == "1 & 2 BHK"


@pytest.mark.parametrize("raw,tidy", [
    ("1611 – 1915 sq.ft", "1,611 – 1,915 sq ft"),
    ("334.76 sq.ft", "334.76 sq ft"),
    ("660", "660 sq ft"),
])
def test_areas_are_tidied(raw, tidy):
    assert facts.tidy_area(raw) == tidy


@pytest.mark.parametrize("raw,tidy,start", [
    ("₹29.95 Lakh – ₹31.15 Lakh", "₹29.95 – 31.15 Lakh", "₹29.95 Lakh"),
    ("Rs 52 lakhs onwards", "₹52 Lakh onwards", "₹52 Lakh"),
    ("₹1.2 Cr", "₹1.2 Cr", "₹1.2 Cr"),
])
def test_prices_are_tidied(raw, tidy, start):
    assert facts.tidy_price(raw) == tidy
    assert facts.price_from(raw) == start


def test_selling_points_become_headline_and_detail():
    c = facts.parse_claim("Pre-certified IGBC Gold project ensuring 45% water savings and 70% common area "
                          "solar power.", "affordable")
    assert c.headline == "Pre-certified IGBC Gold project"
    assert c.detail.startswith("Ensures 45% water savings")
    assert c.topic == "green" and c.figure == "45%"
    assert [s.value for s in c.stats] == ["45%", "70%"]
    colon = facts.parse_claim("Strict privacy design: Across all 420 units, zero homes face each other.")
    assert colon.headline == "Strict privacy design" and colon.detail.startswith("Across all 420 units")
    located = facts.parse_claim("Located on the Bhiwandi-Kalyan Corridor near the proposed metro.", "affordable")
    assert located.topic == "connectivity" and located.detail.startswith("On the Bhiwandi")


def test_luxury_register_polishes_words_not_facts():
    c = facts.parse_claim("Three massive themed clubhouses spanning over 77,850 sq.ft.", "luxury")
    assert "massive" not in c.headline.lower() and "77,850" in c.figure


def test_amenity_zones_keep_named_zones_and_numbers_whole():
    zones = facts.amenity_zones([
        "Kids' Zone: Aqua Park with water slides, Toddlers' Pad, and Skating Lane.",
        "Landscaped gardens featuring 1,200+ trees, fragrance garden, and picnic lawn.",
        "Clubhouse with multi-purpose hall, indoor games, gym, and terrace lounge.",
        "Swimming Pool and dedicated Kids' Pool.",
    ])
    by_title = {z.title: z for z in zones}
    assert by_title["Kids' Zone"].items == ["Aqua Park with water slides", "Toddlers' Pad", "Skating Lane"]
    garden = next(z for z in zones if z.topic == "landscape")
    assert "1,200+ trees" in garden.items                       # not split at the thousands comma
    club = next(z for z in zones if z.topic == "clubhouse")
    assert club.items[0] == "Multi-purpose hall"                 # not "Clubhouse with ..."
    pool = next(z for z in zones if z.topic == "pool")
    assert pool.items == ["Swimming Pool", "Dedicated Kids' Pool"]


def test_places_are_tidied_and_ordered_by_what_buyers_weigh():
    places = facts.places([Landmark(name="VITS Upper Thane", distance="~1 km", category="hotel"),
                           Landmark(name="Rajnouli Metro Station", distance="~500 m (2 min)"),
                           Landmark(name="Orion Mall", distance="Proximity")])
    assert places[0].name == "Rajnouli Metro Station" and places[0].distance == "≈ 500 m · 2 min"
    assert places[-1].kind == "hotel"
    assert next(p for p in places if p.name == "Orion Mall").distance == ""   # never "Proximity"


def test_the_audience_is_shortened_not_rewritten():
    phrase = facts.audience_phrase("Budget-conscious nuclear families, first-time homebuyers, and working "
                                   "professionals in the Kalyan-Bhiwandi hubs seeking an affordable life.")
    assert phrase == "budget-conscious nuclear families and first-time homebuyers"


def test_the_whole_model():
    prop = PropertyInput(
        property_id="p", name="Brigade Citrine", developer="Brigade Enterprises Ltd",
        location=PropertyLocation(city="Bengaluru", area="Old Madras Road"), category="mid",
        price="₹80 Lakh onwards", project_size="4.3 Acres, 420 Units", possession="June 2029",
        unit_types=[UnitType(name="1 Bed + 1 Toilet", carpet_area="434.86 sq.ft"),
                    UnitType(name="4 Bed + 4 Toilet", carpet_area="1540.22 sq.ft")],
        contact=PropertyContact(phone="+91 91376 11941", website="https://www.brigadegroup.com/"))
    c = facts.build(prop, "mid")
    assert c.config == "1 & 4 BHK" and c.noun == "Residences"
    assert c.price_from == "₹80 Lakh" and c.area_range == "434.86 – 1,540.22 sq ft"
    assert [(f.value, f.unit) for f in c.figures] == [("4.3", "Acres"), ("420", "Residences")]
    assert c.website == "www.brigadegroup.com"
