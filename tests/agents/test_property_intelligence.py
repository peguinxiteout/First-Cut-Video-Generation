from __future__ import annotations

from src.agents.brochure.property_intelligence import _coerce_facts_to_strings


def test_coerces_nested_dict_into_flat_string():
    """Regression test: exactly the shape that crashed a real generation —
    the model echoed property.json's nested location object verbatim."""
    facts = {"location": {"city": "Noida", "area": "", "address": ""}}
    coerced = _coerce_facts_to_strings(facts)
    assert coerced == {"location": "city: Noida"}


def test_coerces_empty_list_by_dropping_the_key():
    """Regression test: an empty amenities list (facts.amenities = [])
    crashed validation. An empty fact isn't useful, so it's dropped
    entirely rather than coerced into an empty string."""
    facts = {"amenities": []}
    coerced = _coerce_facts_to_strings(facts)
    assert coerced == {}


def test_coerces_nested_dict_with_all_fields_populated():
    facts = {"contact": {"phone": "+91-0", "email": "a@b.com", "website": ""}}
    coerced = _coerce_facts_to_strings(facts)
    assert coerced == {"contact": "phone: +91-0, email: a@b.com"}


def test_coerces_populated_list_by_joining():
    facts = {"amenities": ["Pool", "Gym", "Clubhouse"]}
    coerced = _coerce_facts_to_strings(facts)
    assert coerced == {"amenities": "Pool, Gym, Clubhouse"}


def test_leaves_plain_strings_untouched():
    facts = {"city": "Noida", "configuration": "3 BHK"}
    coerced = _coerce_facts_to_strings(facts)
    assert coerced == {"city": "Noida", "configuration": "3 BHK"}


def test_drops_none_values():
    facts = {"price": None, "city": "Noida"}
    coerced = _coerce_facts_to_strings(facts)
    assert coerced == {"city": "Noida"}


def test_the_exact_three_field_combination_from_the_real_error(sample_property_input):
    """Reproduces the exact facts payload from the reported error, end to
    end through PropertyKnowledge validation — must not raise."""
    from src.data.schemas import PropertyKnowledge

    raw_result = {
        "facts": {
            "location": {"city": "Noida", "area": "", "address": ""},
            "amenities": [],
            "contact": {"phone": "", "email": "", "website": ""},
            "name": "Sky Villas at Estate 128",
        },
        "selling_points": ["Prime Noida location"],
        "missing_information": [],
    }
    raw_result["facts"] = _coerce_facts_to_strings(raw_result["facts"])
    knowledge = PropertyKnowledge.model_validate(raw_result)  # must not raise
    assert "amenities" not in knowledge.facts  # empty list -> dropped
    assert "contact" not in knowledge.facts  # all-empty dict -> dropped
    assert knowledge.facts["location"] == "city: Noida"
    assert knowledge.facts["name"] == "Sky Villas at Estate 128"
