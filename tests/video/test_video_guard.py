from __future__ import annotations

import pytest

from src.agents.brochure import facts, strategy
from src.agents.brochure.property_intelligence import ledger_from_form
from src.agents.video.guard import VideoGuard, figures, number_words
from src.data.schemas import PropertyInput, PropertyKnowledge, PropertyLocation


@pytest.fixture
def prop() -> PropertyInput:
    return PropertyInput(
        property_id="G1", name="Aurelia Heights", developer="Northlight", category="mid",
        location=PropertyLocation(city="Pune", area="Baner"), configuration="3 BHK", area="1,450 sq ft",
        price="Rs 1.20 Cr", possession="December 2028", rera_number="P52100012345",
        usps=["Two minutes from the Baner metro station", "25 acres of open space"],
        amenities=["Swimming pool", "Clubhouse of 25,000 sq ft"])


def _guard(prop, knowledge=None) -> VideoGuard:
    knowledge = knowledge or ledger_from_form(prop)
    return VideoGuard(prop, knowledge, facts.build(prop, "mid"), strategy.for_category("mid"))


def test_a_figure_in_the_facts_passes(prop):
    guard = _guard(prop)
    for text in ("Possession in December 2028.", "A 3 BHK home.", "Set in 25 acres.", "Rs 1.2 Cr onwards.",
                 "Call P52100012345.", "1450 sq ft of space.", "Clubhouse of 25000 sq ft."):
        assert guard.why(text) == "", text


def test_a_figure_that_is_not_in_the_facts_is_refused_and_named(prop):
    guard = _guard(prop)
    assert guard.why("Set in 50 acres.") == "'50' is not a figure in the property's facts"
    assert "'4'" in guard.why("A 4 BHK home.")
    assert not guard.ok("Only 12 homes left.")


def test_a_run_of_digits_inside_another_figure_is_not_a_fact(prop):
    # "2" and "202" occur inside 2028; the brochure's substring check would let them through.
    guard = _guard(prop)
    assert "'2'" in guard.why("2 towers.")
    assert "'202'" in guard.why("Since 202.")


def test_thousands_separators_and_trailing_zeros_do_not_matter():
    assert figures("1,450 and 1450 and 2.50 and 2.5") == {"1450", "2.5"}


def test_number_words_are_figures_too(prop):
    guard = _guard(prop)
    for text in ("Twenty amenities.", "Seven towers.", "A hundred homes.", "Twenty-five acres of parks."):
        assert guard.why(text), text
    assert "'twenty'" in guard.why("Twenty amenities.")


def test_a_number_word_is_allowed_when_the_facts_use_it_or_its_value(prop):
    guard = _guard(prop)
    assert guard.why("Two minutes from the metro.") == ""          # the selling point says "Two"
    knowledge = PropertyKnowledge(facts={"towers": "3 towers"})
    assert _guard(prop, knowledge).why("Three towers.") == ""      # 3 is a fact


def test_a_fact_only_the_ledger_knows_is_allowed(prop):
    knowledge = PropertyKnowledge(facts={"floors": "31 storeys"}, specifications=["Vitrified tiles, 600 mm"])
    guard = _guard(prop, knowledge)
    assert guard.why("A 31 storey tower.") == "" and guard.why("600 mm tiles.") == ""
    assert guard.why("A 32 storey tower.")


def test_a_damaged_character_is_refused(prop):
    assert "damaged character" in _guard(prop).why("Welcome to Const�ra.")


def test_the_brochures_other_rules_still_apply(prop):
    guard = _guard(prop)
    assert "describes the picture" in guard.why("This image shows the pool.")
    assert "register avoids" in guard.why("A palatial address.")
    assert guard.why("") == "it is empty" and guard.why("   ") == "it is empty"


def test_a_repeat_is_refused_unless_the_check_is_off(prop):
    guard = _guard(prop)
    guard.keep("Book a site visit.")
    assert guard.why("Book a site visit.") == "it repeats a line already used"
    assert guard.why("Book a site visit.", repeat=False) == ""      # an overlay may say it again


def test_ok_agrees_with_why(prop):
    guard = _guard(prop)
    assert guard.ok("Welcome to Aurelia Heights.") and not guard.ok("Set in 50 acres.")
    assert guard.numbers_ok("3 BHK") and not guard.numbers_ok("7 BHK")


def test_zero_in_a_compound_is_not_a_quantity_but_zero_brokerage_is(prop):
    guard = _guard(prop)
    assert guard.why("A zero-waste community.") == ""
    assert "'zero'" in guard.why("Zero brokerage.")


def test_word_and_figure_helpers():
    assert number_words("Twenty-five and THREE, no Someone") == {"twenty", "five", "three"}
    assert figures("RAA09999") == {"9999"}
