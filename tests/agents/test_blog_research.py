"""The Research and Fact Check agents.

These two exist to close one gap. Ask a language model to write about
"Shela, Ahmedabad" and it produces confident paragraphs about schools,
metro lines and travel times, with no idea whether any of them are there.
Research goes and looks; the fact check makes the looking bind, because a
model handed a real distance will still write the rounder number if it
reads better.

The tests are organised around three claims:

  1. Research finds real things and writes them down with their
     provenance attached.
  2. Research CANNOT break a blog. No network, no model, no location in
     the topic — the post is still written, the way it was before any of
     this existed.
  3. The fact check catches an invented figure and names the sentence, so
     the revision is not blind.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.blog import fact_check, research
from src.data.schemas import (
    BlogContent, BlogInput, BlogSection, JourneyTime, NearbyPlace, NewsItem, ResearchPack,
)
from src.tools import ToolBox
from src.tools.base import ENCYCLOPEDIA, GEOCODE, NEARBY, NEWS, ROUTE, WEB_SEARCH, ToolResult


class FakeBox(ToolBox):
    """A toolbox with the network replaced by canned answers."""

    def __init__(self, answers: dict, enabled: bool = True):
        self._answers = answers
        self._enabled = enabled
        self.notes: list[str] = []
        self.diagnostics: list[str] = []
        self.asked: list[str] = []

    @property
    def enabled(self):
        return self._enabled

    def _result(self, capability, key=None):
        self.asked.append(capability)
        value = self._answers.get(key or capability)
        if value is None:
            note = f"no {capability} available"
            self.notes.append(note)
            return ToolResult.failed("fake", capability, note)
        return ToolResult("fake", capability, data=value)

    def geocode(self, place, **kwargs):
        return self._result(GEOCODE, f"geocode:{place}")

    def nearby_places(self, **kwargs):
        return self._result(NEARBY)

    def travel_distance(self, **kwargs):
        return self._result(ROUTE, f"route:{kwargs.get('label')}")

    def encyclopedia(self, query):
        return self._result(ENCYCLOPEDIA)

    def search_news(self, query, **kwargs):
        return self._result(NEWS)

    def search_web(self, query, **kwargs):
        return self._result(WEB_SEARCH)

    def keyword_suggestions(self, seed, **kwargs):
        return self._result("keywords")


SHELA = {"display_name": "Shela, Ahmedabad, Gujarat, India", "lat": 23.0087, "lon": 72.4923}

FULL_ANSWERS = {
    "geocode:Shela, Ahmedabad, India": SHELA,
    "geocode:international airport India": {"lat": 23.0772, "lon": 72.6347},
    "route:international airport": {"distance_km": 18.4, "free_flow_minutes": 35},
    NEARBY: [
        {"name": "Delhi Public School", "kind": "school", "distance_m": 2100},
        {"name": "Shela Lake Garden", "kind": "park", "distance_m": 900},
    ],
    ENCYCLOPEDIA: {"summary": "Shela is a locality in Ahmedabad on the ring road.",
                   "url": "https://en.wikipedia.org/wiki/Shela"},
    NEWS: [{"title": "Metro phase 2 reaches Shela", "source": "Times of India",
            "published": "12 Sep 2026", "url": "https://example.com/metro"}],
    WEB_SEARCH: [{"title": "Best areas to buy in Ahmedabad", "url": "https://x.test", "snippet": ""}],
}


def blog_input(**overrides):
    base = dict(blog_id="B", topic="Is Shela, Ahmedabad a good bet for first-time buyers?")
    base.update(overrides)
    return BlogInput(**base)


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    """No LLM in these tests; the deterministic plan takes over. That path
    has to work anyway — it is what runs when the model is rate-limited."""
    def refuse(*_a, **_k):
        raise RuntimeError("no model in tests")
    monkeypatch.setattr("src.agents.blog.research.call_with_fallback", refuse)
    monkeypatch.setattr("src.agents.blog.fact_check.call_with_fallback", refuse)


# --- 1. research finds real things ------------------------------------------

def test_research_gathers_location_amenities_journeys_news_and_background(monkeypatch):
    monkeypatch.setattr("src.agents.blog.research._plan", lambda _bi: {
        "subject": "Shela", "place": "Shela, Ahmedabad, India", "country_code": "in",
        "searches": ["best areas ahmedabad"], "news_query": "Shela real estate"})
    pack = research.run(blog_input(), FakeBox(FULL_ANSWERS))

    assert pack.located and pack.place_name.startswith("Shela")
    # Order comes from the tool (nearest first — see test_tools.py); the
    # agent carries it through and converts metres to the kilometres a
    # sentence would use.
    assert {p.name for p in pack.nearby} == {"Shela Lake Garden", "Delhi Public School"}
    assert {p.distance_km for p in pack.nearby} == {0.9, 2.1}
    assert pack.journeys[0].distance_km == 18.4
    assert pack.journeys[0].free_flow_minutes == 35
    assert pack.news[0].source == "Times of India"
    assert "ring road" in pack.background
    assert "https://en.wikipedia.org/wiki/Shela" in pack.sources


def test_every_figure_carries_what_kind_of_figure_it_is():
    """A number handed to a model without its caveat comes back out as a
    bare assertion, and "35 minutes to the airport" is a very different
    promise from "about 35 minutes in clear traffic"."""
    pack = ResearchPack(
        place_name="Shela",
        nearby=[NearbyPlace(name="DPS", kind="school", distance_km=2.1)],
        journeys=[JourneyTime(destination="airport", distance_km=18.4, free_flow_minutes=35)],
        news=[NewsItem(title="Metro reaches Shela", source="TOI")])
    block = research.as_prompt_block(pack)

    assert "straight-line" in block, "a straight line is not a drive"
    assert "clear traffic" in block, "a free-flow time is not a commute"
    assert "ANNOUNCED" in block, "a headline is not an event"


def test_competitor_results_are_labelled_as_not_facts():
    """These are pages nobody has vetted. They are useful for the SHAPE —
    what everyone covers, and what nobody does — and repeating their
    claims as true is how a blog inherits a competitor's mistakes."""
    block = research.as_prompt_block(ResearchPack(
        competitor_topics=["Best areas to buy in Ahmedabad 2026"]))
    assert "NOT facts" in block


# --- 2. research cannot break a blog ----------------------------------------

def test_with_tools_turned_off_the_pack_is_empty_and_says_why():
    pack = research.run(blog_input(), FakeBox({}, enabled=False))
    assert pack.is_empty
    assert any("turned off" in note for note in pack.notes)


def test_every_tool_failing_still_produces_a_pack(monkeypatch):
    monkeypatch.setattr("src.agents.blog.research._plan", lambda _bi: {
        "subject": "Shela", "place": "Shela, Ahmedabad, India", "country_code": "in",
        "searches": [], "news_query": ""})
    pack = research.run(blog_input(), FakeBox({}))
    assert isinstance(pack, ResearchPack)
    assert pack.is_empty and pack.notes


def test_a_topic_with_no_place_in_it_is_not_an_error():
    pack = research.run(blog_input(topic="How to read a home loan agreement"), FakeBox({}))
    assert isinstance(pack, ResearchPack)


def test_the_place_is_still_found_when_the_planning_model_is_unavailable():
    """The deterministic fallback. It is crude and only has to beat
    nothing — but it is also what runs whenever the model is rate-limited,
    which on a free tier is often."""
    plan = research._plan_without_a_model(
        blog_input(topic="5 reasons to buy in Whitefield, Bengaluru"))
    assert "Whitefield" in plan["place"]


def test_an_empty_pack_tells_the_writer_to_say_less_rather_than_more():
    """The one instruction that matters when research found nothing. A
    model given a blank where it expected facts fills the blank in."""
    block = research.as_prompt_block(ResearchPack())
    assert "do not state any local facts" in block


# --- 3. the fact check binds the research -----------------------------------

def content_with(*sentences: str) -> BlogContent:
    return BlogContent(title="T", intro_markdown="",
                       sections=[BlogSection(heading="H", body_markdown=" ".join(sentences))],
                       conclusion_markdown="")


PACK = ResearchPack(
    journeys=[JourneyTime(destination="airport", distance_km=18.4, free_flow_minutes=35)],
    nearby=[NearbyPlace(name="Delhi Public School", kind="school", distance_km=2.1)])


def test_a_figure_that_matches_the_evidence_needs_no_model_call():
    """Most claims are settled by arithmetic. Paying a model to check
    whether 18.4 is in a list of numbers would be slower and less
    reliable than checking."""
    result = fact_check.run(blog_input(), content_with("The airport is 18.4 km away."), PACK)
    assert result.ok
    assert result.verified


def test_an_invented_figure_is_caught_and_the_sentence_is_named():
    """The failure this whole layer exists to prevent: research found 18.4
    km and the draft says 8. Naming the sentence is what makes the
    revision useful — before this the loop was blind."""
    result = fact_check.run(
        blog_input(), content_with("The airport is just 8 km from the project."), PACK)
    assert not result.ok
    assert any("8 km" in claim for claim in result.unsupported)
    note = fact_check.as_revision_note(result)
    assert "8 km" in note and "rewrite" in note.lower()


def test_the_author_s_own_facts_count_as_evidence():
    """Anything the person vouched for is allowed, whether or not research
    corroborates it. It is their claim to make."""
    result = fact_check.run(
        blog_input(local_facts=["Apartments listed at ₹11,000 per sq ft"]),
        content_with("Apartments here are listed at ₹11,000 per sq ft."), PACK)
    assert result.ok


def test_a_sentence_that_mixes_a_real_figure_with_an_invented_one_is_not_waved_through():
    """The cheap pass requires ALL figures to match, precisely so that one
    true number cannot carry a false one past the check."""
    result = fact_check.run(
        blog_input(), content_with("The airport is 18.4 km away and prices rose 40% last year."),
        PACK)
    assert not result.ok


def test_opinions_are_not_fact_checking_failures():
    """Flagging a view as unsupported would train the writer out of having
    one, which makes the posts worse rather than more accurate. Only
    falsifiable statements are checked."""
    result = fact_check.run(
        blog_input(),
        content_with("Shela suits families who want space more than a short commute."), PACK)
    assert result.ok
    assert not result.unsupported


def test_a_post_with_nothing_checkable_says_so():
    result = fact_check.run(blog_input(), content_with("A calm, well-planned neighbourhood."), None)
    assert result.ok and result.notes


def test_when_the_checker_cannot_reach_the_model_it_errs_towards_flagging():
    """Being wrong in that direction removes a sentence from the post.
    Being wrong the other way prints an invented fact."""
    result = fact_check.run(
        blog_input(), content_with("Roughly a fifteen minute drive covers 22 km."), PACK)
    assert not result.ok
    assert result.notes


def test_extracting_claims_finds_figures_and_ignores_prose():
    claims = fact_check.extract_claims(content_with(
        "The area is pleasant.", "It sits 4.2 km from the ring road.",
        "Prices start at ₹75 lakh.", "Residents like it."))
    assert len(claims) == 2
    assert all(any(ch.isdigit() for ch in c) for c in claims)
