"""The research toolbox.

Two properties matter more than any individual tool, and both are about
what happens when things go wrong, because everything here talks to a
free public API with no SLA.

A TOOL NEVER BREAKS A GENERATION. If research could fail a blog, the blog
would be less reliable WITH research than without it and nobody would
leave it on. So every failure path below ends in a ToolResult, never an
exception, and the blog still gets written.

A TOOL NEVER INVENTS. An empty result travels as an empty result with a
reason attached. That matters here more than most places: downstream of
these tools is a language model writing marketing copy about property,
and a model handed a blank where it expected facts will fill the blank
in — which is the exact failure this whole layer exists to prevent.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.tools import ToolBox, available_tools, toolbox_for
from src.tools.base import NEWS, WEB_SEARCH, Tool, ToolCache, ToolResult
from src.tools.http import HttpClient, HttpError
from src.tools.places import GeocodeTool, NearbyPlacesTool, haversine_m
from src.tools.routing import RouteTool


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def settings(**overrides):
    base = dict(tools_enabled=True, tools_timeout_s=5.0, tavily_api_key="")
    base.update(overrides)
    return SimpleNamespace(**base)


def client(responses, **kwargs):
    """An HttpClient whose transport replays `responses` in order."""
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        item = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    http = HttpClient(settings=settings(**kwargs), get=get, sleep=lambda _s: None)
    http.calls = calls
    return http


# --- the kill switch ---------------------------------------------------------

def test_nothing_reaches_the_network_when_tools_are_turned_off():
    """The guarantee the test suite itself relies on. Without it a research
    tool would quietly call a live API during tests, making the suite slow,
    flaky and dependent on somebody else's uptime."""
    http = HttpClient(settings=settings(tools_enabled=False),
                      get=lambda *a, **k: pytest.fail("a tool reached the network"))
    with pytest.raises(HttpError):
        http.get("https://example.invalid")

    box = ToolBox([WEB_SEARCH, NEWS], settings=settings(tools_enabled=False), http=http)
    for result in (box.search_web("anything"), box.search_news("anything")):
        assert not result.ok
        assert "turned off" in result.note


def test_the_whole_suite_runs_with_tools_disabled():
    """conftest.py sets this. If it ever stops, this test says so before a
    developer finds out from a rate-limit ban."""
    from src.core.config import settings as live

    assert live.tools_enabled is False, "tests/conftest.py must disable external tools"


# --- failure is never an exception -------------------------------------------

def test_a_dead_endpoint_becomes_a_result_not_a_crash():
    http = client([ConnectionError("network unreachable")])
    tool = GeocodeTool(settings=settings(), http=http, cache=None)
    result = tool.run(place="Shela, Ahmedabad")
    assert isinstance(result, ToolResult) and not result.ok
    assert "Shela" in result.note


def test_a_server_error_is_retried_then_reported():
    http = client([FakeResponse(503), FakeResponse(503)])
    result = GeocodeTool(settings=settings(), http=http, cache=None).run(place="X")
    assert not result.ok
    assert len(http.calls) == 2, "a 503 is worth one retry"


def test_a_404_is_an_answer_and_is_not_retried():
    http = client([FakeResponse(404)])
    GeocodeTool(settings=settings(), http=http, cache=None).run(place="X")
    assert len(http.calls) == 1


def test_a_nonsense_response_body_does_not_crash_the_tool():
    http = client([FakeResponse(200, payload=None, text="<html>not json</html>")])
    result = GeocodeTool(settings=settings(), http=http, cache=None).run(place="X")
    assert not result.ok


# --- an empty answer is an answer --------------------------------------------

def test_finding_nothing_is_reported_as_finding_nothing():
    """Distinct from a failure, and it has to be: 'the search failed' and
    'there are no schools nearby' mean different things to the writer, and
    conflating them is how a post ends up asserting something nobody
    checked."""
    http = client([FakeResponse(200, payload=[])])
    result = GeocodeTool(settings=settings(), http=http, cache=None).run(place="Atlantis")
    assert result.ok, "the tool worked; the answer was empty"
    assert result.empty
    assert "no entry" in result.note


# --- places ------------------------------------------------------------------

NOMINATIM_HIT = [{
    "display_name": "Shela, Ahmedabad, Gujarat, India", "lat": "23.0087", "lon": "72.4923",
    "type": "suburb", "address": {"city": "Ahmedabad", "state": "Gujarat", "country": "India"},
}]


def test_a_place_name_becomes_coordinates():
    http = client([FakeResponse(200, payload=NOMINATIM_HIT)])
    result = GeocodeTool(settings=settings(), http=http, cache=None).run(place="Shela, Ahmedabad")
    assert result.ok
    assert round(result.data["lat"], 3) == 23.009
    assert result.data["city"] == "Ahmedabad"


def test_openstreetmap_is_told_who_is_calling():
    """Nominatim and Overpass run on donated hardware and their usage
    policy requires an identifying User-Agent. Sending the default
    python-requests string is how a project gets its IP blocked — and it
    takes everyone on the same office network with it."""
    http = client([FakeResponse(200, payload=NOMINATIM_HIT)])
    GeocodeTool(settings=settings(), http=http, cache=None).run(place="Shela")
    agent = http.calls[0]["headers"]["User-Agent"]
    assert "AgenticContentPlatform" in agent and "python-requests" not in agent


OVERPASS_HIT = {"elements": [
    {"tags": {"name": "Delhi Public School", "amenity": "school"}, "lat": 23.02, "lon": 72.50},
    {"tags": {"name": "Delhi Public School", "amenity": "school"},      # same school, as a way
     "center": {"lat": 23.0201, "lon": 72.5001}},
    {"tags": {"amenity": "school"}, "lat": 23.03, "lon": 72.51},        # unnamed
    {"tags": {"name": "Shela Lake Garden", "leisure": "park"}, "lat": 23.0090, "lon": 72.4930},
]}


def test_nearby_places_come_back_nearest_first():
    http = client([FakeResponse(200, payload=OVERPASS_HIT)])
    result = NearbyPlacesTool(settings=settings(), http=http, cache=None).run(
        lat=23.0087, lon=72.4923, kinds=("school", "park"))
    distances = [p["distance_m"] for p in result.data]
    assert distances == sorted(distances)
    assert result.data[0]["name"] == "Shela Lake Garden"


def test_the_same_place_listed_twice_is_only_reported_once():
    """Overpass returns a school as a node AND as a way. A blog that lists
    it twice reads as though nobody checked it."""
    http = client([FakeResponse(200, payload=OVERPASS_HIT)])
    result = NearbyPlacesTool(settings=settings(), http=http, cache=None).run(
        lat=23.0087, lon=72.4923, kinds=("school", "park"))
    names = [p["name"] for p in result.data]
    assert len(names) == len(set(names)) == 2


def test_a_place_with_no_name_is_dropped():
    """An unnamed node cannot go in a sentence."""
    http = client([FakeResponse(200, payload=OVERPASS_HIT)])
    result = NearbyPlacesTool(settings=settings(), http=http, cache=None).run(
        lat=23.0087, lon=72.4923, kinds=("school",))
    assert all(p["name"] for p in result.data)


def test_distance_between_two_known_points_is_right():
    """Ahmedabad to Gandhinagar is about 23 km in a straight line."""
    metres = haversine_m(23.0225, 72.5714, 23.2156, 72.6369)
    assert 21_000 < metres < 25_000


# --- routing -----------------------------------------------------------------

def test_a_driving_route_reports_distance_and_free_flow_time():
    """The duration is named free_flow_minutes because that is what OSRM
    computes — it has no traffic model. Calling it a commute would be
    inventing the one number a reader would actually act on."""
    http = client([FakeResponse(200, payload={
        "code": "Ok", "routes": [{"distance": 18400.0, "duration": 2100.0}]})])
    result = RouteTool(settings=settings(), http=http, cache=None).run(
        from_lat=23.0, from_lon=72.5, to_lat=23.07, to_lon=72.63, label="airport")
    assert result.data["distance_km"] == 18.4
    assert result.data["free_flow_minutes"] == 35
    assert "free_flow" in "".join(result.data.keys())


def test_no_route_found_is_not_a_failure():
    http = client([FakeResponse(200, payload={"code": "NoRoute", "routes": []})])
    result = RouteTool(settings=settings(), http=http, cache=None).run(
        from_lat=23.0, from_lon=72.5, to_lat=51.5, to_lon=-0.12, label="London")
    assert result.ok and result.empty


# --- caching -----------------------------------------------------------------

def test_the_same_question_is_only_asked_once(tmp_path):
    """The revision loop re-runs the writer; it must not re-run the
    research. These APIs are rate-limited and a repeated question wastes
    an allowance that is shared with everyone else using them."""
    cache = ToolCache(tmp_path)
    http = client([FakeResponse(200, payload=NOMINATIM_HIT)])
    for _ in range(3):
        result = GeocodeTool(settings=settings(), http=http, cache=cache).run(place="Shela")
    assert len(http.calls) == 1
    assert result.from_cache


def test_a_corrupt_cache_entry_is_a_miss_not_an_error(tmp_path):
    cache = ToolCache(tmp_path)
    cache.put("nominatim", {"q": "Shela"}, {"lat": 1})
    next(tmp_path.glob("*.json")).write_text("{ not json", encoding="utf-8")
    assert cache.get("nominatim", {"q": "Shela"}, ttl_s=3600) is None


def test_a_toolbox_without_a_data_directory_simply_does_not_cache():
    """Constructing one from a bare settings object is legitimate — a
    script, a test, this package used from somewhere else — and should
    mean "no cache", not a crash halfway through a research run."""
    box = ToolBox([WEB_SEARCH], settings=settings())
    assert box._cache is None


def test_a_cache_that_cannot_be_written_does_not_break_the_tool(tmp_path):
    unwritable = tmp_path / "file-not-a-directory"
    unwritable.write_text("x", encoding="utf-8")
    cache = ToolCache(unwritable / "cache")
    http = client([FakeResponse(200, payload=NOMINATIM_HIT)])
    result = GeocodeTool(settings=settings(), http=http, cache=cache).run(place="Shela")
    assert result.ok


# --- the toolbox -------------------------------------------------------------

class AlwaysFails(Tool):
    name, capability = "always_fails", WEB_SEARCH

    def run(self, **kwargs):
        return ToolResult.failed(self.name, self.capability, "provider is down")


class AlwaysWorks(Tool):
    name, capability = "always_works", WEB_SEARCH

    def run(self, **kwargs):
        return ToolResult(self.name, self.capability, data=[{"title": "ok"}])


class NeedsAKey(Tool):
    name, capability, requires = "needs_a_key", WEB_SEARCH, ("tavily_api_key",)

    def run(self, **kwargs):
        return ToolResult(self.name, self.capability, data=[{"title": "premium"}])


def test_a_failed_provider_hands_over_to_the_next(monkeypatch):
    """The reason capabilities exist rather than vendors: the agent asking
    for a web search does not know, and must not know, that one of them
    was down."""
    from src.tools import PROVIDERS

    monkeypatch.setitem(PROVIDERS, WEB_SEARCH, (AlwaysFails, AlwaysWorks))
    box = ToolBox([WEB_SEARCH], settings=settings(), http=client([FakeResponse(200, payload={})]))
    result = box.search_web("anything")
    assert result.ok and result.tool == "always_works"
    # The operator sees which provider failed...
    assert any("down" in line for line in box.diagnostics)
    # ...and the reader is not told, because the fallback worked and
    # "DuckDuckGo was down so we used Tavily" is not news to them.
    assert not box.notes


def test_a_provider_without_its_key_is_skipped_silently(monkeypatch):
    from src.tools import PROVIDERS

    monkeypatch.setitem(PROVIDERS, WEB_SEARCH, (NeedsAKey, AlwaysWorks))
    box = ToolBox([WEB_SEARCH], settings=settings(), http=client([FakeResponse(200, payload={})]))
    assert box.search_web("anything").tool == "always_works"


def test_a_keyed_provider_is_preferred_once_the_key_exists(monkeypatch):
    from src.tools import PROVIDERS

    monkeypatch.setitem(PROVIDERS, WEB_SEARCH, (NeedsAKey, AlwaysWorks))
    box = ToolBox([WEB_SEARCH], settings=settings(tavily_api_key="k"),
                  http=client([FakeResponse(200, payload={})]))
    assert box.search_web("anything").tool == "needs_a_key"


def test_the_writer_has_no_tools_at_all():
    """Deliberate, not an oversight. A writer that can search mid-sentence
    wanders: it produces a different post every run and makes the result
    impossible to evaluate, because a weak post can no longer be traced to
    weak research or weak writing. Research happens in the research step
    and is written down; the writer consumes that."""
    assert toolbox_for("writer")._capabilities == ()
    assert not toolbox_for("writer").search_web("anything").ok


def test_each_agent_gets_only_the_tools_for_its_job():
    research = toolbox_for("research")._capabilities
    seo = toolbox_for("seo")._capabilities
    assert "geocode" in research and "geocode" not in seo
    assert "keywords" in seo and "keywords" not in research


def test_asking_for_a_capability_this_agent_lacks_is_a_clean_refusal():
    result = toolbox_for("seo").geocode("Shela")
    assert not result.ok and "no geocode tool" in result.note


def test_most_of_the_toolbox_works_with_no_configuration_at_all():
    """The reason research is on by default. OpenStreetMap, Wikipedia,
    Google News, GDELT and search autocomplete all answer without an
    account, so a fresh checkout gets real research rather than a feature
    that needs five signups first."""
    rows = available_tools()
    keyless = [r for r in rows if r["available"]]
    assert len(keyless) >= 6, [r["tool"] for r in rows if not r["available"]]
