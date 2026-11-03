"""Research depth follows the archetype; official sources become evidence;
an unreachable network costs two lookups, not ten."""
from __future__ import annotations

from src.agents.blog import fact_check, research
from src.data.schemas import BlogContent, BlogInput, BlogSection
from tests.agents.test_blog_research import FULL_ANSWERS, FakeBox


def no_model(monkeypatch):
    def refuse(*_a, **_k):
        raise RuntimeError("no model in tests")
    monkeypatch.setattr("src.agents.blog.research.call_with_fallback", refuse)


def test_an_explainer_does_not_go_looking_for_nearby_pharmacies(monkeypatch):
    no_model(monkeypatch)
    box = FakeBox(FULL_ANSWERS)
    pack = research.run(BlogInput(blog_id="B", topic="What is RERA carpet area?"), box)
    assert pack.archetype == "explainer"
    assert "geocode" not in box.asked and "nearby" not in box.asked


def test_a_locality_guide_does(monkeypatch):
    no_model(monkeypatch)
    box = FakeBox(FULL_ANSWERS)
    pack = research.run(BlogInput(blog_id="B", topic="Is Shela, Ahmedabad a good place to buy a home?"), box)
    assert pack.archetype == "locality_guide" and pack.located and pack.nearby


def test_only_authoritative_results_are_kept_as_official_sources_and_they_count_as_evidence(monkeypatch):
    no_model(monkeypatch)
    answers = dict(FULL_ANSWERS)
    answers["web_search"] = [
        {"title": "GujRERA project search", "url": "https://gujrera.gujarat.gov.in/search",
         "snippet": "Projects must register before advertising; 1,234 projects listed."},
        {"title": "Top 10 flats in Shela", "url": "https://www.99acres.com/shela", "snippet": "Best deals 50% off"},
    ]
    bi = BlogInput(blog_id="B", topic="Is Shela, Ahmedabad a good place to buy a home?")
    pack = research.run(bi, FakeBox(answers))
    assert [r.url for r in pack.official_sources] == ["https://gujrera.gujarat.gov.in/search"]
    draft = BlogContent(title="T", sections=[BlogSection(heading="H", body_markdown="GujRERA lists 1,234 projects.")])
    assert fact_check.run(bi, draft, pack).ok
    assert "OFFICIAL SOURCES" in research.as_prompt_block(pack)


class DownBox(FakeBox):
    def _result(self, capability, key=None):
        self.asked.append(capability)
        note = f"{capability} failed: https://x.test: ProxyError: Tunnel connection failed: 403 Forbidden"
        self.notes.append(note)
        from src.tools.base import ToolResult
        return ToolResult.failed("fake", capability, note)


def test_an_unreachable_network_stops_research_early_with_a_plain_note(monkeypatch):
    no_model(monkeypatch)
    box = DownBox({})
    pack = research.run(BlogInput(blog_id="B", topic="Is Shela, Ahmedabad a good place to buy a home?"), box)
    assert len(box.asked) <= 3
    assert any("could not be reached" in n for n in pack.notes)
    assert not any("ProxyError" in n for n in pack.notes)
