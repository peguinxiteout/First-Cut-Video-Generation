"""The offline path: brief derivations, archetype planning, and the
deterministic composer that writes the post when no model answers."""
from __future__ import annotations

import pytest

from src.agents.blog import archetypes, brief, composer, editor, fact_check, planner, writer
from src.agents.blog.article import full_markdown, plain
from src.agents.blog.normaliser import find_placeholders
from src.data.schemas import BlogInput, JourneyTime, NearbyPlace, NewsItem, ResearchPack
from src.evaluation import blog_metrics, blog_rubric

BRIEFS = {
    "locality_guide": dict(
        topic="Is Shela, Ahmedabad a good place to buy a home in 2026?", audience="young families",
        sections=["strengths", "weaknesses", "location_analysis", "pricing_value", "buyer_considerations"],
        custom_sections=["Why Ahmedabad is becoming a major real estate investment hub"],
        local_facts=["Shela is about 4 km from SP Ring Road", "2BHK apartments are listed at ₹55–75 lakh",
                     "Several CBSE schools within 3 km", "Ahmedabad Metro Phase 2 under construction"],
        brand_name="Skyline Homes", cta_text="Book a weekday site visit"),
    "comparison": dict(
        topic="Whitefield vs Sarjapur Road, Bengaluru: which suits IT professionals?",
        sections=["comparison", "pros_cons"],
        local_facts=["Whitefield has an operational Purple Line metro connection",
                     "Sarjapur Road has no operational metro line yet",
                     "Whitefield 2BHK resale homes are listed at ₹85 lakh–1.1 crore"]),
    "explainer": dict(topic="What is RERA carpet area? A guide for first-time buyers"),
    "buyers_guide": dict(topic="5 tips for first-time home buyers", key_points=["Budget for registration costs"]),
    "investment_analysis": dict(topic="Investing in Gurugram Sector 150 apartments",
                                sections=["investment_perspective"]),
    "how_to": dict(topic="How to register a property in Pune: step by step"),
    "market_update": dict(topic="Ahmedabad property market trends and outlook"),
}


def make(archetype_key: str, **extra) -> BlogInput:
    return BlogInput(blog_id="B", **{**BRIEFS[archetype_key], **extra})


# --- brief derivations ----------------------------------------------------------

def test_location_audience_and_keyword_are_derived_from_the_topic():
    ctx = brief.context(make("locality_guide"))
    assert (ctx.location.short, ctx.location.city, ctx.location.state) == ("Shela", "Ahmedabad", "Gujarat")
    assert ctx.audience["id"] == "families"
    assert ctx.primary_keyword == "homes in shela ahmedabad"
    assert ctx.keyword_display == "homes in Shela Ahmedabad"


def test_comparison_options_are_extracted_without_the_city():
    assert brief.comparison_options(make("comparison")) == ["Whitefield", "Sarjapur Road"]


def test_placeholder_contact_details_in_the_brief_are_ignored():
    bi = BlogInput(blog_id="B", topic="T", cta_url="https://...", cta_phone="+91 XXXXX XXXXX",
                   cta_email="info@example.com")
    assert brief.contact_details(bi) == {"url": "", "phone": "", "email": ""}
    real = BlogInput(blog_id="B", topic="T", cta_url="https://skyline.in/visit", cta_phone="+91 98100 12345",
                     cta_email="sales@skyline.in")
    assert brief.contact_details(real) == {"url": "https://skyline.in/visit", "phone": "+91 98100 12345",
                                           "email": "sales@skyline.in"}


def test_an_explicit_brief_beats_derivation():
    bi = make("buyers_guide", article_type="explainer", primary_keyword="first home checklist",
              location="Bopal, Ahmedabad")
    ctx = brief.context(bi)
    assert archetypes.classify(bi, ctx) == "explainer"
    assert ctx.primary_keyword == "first home checklist" and ctx.location.short == "Bopal"


@pytest.mark.parametrize("key", list(BRIEFS))
def test_each_topic_gets_its_own_archetype(key):
    assert archetypes.classify(make(key)) == key


def test_requested_sections_are_never_dropped_from_the_plan():
    bi = make("explainer", sections=["amenities", "investment_perspective"])
    outline = planner.deterministic_outline(bi)
    covered = {c for p in outline.sections for c in p.covers}
    assert {"amenities", "investment_perspective"} <= covered


def test_custom_sections_are_planned_in_the_authors_words():
    outline = planner.deterministic_outline(make("locality_guide"))
    assert "Why Ahmedabad is becoming a major real estate investment hub" in outline.section_headings
    # ...before the closing decision sections.
    assert outline.sections[-1].slot == "due_diligence"


# --- the composer -----------------------------------------------------------------

def compose(bi: BlogInput, research: ResearchPack | None = None):
    outline = planner.deterministic_outline(bi, research)
    content = writer.finish(composer.compose(bi, outline, [], research), bi, outline, research)
    return outline, content


@pytest.mark.parametrize("key", list(BRIEFS))
def test_the_offline_writer_never_invents_a_figure(key):
    """Every figure in a composed post traces to the brief or research."""
    bi = make(key)
    _, content = compose(bi)
    text = plain(full_markdown(content))
    evidence = fact_check.evidence_for(bi, None)
    assert fact_check.untraceable_figures(text, evidence) == []
    assert not find_placeholders(text)


@pytest.mark.parametrize("key", list(BRIEFS))
def test_the_offline_post_is_structured_and_clean_after_editing(key):
    bi = make(key)
    outline, content = compose(bi)
    edited, report = editor.run(bi, content)
    assert not report.removed_claims, report.removed_claims
    assert not report.blocking
    m = blog_metrics.measure(edited, bi, outline_text=outline.model_dump_json())
    assert m["tables"] >= 1 and m["checklist_items"] >= 3
    assert m["long_paragraphs"] == 0 and m["duplicate_sentences"] == 0
    assert m["markup_leaks"] == 0 and m["missing_sections"] == 0
    assert m["conclusion_decisive"] and m["cta_present"] and not m["hook_generic"]


def test_the_composer_uses_research_with_its_provenance():
    bi = make("locality_guide")
    pack = ResearchPack(place_name="Shela, Ahmedabad",
                        nearby=[NearbyPlace(name="Delhi Public School", kind="school", distance_km=2.1)],
                        journeys=[JourneyTime(destination="international airport", distance_km=18.4,
                                              free_flow_minutes=35)],
                        news=[NewsItem(title="Metro phase 2 reaches Shela", source="Times of India",
                                       url="https://timesofindia.indiatimes.com/x")])
    _, content = compose(bi, pack)
    text = full_markdown(content)
    assert "Delhi Public School" in text and "18.4 km" in text and "clear traffic" in text
    assert "Measured" in text or "measured" in text
    assert any(r.kind == "news" for r in content.references)
    assert fact_check.untraceable_figures(plain(text), fact_check.evidence_for(bi, pack)) == []


def test_without_facts_the_post_says_so_instead_of_inventing():
    bi = BlogInput(blog_id="B", topic="Is Bopal, Ahmedabad a good place to live?")
    _, content = compose(bi)
    text = full_markdown(content)
    assert "could not verify" in text
    m = blog_metrics.measure(content, bi)
    dims = {d.name: d for d in blog_rubric.score_dimensions(m, location_known=True)}
    assert dims["content_quality"].score < 90          # honest: no specifics, lower value


def test_references_come_only_from_real_sources():
    bi = make("comparison", reference_links=["BMRCL | https://english.bmrc.co.in", "https://example.com/x",
                                             "https://www.99acres.com/whitefield"])
    _, content = compose(bi)
    urls = [r.url for r in content.references if r.url]
    assert "https://english.bmrc.co.in" in urls and "https://rera.karnataka.gov.in" in urls
    assert not any("example.com" in u or "99acres" in u for u in urls)
