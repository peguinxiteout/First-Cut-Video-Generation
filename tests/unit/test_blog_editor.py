"""The Editor: accuracy, compliance and repetition fixes on a whole draft,
and the validation report of what only a writer can fix."""
from __future__ import annotations

from src.agents.blog import editor
from src.data.schemas import (
    BlogContent, BlogFAQ, BlogInput, BlogSection, FactCheckResult, JourneyTime, NearbyPlace, ResearchPack,
)


def brief(**kw) -> BlogInput:
    base = dict(blog_id="B", topic="Is Shela, Ahmedabad a good place to buy?",
                local_facts=["2BHK apartments are listed at ₹55–75 lakh", "Shela is about 4 km from SP Ring Road"])
    base.update(kw)
    return BlogInput(**base)


def draft(*bodies: str, **kw) -> BlogContent:
    sections = [BlogSection(heading=f"Section about the area {i}", body_markdown=b) for i, b in enumerate(bodies)]
    base = dict(title="T", intro_markdown="If you are weighing homes in Shela, here is what is on record.",
                sections=sections, conclusion_markdown="Shela suits families who want space.",
                cta_heading="Visit", cta_markdown="Book a weekday visit and bring the checklist with you.")
    base.update(kw)
    return BlogContent(**base)


def test_untraceable_figures_are_removed_and_traceable_ones_kept():
    content, report = editor.run(brief(), draft(
        "2BHK apartments are listed at ₹55–75 lakh. Prices rose 40% last year. Shela is about 4 km from "
        "SP Ring Road."))
    body = content.sections[0].body_markdown
    assert "₹55–75 lakh" in body and "4 km" in body
    assert "40%" not in body
    assert any("40%" in c for c in report.removed_claims)
    assert any("traces to neither" in f for f in report.fixes)


def test_research_figures_count_as_evidence():
    pack = ResearchPack(journeys=[JourneyTime(destination="airport", distance_km=18.4, free_flow_minutes=35)],
                        nearby=[NearbyPlace(name="Delhi Public School", kind="school", distance_km=2.1)])
    content, report = editor.run(brief(), draft("The airport is 18.4 km by road, about 35 minutes in clear "
                                                "traffic. Delhi Public School is about 2.1 km away."), pack)
    assert "18.4 km" in content.sections[0].body_markdown
    assert "2.1 km" in content.sections[0].body_markdown
    assert not report.removed_claims


def test_untraceable_rows_and_items_go_but_the_table_stays():
    body = ("| What | Detail |\n|---|---|\n| 2BHK | ₹55–75 lakh |\n| Airport | 12 km |\n\n"
            "- Schools are close\n- The mall is 900 m away")
    content, _ = editor.run(brief(), draft(body))
    out = content.sections[0].body_markdown
    assert "| 2BHK | ₹55–75 lakh |" in out and "12 km" not in out
    assert "Schools are close" in out and "900 m" not in out


def test_claims_the_fact_checker_judged_unsupported_are_removed():
    check = FactCheckResult(unsupported=["Shela is about 4 km from SP Ring Road."])
    content, report = editor.run(brief(), draft("Shela is about 4 km from SP Ring Road. Families like it."),
                                 fact_check=check)
    assert "4 km" not in content.sections[0].body_markdown
    assert report.removed_claims


def test_return_promises_and_stray_cities_are_removed():
    content, report = editor.run(brief(), draft(
        "Buyers here enjoy guaranteed returns. Unlike Mumbai, prices here are moderate. Families like it."))
    body = content.sections[0].body_markdown
    assert "guaranteed" not in body and "Mumbai" not in body and "Families like it." in body
    assert len(report.removed_claims) == 2


def test_repeated_sentences_are_removed_across_sections():
    repeated = "School runs stay short because several schools are close to most projects."
    content, report = editor.run(brief(), draft(repeated + " Roads are good.", "Parks are nearby. " + repeated))
    assert content.sections[0].body_markdown.count("School runs") == 1
    assert "School runs" not in content.sections[1].body_markdown
    assert any("repeated" in f for f in report.fixes)


def test_a_cta_made_only_of_placeholders_is_rebuilt_from_the_brief():
    content, report = editor.run(brief(cta_text="Book a site visit", brand_name="Skyline Homes"),
                                 draft("Families like it here.", cta_markdown="Call +91 XXXXX XXXXX or visit [us](#)."))
    assert "XXXXX" not in content.cta_markdown
    assert content.cta_heading == "Book a site visit" and "Skyline Homes" in content.cta_markdown


def test_headings_titles_and_takeaways_lose_markup():
    content, _ = editor.run(brief(), draft("Body text here for the section.", title="## **Shela guide**",
                                           key_takeaways=["- **Schools** are close"]))
    assert content.title == "Shela guide"
    assert content.key_takeaways == ["Schools are close"]


def test_takeaways_are_derived_when_missing():
    content, report = editor.run(brief(), draft(
        "Shela suits families who want space more than a short commute. Roads are good."))
    assert content.key_takeaways and "Shela suits families" in content.key_takeaways[0]


def test_what_cannot_be_fixed_mechanically_is_reported():
    content, report = editor.run(brief(), BlogContent(
        title="T", intro_markdown="Intro text that is long enough to count here.",
        sections=[BlogSection(heading="Strengths", body_markdown="A world-class address. The airport is 18 km "
                              "away from the site."),
                  BlogSection(heading="Location Analysis", body_markdown="Short.")],
        conclusion_markdown="", cta_markdown="", faqs=[BlogFAQ(question="Q?", answer_markdown="A.")]),
        ResearchPack(journeys=[JourneyTime(destination="airport", distance_km=18, free_flow_minutes=30),
                               JourneyTime(destination="airport (alt)", distance_km=22, free_flow_minutes=40)]))
    codes = {i.code for i in report.issues}
    assert {"generic_heading", "promotional", "thin_section"} <= codes
    assert not report.blocking


def test_revision_note_names_removed_claims_and_open_issues():
    _, report = editor.run(brief(), draft("Prices rose 40% last year. A world-class address for families."))
    lines = editor.as_revision_note(report)
    text = "\n".join(lines)
    assert "40%" in text and "REMOVED" in text
    assert "Sales language" in text
