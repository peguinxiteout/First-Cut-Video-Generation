"""The evaluation rubric: deterministic measurements, the five weighted
parameters with a one-line reason each, the LLM blend, the gate, and the
feedback a writer receives."""
from __future__ import annotations

from src.core.types import QualityGateResult
from src.data.schemas import BlogContent, BlogEvaluationResult, BlogFAQ, BlogInput, BlogReference, BlogSection
from src.evaluation import blog_metrics, blog_rubric, blog_scoring

BRIEF = BlogInput(blog_id="B", topic="Is Shela, Ahmedabad a good place to buy a home?", audience="young families",
                  local_facts=["Shela is about 4 km from SP Ring Road", "Several CBSE schools within 3 km"])

GOOD_BODY = [
    ("Homes in Shela Ahmedabad: the facts at a glance",
     "These facts matter most for young families in Shela.\n\n| What | Detail | Source |\n|---|---|---|\n"
     "| SP Ring Road | About 4 km from Shela | Supplied |\n| Several CBSE schools | Within 3 km | Supplied |"),
    ("Where Shela works for your family",
     "Your school runs stay short in Shela.\n\n- **Schools:** several CBSE schools are within 3 km.\n"
     "- **Roads:** Shela is about 4 km from SP Ring Road.\n\nCheck the commute yourself at peak hour."),
    ("The trade-offs you should weigh in Shela",
     "Newer areas in Shela have less resale history, so fair prices are harder to judge. Construction nearby "
     "can mean dust and noise for a few years."),
    ("Before you book in Shela: a checklist",
     "Verify every point before you pay.\n\n- [ ] Check the project on GujRERA.\n- [ ] Confirm the carpet area.\n"
     "- [ ] Get an all-inclusive cost sheet.\n\n1. **Shortlist.** Pick two projects.\n2. **Visit.** Go at peak hour."),
]


def good_content(**kw) -> BlogContent:
    base = dict(
        title="Is Shela, Ahmedabad a Good Place to Buy a Home?",
        intro_markdown=("If you are weighing homes in Shela Ahmedabad, the question is whether daily life works "
                        "for your family. Here is what is on record and what you should verify."),
        sections=[BlogSection(heading=h, body_markdown=b) for h, b in GOOD_BODY],
        key_takeaways=["Shela suits young families who want schools close by."],
        faqs=[BlogFAQ(question="Is Shela good for families?", answer_markdown="It can be if schools matter most.")],
        conclusion_markdown=("Shela suits young families who want schools close by. If you need metro access "
                             "today, look elsewhere — and verify registration before you pay anything."),
        cta_heading="Book a visit", cta_markdown="Book a weekday site visit and bring the checklist with you.",
        meta_title="Is Shela a Good Place to Buy a Home?",
        meta_description=("Weighing homes in Shela Ahmedabad? Verified facts, honest trade-offs and a due-diligence "
                          "checklist for young families."),
        slug="homes-in-shela-ahmedabad", tags=["Shela", "Ahmedabad", "Locality guide"],
        primary_keyword="homes in shela ahmedabad",
        references=[BlogReference(title="GujRERA", url="https://gujrera.gujarat.gov.in", kind="official")],
    )
    base.update(kw)
    return BlogContent(**base)


def score(content, llm=None):
    m = blog_metrics.measure(content, BRIEF)
    return m, {d.name: d for d in blog_rubric.score_dimensions(m, llm_scores=llm, location_known=True)}


def test_a_well_formed_post_passes_every_dimension():
    m, dims = score(good_content())
    failing = {n: (d.score, d.findings) for n, d in dims.items() if not d.passed}
    assert not failing
    assert m["untraceable_figures"] == 0 and m["placeholders"] == 0 and m["tables"] == 1
    assert m["checklist_items"] == 3 and m["brief_fact_coverage"] == 1.0


def test_walls_of_text_hurt_readability_and_say_where():
    wall = " ".join(["Shela has many things that a family might like to consider when buying."] * 9)
    content = good_content(sections=[BlogSection(heading=GOOD_BODY[0][0], body_markdown=wall)]
                           + [BlogSection(heading=h, body_markdown=b) for h, b in GOOD_BODY[1:]])
    m, dims = score(content)
    assert m["long_paragraphs"] >= 1 and m["duplicate_sentences"] >= 1
    assert dims["structure_readability"].score < 80
    assert any("paragraph" in f for f in dims["structure_readability"].findings)


def test_invented_figures_sink_content_quality():
    content = good_content(conclusion_markdown="Prices rose 40% last year and the airport is 12 km away. "
                                               "Shela suits families who want schools close by.")
    m, dims = score(content)
    assert m["untraceable_figures"] == 2
    assert not dims["content_quality"].passed
    assert "40%" in " ".join(dims["content_quality"].findings)
    assert "40%" in dims["content_quality"].summary                 # the one-line reason names it


def test_placeholders_and_a_missing_cta_hurt_quality_and_engagement():
    m, dims = score(good_content(cta_markdown="", conclusion_markdown="Visit example.com for details."))
    assert m["placeholders"] >= 1 and not m["cta_present"]
    assert dims["engagement"].score <= 75 and "call to action" in " ".join(dims["engagement"].findings)
    assert any("Placeholders" in f for f in dims["content_quality"].findings)


def test_keyword_stuffing_hurts_seo_and_stray_geography_hurts_relevance():
    stuffed = " ".join(["Homes in Shela Ahmedabad are popular."] * 12) + " Unlike Mumbai, it is calm."
    m, dims = score(good_content(intro_markdown=stuffed))
    assert m["keyword_density"] > 2.5 and m["stray_places"] == 1
    assert dims["seo"].score < 85 and any("stuffing" in f for f in dims["seo"].findings)
    assert any("mumbai" in f.lower() for f in dims["relevance"].findings)


def test_sales_language_and_return_promises_are_penalised():
    m, dims = score(good_content(conclusion_markdown="A world-class, iconic address with guaranteed returns. "
                                                     "Shela suits families."))
    assert m["promotional_terms"] == 2 and m["prohibited_claims"] == 1
    assert any("Sales language" in f for f in dims["engagement"].findings)
    assert any("Prohibited" in f for f in dims["content_quality"].findings)


def test_generic_headings_and_a_soft_conclusion_hurt_structure():
    content = good_content(sections=[BlogSection(heading="Strengths", body_markdown=GOOD_BODY[1][1]),
                                     BlogSection(heading="Conclusion", body_markdown=GOOD_BODY[2][1])],
                           conclusion_markdown="Overall, there are many things to think about here.")
    m, dims = score(content)
    assert m["generic_headings"] == 2 and not m["conclusion_decisive"]
    assert not dims["structure_readability"].passed


def test_the_llm_score_is_blended_not_trusted_outright():
    _, plain = score(good_content())
    _, blended = score(good_content(), llm={"structure_readability": 0})
    assert blended["structure_readability"].llm_score == 0
    assert blended["structure_readability"].score == round(
        0.6 * plain["structure_readability"].deterministic_score, 1)


def test_gate_revises_when_any_dimension_is_below_threshold_even_if_the_average_is_high():
    _, dims = score(good_content())
    dims = list(dims.values())
    dims[-1] = dims[-1].model_copy(update={"score": 40.0, "passed": False})
    evaluation = BlogEvaluationResult(dimensions=dims)
    evaluation.quality_score = blog_scoring.compute_quality_score(evaluation)
    assert evaluation.quality_score > 78
    assert blog_scoring.apply_quality_gate(evaluation) == QualityGateResult.REVISE


def test_feedback_is_dimension_level_and_actionable():
    content = good_content(conclusion_markdown="Prices rose 40% last year.", cta_markdown="")
    m = blog_metrics.measure(content, BRIEF)
    dims = blog_rubric.score_dimensions(m, location_known=True)
    lines = blog_rubric.feedback(dims, {"content_quality": "Remove the 40% claim in the conclusion."})
    text = "\n".join(lines)
    margins = [float(l.split("(")[1].split("/")[0]) - float(l.split("target ")[1].split(")")[0]) for l in lines]
    assert margins == sorted(margins)                           # worst first
    assert any(l.startswith("Content quality") for l in lines)
    assert "40%" in text and "brief or research" in text
    assert "Remove the 40% claim" in text


def test_records_without_dimensions_are_scored_the_old_way():
    legacy = BlogEvaluationResult(grounding=90, writing_quality=90, structural_completeness=100,
                                  editorial_integrity=100)
    assert blog_scoring.compute_quality_score(legacy) == 93.0


def test_inconsistent_and_implausible_figures_are_detected():
    text = ("The airport is 18 km away. Later we say the airport is 22 km from here. "
            "Prices are ₹2 crore per sq ft and grew 150% overnight.")
    assert blog_metrics.inconsistent_figures(text) == ["airport: 18 km vs 22 km"]
    assert set(blog_metrics.implausible_units(text)) == {"150%", "crore per sq ft"}


def test_five_parameters_whose_weights_sum_to_one_and_the_score_is_weighted():
    _, dims = score(good_content())
    assert list(dims) == ["content_quality", "relevance", "structure_readability", "engagement", "seo"]
    assert abs(sum(d.weight for d in dims.values()) - 1.0) < 1e-9
    values = [d.model_copy(update={"score": s}) for d, s in zip(dims.values(), (100, 50, 50, 50, 50))]
    evaluation = BlogEvaluationResult(dimensions=values)
    assert blog_scoring.compute_quality_score(evaluation) == 65.0          # 0.30*100 + 0.70*50, not 60
    for d in dims.values():
        assert d.summary and "\n" not in d.summary and len(d.summary) <= 180


def test_the_critic_reason_becomes_the_one_line_explanation():
    _, dims = score(good_content(), llm={"seo": 70})
    m = blog_metrics.measure(good_content(), BRIEF)
    with_reason = {d.name: d for d in blog_rubric.score_dimensions(
        m, llm_scores={"seo": 70}, location_known=True, llm_reasons={"seo": "Keyword missing from one heading."})}
    assert with_reason["seo"].summary == "Keyword missing from one heading."
    assert dims["relevance"].summary                                      # measured reason when no model note


def test_the_prompt_lists_the_same_parameters_and_weights_as_the_config():
    from src.core.config import load_yaml_config

    prompt = load_yaml_config("blog/prompts.yaml")["critic"]
    assert "$dimensions" in prompt and "$dimension_scores" in prompt and "$dimension_reasons" in prompt
    spec = blog_rubric.dimension_spec()
    assert [d["name"] for d in spec] == ["content_quality", "relevance", "structure_readability", "engagement", "seo"]
    assert [d["weight"] for d in spec] == [0.30, 0.20, 0.20, 0.15, 0.15]
