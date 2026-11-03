"""Shared fakes for blog tests: a scripted model that writes a BAD first
draft (every formatting and accuracy failure the evaluation found) and a
good revision, so the editor, the rubric and the revision loop can be
tested end to end without any API key."""
from __future__ import annotations

import json

MESSY_INTRO = (
    "In today's fast-paced world, buying a home is a big decision. Nestled in greenery, Shela boasts "
    "everything a family could want and prices rose 40% last year. Visit [our website](https://example.com) "
    "or call +91 XXXXX XXXXX today. " + " ".join(
        "Shela is a locality that many families consider when they look for a home in the west of the city "
        "and it has many features." for _ in range(3))
)

MESSY_SECTIONS = [
    {"heading": "## **Location Analysis**",
     "body_markdown": ("Key benefits: - Close to SP Ring Road - Several CBSE schools - Shela Lake Garden\n"
                       "The metro is only 2 km away and the airport is 12 km away.\n"
                       "## Connectivity\n"
                       "| Place | Distance |\n| SP Ring Road | about 4 km |\n| Airport | 12 km |"),
     "image_id": None},
    {"heading": "Strengths",
     "body_markdown": ("**Great** **value** **for** **money** and **world-class amenities**. "
                       "• Lake views • Wide roads • Quiet streets\n"
                       "2BHK apartments are listed at ₹55–75 lakh which is **excellent** value."),
     "image_id": None},
    {"heading": "Weaknesses",
     "body_markdown": "The Ahmedabad Metro Phase 2 is under construction, so commutes rely on roads for now.",
     "image_id": None},
    {"heading": "Buyer Considerations",
     "body_markdown": "Check these: 1. Verify RERA 2. Visit the site 3. Compare cost sheets",
     "image_id": None},
]

CLEAN_SECTIONS = [
    {"heading": "Homes in Shela Ahmedabad: what is on record",
     "body_markdown": ("For young families weighing Shela, three facts matter most.\n\n"
                       "| What | Detail | Source |\n|---|---|---|\n"
                       "| SP Ring Road | About 4 km from Shela | Supplied |\n"
                       "| Several CBSE schools | Within 3 km | Supplied |\n"
                       "| 2BHK apartments | Listed at ₹55–75 lakh | Supplied |\n\n"
                       "Treat supplied figures as a starting point and confirm them before you pay."),
     "image_id": None},
    {"heading": "Where Shela works well for young families",
     "body_markdown": ("- **Schools:** several CBSE schools are within 3 km, so school runs stay short.\n"
                       "- **Roads:** Shela is about 4 km from SP Ring Road.\n"
                       "- **Green space:** Shela Lake Garden is close to most new projects.\n\n"
                       "> **Tip:** Visit on a weekday morning to see the school-run traffic for yourself."),
     "image_id": None},
    {"heading": "The trade-offs to weigh before choosing Shela",
     "body_markdown": ("The Ahmedabad Metro Phase 2 is under construction, so plan around today's roads.\n\n"
                       "Newer areas also have less resale history, which makes fair prices harder to judge."),
     "image_id": None},
    {"heading": "Before you book in Shela: a due-diligence checklist",
     "body_markdown": ("Run every shortlisted home through these checks.\n\n"
                       "- [ ] Look up the project on GujRERA and match the registration number.\n"
                       "- [ ] Confirm the carpet area in writing.\n"
                       "- [ ] Get an all-inclusive cost sheet before paying a booking amount.\n"
                       "- [ ] Have a lawyer review the title documents."),
     "image_id": None},
]


def writer_payload(clean: bool) -> dict:
    return {
        "title": "Is Shela, Ahmedabad a good place to buy a home?",
        "intro_markdown": (MESSY_INTRO if not clean else
                           "If you are weighing homes in Shela Ahmedabad, the real question is whether daily life "
                           "works for your family. Here is what is on record, what is still being built, and "
                           "what to check before you book."),
        "key_takeaways": [] if not clean else [
            "Shela suits young families who want schools close by.",
            "Ahmedabad Metro Phase 2 is under construction, so plan around roads for now.",
            "Verify registration on GujRERA before paying anything."],
        "sections": MESSY_SECTIONS if not clean else CLEAN_SECTIONS,
        "faqs": [] if not clean else [{"question": "Is Shela good for young families?",
                                       "answer_markdown": "It can be, if you value schools close by and can wait for the metro."}],
        "conclusion_heading": "Conclusion" if not clean else "The verdict",
        "conclusion_markdown": ("In conclusion, Shela is great." if not clean else
                                "Shela suits young families who want schools close by and can wait for the metro. "
                                "If you need metro access today, look elsewhere."),
        "cta_heading": "Contact us",
        "cta_markdown": ("Call us at +91 XXXXX XXXXX or visit www.yourwebsite.com." if not clean else
                         "Book a weekday site visit and bring the checklist above."),
        "cover_image_id": None,
    }


class ScriptedModel:
    """call_with_fallback stand-in. Records calls per role; the writer
    returns the messy draft first and the clean one on revision."""

    def __init__(self, critic_scores: dict | None = None, writer_fails: bool = False):
        self.calls: dict[str, int] = {}
        self.prompts: dict[str, list[str]] = {}
        self.critic_scores = critic_scores
        self.writer_fails = writer_fails

    def __call__(self, role, fn_name, *args, **kwargs):
        role_val = role.value if hasattr(role, "value") else role
        self.calls[role_val] = self.calls.get(role_val, 0) + 1
        self.prompts.setdefault(role_val, []).append(args[0] if args else "")
        if role_val == "blog_research":
            return {"subject": "Shela", "place": "", "country_code": "in", "searches": [], "news_query": ""}
        if role_val == "blog_planner":
            return {"archetype": "locality_guide", "working_title": "Is Shela a good place to buy?",
                    "sections": []}          # too few -> merged with the deterministic plan
        if role_val == "visual_director":
            return {"subject": "a quiet residential street at golden hour", "caption": "Evening in the west of the city"}
        if role_val == "blog_writer":
            if self.writer_fails:
                from src.core.exceptions import ModelProviderError
                raise ModelProviderError("quota exhausted (429)")
            return writer_payload(clean=self.calls[role_val] > 1)
        if role_val == "blog_fact_check":
            return {"supported": []}
        if role_val == "blog_seo":
            return {"meta_description": "Short.", "seo_keywords": ["shela homes"], "slug": "Not A Slug"}
        if role_val == "blog_critic":
            scores = self.critic_scores or {}
            return {"dimensions": scores, "notes": {"structure_readability": "Split the intro."},
                    "reasons": {d: f"Scripted reason for {d}." for d in scores},
                    "grounding": 70, "writing_quality": 70}
        raise AssertionError(f"Unexpected call {role_val} {fn_name}")


SHELA_BRIEF = {
    "blog_id": "BLOG-SHELA",
    "topic": "Is Shela, Ahmedabad a good place to buy a home?",
    "audience": "young families",
    "sections": ["strengths", "weaknesses", "location_analysis", "buyer_considerations"],
    "local_facts": ["Shela is about 4 km from SP Ring Road", "2BHK apartments are listed at ₹55–75 lakh",
                    "Several CBSE schools within 3 km", "Ahmedabad Metro Phase 2 under construction",
                    "Shela Lake Garden is a 10-minute walk from most new projects"],
    "tone": "friendly",
    "target_word_count": 1000,
    "brand_name": "Skyline Homes",
    "cta_text": "Book a weekday site visit",
    "cta_url": "https://...",
    "cta_phone": "+91 XXXXX XXXXX",
}


def write_brief(directory, brief: dict = SHELA_BRIEF):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "blog_input.json").write_text(json.dumps(brief), encoding="utf-8")
    return directory


AGENT_MODULES = ["src.agents.blog.research", "src.agents.blog.planner", "src.agents.blog.visual_director",
                 "src.agents.blog.writer", "src.agents.blog.fact_check", "src.agents.blog.seo",
                 "src.agents.blog.critic"]


def patch_models(monkeypatch, fake) -> None:
    for module in AGENT_MODULES:
        monkeypatch.setattr(f"{module}.call_with_fallback", fake)
    monkeypatch.setattr("src.agents.base.time.sleep", lambda *_a, **_k: None)
