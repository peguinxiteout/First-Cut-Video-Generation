"""The critic: a model's judgement when one is available, always held to
what the rendered pages measured."""
from __future__ import annotations

from unittest.mock import patch

from src.agents.brochure import critic, qa
from src.agents.brochure.book import Book, BookPage, Copy
from src.data.schemas import PropertyKnowledge

KNOWLEDGE = PropertyKnowledge(facts={"name": "Test Estates", "city": "Ahmedabad"})


def _book():
    return Book(template="mid_landscape", template_name="Mid-Tier - Landscape", category="mid",
                orientation="landscape", width_mm=297, height_mm=210,
                pages=[BookPage(number=1, kind="cover", words=Copy(title="Test Estates")),
                       BookPage(number=2, kind="feature", words=Copy(title="Poolside, any day", source="offline"))])


def _report(defects=(), whitespace=(0.1, 0.32)):
    pages = [qa.PageReport(number=i + 1, kind=k, whitespace=w)
             for i, (k, w) in enumerate(zip(("cover", "feature"), whitespace))]
    return qa.Report(pages=pages, defects=list(defects), advisories=[])


def test_without_a_model_the_measurements_evaluate_it(tmp_path):
    evaluation = critic.run_book(pages_dir=tmp_path, knowledge=KNOWLEDGE, book=_book(),
                                 report=_report(), use_model=False)
    assert evaluation.judge == "measured"
    assert evaluation.checks["whitespace"] == "pass"
    assert evaluation.image_integrity == 100
    assert 0 < evaluation.quality_score <= 100


def test_a_defect_costs_layout_integrity(tmp_path):
    clean = critic.run_book(pages_dir=tmp_path, knowledge=KNOWLEDGE, book=_book(),
                            report=_report(), use_model=False)
    broken = critic.run_book(pages_dir=tmp_path, knowledge=KNOWLEDGE, book=_book(),
                             report=_report(["Page 2: copy runs past its box (text box, 4.0mm)",
                                             "Page 2: an image did not load (IMG1)"]),
                             use_model=False)
    assert broken.layout_integrity < clean.layout_integrity
    assert broken.image_integrity < 100
    assert broken.checks["boundaries"] != "pass" and broken.checks["image_integrity"] != "pass"
    assert broken.quality_score < clean.quality_score


def test_the_model_judgement_is_captured_and_capped_by_the_measurements(tmp_path):
    judged = {
        "factual_consistency_reasoning": "Name and city match; no contradictions found.",
        "factual_consistency_status": "consistent",
        "visual_design_quality_reasoning": "Balanced pages.",
        "visual_design_quality": 99,
        "content_quality_reasoning": "Specific, benefit-led copy.",
        "content_quality": 88,
    }
    with patch("src.agents.brochure.critic._judge", return_value=judged):
        evaluation = critic.run_book(pages_dir=tmp_path, knowledge=KNOWLEDGE, book=_book(),
                                     report=_report(["Page 2: copy runs past its box (text box, 4.0mm)"],
                                                    whitespace=(0.1, 0.85)))
    assert evaluation.judge == "model"
    assert evaluation.factual_consistency == 100
    assert "no contradictions found" in evaluation.factual_consistency_reasoning
    assert evaluation.content_quality == 88
    # a measured defect and an empty page: the model cannot rate the design 99
    assert evaluation.visual_design_quality < 99
    assert evaluation.checks["whitespace"] != "pass"


def test_sparse_fact_ledger_is_not_penalized_by_the_rubric_text_itself():
    from src.core.config import load_yaml_config

    prompt = load_yaml_config("brochure/prompts.yaml")["critic_combined"]
    normalized = " ".join(prompt.split())
    assert "95-100" in normalized
    assert "however few or many facts were available" in normalized
    assert "Scarcity of source information is out of scope" in normalized
    assert "artistic impression" not in normalized.lower()
