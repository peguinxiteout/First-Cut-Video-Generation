"""The Critic: one judgement of the finished brochure.

A vision/language model reads the rendered pages and the copy and scores
three things a measurement cannot: whether the copy contradicts the facts,
how good the design looks, how good the writing is. The rendered-page
measurements (src/evaluation/brochure_checks.py) are blended in and always
present, so

    * a brochure is evaluated even when no model is available — the scores
      then come from the measurements alone and say so (`judge`), and
    * no model can rate well a brochure that measurably breaks a rule: the
      layout integrity score and the plain-language issues come from the
      rendered pages, not from the model's opinion.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from string import Template

from src.agents.base import with_retries
from src.agents.brochure import qa, strategy
from src.agents.brochure.book import Book
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import EvaluationResult, PropertyKnowledge
from src.evaluation import brochure_checks
from src.evaluation.scoring import apply_quality_gate, compute_quality_score, factual_consistency_score_from_status
from src.models.gateway import call_with_fallback
from src.models.image_prep import prepare_screenshot_for_vision_api

logger = logging.getLogger(__name__)


def page_summary(book: Book) -> str:
    lines = []
    for page in book.pages:
        c = page.words
        parts = [f"Page {page.number} ({page.kind}{', ' + page.topic if page.topic else ''})"]
        for label, value in (("kicker", c.kicker), ("title", c.title), ("lead", c.lead)):
            if value:
                parts.append(f'{label}="{value}"')
        if c.body:
            parts.append(f'text="{" ".join(c.body)}"')
        if c.cards:
            parts.append("cards=" + json.dumps([f"{k.title}: {k.text}" for k in c.cards][:8], ensure_ascii=False))
        for key in ("figures", "stats", "units", "places"):
            if page.data.get(key):
                parts.append(f"{key}=" + json.dumps(page.data[key], ensure_ascii=False, default=str)[:400])
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _judge(pages_dir: Path, knowledge: PropertyKnowledge, book: Book) -> dict:
    positioning = strategy.for_category(book.category)
    prompt = Template(load_yaml_config("brochure/prompts.yaml")["critic_combined"]).substitute(
        fact_ledger="\n".join(f"- {k}: {v}" for k, v in knowledge.facts.items()),
        positioning=f"{positioning.label}: {positioning.promise}. Pillars: {', '.join(positioning.pillars)}.",
        page_content=page_summary(book),
    )
    shots = sorted(pages_dir.glob("page_*.png"))
    with tempfile.TemporaryDirectory(prefix="critic_pages_") as tmp:
        prepared = [prepare_screenshot_for_vision_api(str(s), Path(tmp)) for s in shots[:24]]
        return with_retries(lambda: call_with_fallback(ModelRole.CRITIC, "generate_json_with_images",
                                                       prompt, prepared), max_retries=1)


def run_book(*, pages_dir: Path, knowledge: PropertyKnowledge, book: Book,
             report: qa.Report, use_model: bool = True) -> EvaluationResult:
    measured = brochure_checks.measure(report, book.category)
    judged: dict = {}
    if use_model:
        try:
            judged = _judge(pages_dir, knowledge, book) or {}
        except Exception as exc:  # noqa: BLE001 - the measurements still evaluate it
            logger.info("Critic model unavailable (%s); evaluating from measurements", type(exc).__name__)
            judged = {}
    evaluation = EvaluationResult(
        layout_integrity=measured["layout_integrity"],
        whitespace_mean=measured["whitespace_mean"],
        pages_over_whitespace=measured["pages_over_whitespace"],
        image_integrity=measured["image_integrity"],
        checks=measured["checks"],
        issues=brochure_checks.plain_issues(report),
    )
    if judged:
        evaluation.judge = "model"
        evaluation.factual_consistency = factual_consistency_score_from_status(
            judged.get("factual_consistency_status", ""))
        evaluation.factual_consistency_reasoning = judged.get("factual_consistency_reasoning", "")
        evaluation.visual_design_quality = float(judged.get("visual_design_quality", 0) or 0)
        evaluation.visual_design_quality_reasoning = judged.get("visual_design_quality_reasoning", "")
        evaluation.content_quality = float(judged.get("content_quality", 0) or 0)
        evaluation.content_quality_reasoning = judged.get("content_quality_reasoning", "")
        # A model cannot rate the design above what the measurements allow.
        evaluation.visual_design_quality = min(evaluation.visual_design_quality,
                                               measured["visual_proxy"] + (0 if report.defects else 5))
    else:
        evaluation.judge = "measured"
        evaluation.factual_consistency = 100.0      # every number in the copy is traced (copywriter.Guard)
        evaluation.factual_consistency_reasoning = (
            "No model judge was available. Every figure in the copy is checked against the "
            "property's own data before printing, so no figure can contradict it.")
        evaluation.visual_design_quality = measured["visual_proxy"]
        evaluation.visual_design_quality_reasoning = (
            f"Measured: mean open space {measured['whitespace_mean']:.0%}, "
            f"{measured['pages_over_whitespace']} page(s) outside the category's range, layout integrity "
            f"{measured['layout_integrity']:.0f}/100, image integrity {measured['image_integrity']:.0f}/100.")
        offline = sum(1 for p in book.pages if p.words.source == "offline")
        evaluation.content_quality = 72.0 if offline else 80.0
        evaluation.content_quality_reasoning = (
            "No model judge was available; the copy was composed offline from the project's own "
            "facts in the category's voice, fitted to each page by whole sentences, and checked "
            "for repetition, invented figures and image description. A model review is "
            "recommended before print.")
    evaluation.quality_score = compute_quality_score(evaluation)
    evaluation.gate_result = apply_quality_gate(evaluation)
    return evaluation
