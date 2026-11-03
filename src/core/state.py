"""
The shared state object LangGraph passes between brochure workflow nodes.
Each node reads what it needs and returns a dict of the keys it updates —
LangGraph merges that into the running state.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.agents.brochure.book import Book

from src.data.schemas import (
    BlogContent,
    BlogEvaluationResult,
    BlogImageAsset,
    BlogInput,
    BlogOutline,
    BlogRevisionRound,
    BlogValidationReport,
    EvaluationResult,
    FactCheckResult,
    ImageAnalysis,
    ImageAsset,
    PropertyInput,
    ResearchPack,
    PropertyKnowledge,
)


class BrochureState(TypedDict, total=False):
    """Every key a brochure node returns MUST be declared here: LangGraph
    drops undeclared keys from the state it returns, silently — which is
    how a gate's findings once went missing between the gate and the UI."""
    # Input
    generation_id: str
    property_input: PropertyInput
    images: list[ImageAsset]
    template_name: str              # a catalog id, e.g. "luxury_portrait"
    documents_text: str

    # Agents
    property_knowledge: PropertyKnowledge
    image_analyses: dict[str, ImageAnalysis]
    asset_notes: list[str]
    ai_image_paths: list[str]
    palette_name: str
    book: Book
    agent_reports: dict[str, list[str]]

    # Rendering and QA
    html_path: str
    pdf_path: str
    screenshot_path: str
    defects: list[str]
    qa_report: Any

    # Evaluation
    evaluation: EvaluationResult
    revision_count: int
    error: Optional[str]


class BlogState(TypedDict, total=False):
    generation_id: str
    blog_input: BlogInput
    images: list[BlogImageAsset]
    template_name: str

    # What the Research Agent found before anything was planned or
    # written. The Writer's only source of outside fact.
    #
    # Named ..._pack rather than plainly `research` because LangGraph
    # refuses a node whose name collides with a state key, and the NODE
    # wants the readable name — it is what appears in the workflow
    # evaluation the person sees.
    research_pack: ResearchPack

    outline: BlogOutline
    content: BlogContent

    # Which claims in the current draft the evidence supports. Feeds the
    # critic's issues, and names the sentences to fix in the next pass.
    # Suffixed for the same reason as research_pack above.
    fact_check_result: FactCheckResult

    # AI hero image (see node_visual). None if generation was unavailable.
    hero_image: Optional[BlogImageAsset]
    hero_caption: str
    image_notes: list[str]

    markdown_path: str
    html_path: str
    # The printed article. Declared here or LangGraph drops it from the
    # returned state, which is how the brochure's gate findings went
    # missing once already.
    pdf_path: str

    evaluation: BlogEvaluationResult
    revision_count: int
    # What was wrong last time, in words the Writer can act on. Empty on
    # the first pass.
    revision_note: str

    # Editor (markdown normaliser + validator): what it fixed, and what it
    # could not fix mechanically.
    validation_report: BlogValidationReport
    # "llm" or "fallback" — whether the current draft came from the model
    # or the deterministic composer. A fallback draft cannot act on
    # revision feedback, so the loop does not spin on it.
    writer_mode: str
    # One entry per critic pass: score, failing dimensions, feedback sent.
    revision_history: list[BlogRevisionRound]
    # The best-scoring draft so far. A revision that scores worse than the
    # draft it replaced is not what gets published.
    best_content: BlogContent
    best_evaluation: BlogEvaluationResult
    best_validation: BlogValidationReport
    # What the rendered PDF audit found (overflow, leaked markup, pages).
    render_report: list[str]
    pdf_pages: int
