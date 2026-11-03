"""
The blog generation workflow — LangGraph StateGraph:

    research -> planner -> visual -> writer -> fact_check -> editor -> seo
             -> critic -> [route] -> render
                  ^           |
                  +- revise --+   (bounded; dimension-level feedback)

WHY IT IS NOT JUST "BRIEF -> LLM -> POST"

A language model asked to write about a place produces confident
paragraphs about schools, metro lines and travel times, and has no idea
whether any of them are there. So the pipeline separates the jobs:

    research     goes and looks (maps, routing, encyclopaedia, news,
                 official sources), at a depth chosen by the article's
                 archetype. Writes it down as a ResearchPack.

    planner      picks the ARCHETYPE (locality guide, buyer's guide,
                 comparison, investment analysis, explainer, how-to,
                 market update) and a section plan in which every section
                 has a job and a FORMAT (table, checklist, steps...).

    visual       the hero image, before the writer, so the revision loop
                 never regenerates it.

    writer       has NO tools. The model writes from the plan; if no model
                 answers, the deterministic composer writes the same plan
                 from the same facts.

    fact_check   pulls every checkable claim out of the draft and matches
                 it against the brief and the research.

    editor       normaliser + validator: fixes formatting mechanically
                 (collapsed bullets, stray markdown, tables, bold, walls of
                 text, placeholder links), REMOVES untraceable figures and
                 repeated sentences, and reports what only a writer can fix.

    seo          meta title/description, slug, keywords, tags — validated
                 and rebuilt deterministically when needed.

    critic       the five-parameter rubric (deterministic measurements,
                 blended with the LLM critic when reachable) and the gate.

    route        below threshold, and a model is available to act on it,
                 and rounds remain -> back to the writer WITH the
                 dimension-level feedback. The best-scoring draft is kept:
                 a revision that scores worse does not replace it.

    render       the final, best draft only: markdown, HTML preview and the
                 PDF — the primary deliverable. No PDF means the run failed.

Call `run_blog_workflow(blog_dir)` — mirrors run_brochure_workflow's shape.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.blog import (
    critic, editor, fact_check, llm, planner, research as research_agent, seo, visual_director, writer,
)
from src.agents.image import generator as image_generator
from src.agents.image.prompting import StyleProfile
from src.models.image_gen import get_image_service
from src.core.config import settings
from src.core.exceptions import RenderingError
from src.core.state import BlogState
from src.core.types import QualityGateResult
from src.data import blog_repository as repository
from src.data.blog_ingestion import discover_blog_images, load_blog_input
from src.data.schemas import (
    AgentStepResult, BlogGenerationRecord, BlogImageAsset, BlogRevisionRound, ResearchPack, WorkflowEvaluation,
)
from src.evaluation import blog_scoring
from src.tools import toolbox_for
from src.observability import tracing
from src.rendering.blog_renderer import render_blog_files, render_blog_pdf
from src.workflows._instrumentation import instrumented, make_step_log_var

logger = logging.getLogger(__name__)
GENERATED_DIR = settings.data_dir_resolved / "generated_blogs"

_STEP_NAMES = ["research", "planner", "visual", "writer", "fact_check", "editor", "seo", "critic", "render"]

HERO_IMAGE_ID = "AI_HERO"

_step_log_var = make_step_log_var("blog_step_log")
_instrumented = lambda step_name: instrumented(step_name, _step_log_var)  # noqa: E731


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@_instrumented("research")
def node_research(state: BlogState) -> dict:
    """Find out about the place before writing about it. Soft-failing in
    every direction: the pack comes back with notes saying what failed."""
    pack = research_agent.run(state["blog_input"], toolbox_for("research"))
    return {"research_pack": pack}


@_instrumented("planner")
def node_planner(state: BlogState) -> dict:
    outline = planner.run(state["blog_input"], state.get("research_pack"))
    return {"outline": outline}


@_instrumented("visual")
def node_visual(state: BlogState) -> dict:
    """Generate the hero image. Soft-failing: if generation is unavailable
    the post is still written and rendered, just without a hero."""
    blog_input = state["blog_input"]
    brief = visual_director.run(blog_input, state["outline"].working_title)
    out_dir = GENERATED_DIR / state["generation_id"]
    report = image_generator.run([brief], StyleProfile(), get_image_service(), out_dir)
    if not report.visuals:
        return {"hero_image": None, "hero_caption": "", "image_notes": report.skipped}
    if report.is_mock:
        # The offline placeholder is a flat gradient, not a picture: it is
        # never printed in a post (nor in a brochure). The post goes out
        # without a hero until an image provider is configured.
        logger.info("Image generation offline: the post is rendered without a hero image")
        return {"hero_image": None, "hero_caption": "",
                "image_notes": ["No hero image: image generation is not configured "
                                "(set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env)."]
                + report.skipped}
    visual = report.visuals[0]
    hero = BlogImageAsset(image_id=HERO_IMAGE_ID, filename=Path(visual.path).name, path=visual.path)
    return {"hero_image": hero, "hero_caption": brief.caption, "image_notes": report.skipped}


@_instrumented("writer")
def node_writer(state: BlogState) -> dict:
    content = writer.run(state["blog_input"], state["outline"], state["images"],
                         research=state.get("research_pack"),
                         revision_note=state.get("revision_note", ""))
    return {"content": content, "writer_mode": content.writer_mode}


@_instrumented("fact_check")
def node_fact_check(state: BlogState) -> dict:
    """Check the draft against the evidence. What fails goes to the Editor
    (which removes it) and to the next revision note (which names it)."""
    result = fact_check.run(state["blog_input"], state["content"], state.get("research_pack"))
    return {"fact_check_result": result}


@_instrumented("editor")
def node_editor(state: BlogState) -> dict:
    content, report = editor.run(state["blog_input"], state["content"], state.get("research_pack"),
                                 state.get("fact_check_result"))
    return {"content": content, "validation_report": report}


@_instrumented("seo")
def node_seo(state: BlogState) -> dict:
    content = seo.run(state["content"], state.get("research_pack"), toolbox_for("seo"),
                      blog_input=state["blog_input"])
    hero = state.get("hero_image")
    if hero is not None:
        # The hero always comes from the Visual Director, never the writer.
        content.cover_image_id = hero.image_id
        content.hero_is_ai_generated = True
        content.hero_caption = state.get("hero_caption", "")
    elif not content.cover_image_id and state.get("images"):
        # No generated hero: the author's own first unused photo leads.
        used = {s.image_id for s in content.sections if s.image_id}
        spare = [img for img in state["images"] if img.image_id not in used] or state["images"]
        content.cover_image_id = spare[0].image_id
    return {"content": content}


@_instrumented("critic")
def node_critic(state: BlogState) -> dict:
    evaluation = critic.run(state["blog_input"], state["content"], state.get("outline"),
                            state.get("fact_check_result"), research=state.get("research_pack"),
                            validation=state.get("validation_report"))
    round_no = state.get("revision_count", 0)
    tracing.log_final_scores(evaluation, step=round_no)

    out: dict = {"evaluation": evaluation}
    best = state.get("best_evaluation")
    if best is None or evaluation.quality_score >= best.quality_score + (
            blog_scoring.min_improvement() if round_no else 0):
        out.update(best_content=state["content"], best_evaluation=evaluation,
                   best_validation=state.get("validation_report"))
    decision = _decide(evaluation, state)
    history = list(state.get("revision_history", []) or [])
    history.append(BlogRevisionRound(
        round=round_no, quality_score=evaluation.quality_score, gate_result=evaluation.gate_result.value,
        failing_dimensions=[d.name for d in evaluation.dimensions if not d.passed],
        feedback=evaluation.feedback if decision == "revise" else [],
        writer_mode=state.get("writer_mode", ""), decision=decision))
    out["revision_history"] = history
    return out


def _decide(evaluation, state: BlogState) -> str:
    if evaluation.gate_result == QualityGateResult.PASS:
        return "done"
    if state.get("revision_count", 0) >= blog_scoring.max_revisions():
        return "done: revision limit reached"
    if state.get("writer_mode") != "llm" or llm.unavailable_reason():
        return "skipped: no model available to act on feedback (deterministic draft kept)"
    if not evaluation.feedback:
        return "done: nothing specific to fix"
    return "revise"


def node_prepare_revision(state: BlogState) -> dict:
    """The note the Writer gets: dimension-level feedback, the sentences
    the Editor had to remove, and what it could not fix."""
    evaluation = state["evaluation"]
    note = "\n".join(f"- {line}" if not line.startswith("-") else line for line in evaluation.feedback)
    return {"revision_count": state.get("revision_count", 0) + 1, "revision_note": note}


def route_after_critic(state: BlogState) -> str:
    history = state.get("revision_history") or []
    return "revise" if history and history[-1].decision == "revise" else "done"


@_instrumented("render")
def node_render(state: BlogState) -> dict:
    """Render the BEST draft. The PDF is the deliverable: if it cannot be
    produced the run fails, with the markdown and preview still on disk."""
    content = state.get("best_content") or state["content"]
    images_by_id = {img.image_id: img for img in state["images"]}
    hero = state.get("hero_image")
    if hero is not None:
        images_by_id[hero.image_id] = hero
    gen_dir = GENERATED_DIR / state["generation_id"]
    markdown_path, html_path = render_blog_files(content, images_by_id, gen_dir, blog_input=state["blog_input"])
    pdf_path = render_blog_pdf(gen_dir)
    if pdf_path is None:
        pdf_path = render_blog_pdf(gen_dir)      # one retry: a cold Chromium start can time out
    if pdf_path is None:
        raise RenderingError("The article PDF could not be produced (is Playwright Chromium installed? "
                             "run `playwright install chromium`).")
    from src.rendering.blog_renderer import audit_pdf

    headings = [s.heading for s in content.sections] + [content.conclusion_heading,
                                                         "Frequently asked questions", "Sources and where to verify"]
    headings += [f.question for f in content.faqs]
    audit = audit_pdf(pdf_path, headings)
    stranded = frozenset(h for h in (s.heading for s in content.sections)
                         if any(f"“{h}”" in note for note in audit["notes"]))
    if stranded:
        # One mechanical repair: start each stranded section on a new page.
        render_blog_files(content, images_by_id, gen_dir, blog_input=state["blog_input"], break_before=stranded)
        if render_blog_pdf(gen_dir) is not None:
            audit = audit_pdf(pdf_path, headings)
            audit["notes"].append("Re-rendered with a page break before: " + "; ".join(sorted(stranded)) + ".")
    return {"markdown_path": str(markdown_path), "html_path": str(html_path), "pdf_path": str(pdf_path),
            "content": content, "evaluation": state.get("best_evaluation") or state.get("evaluation"),
            "validation_report": state.get("best_validation") or state.get("validation_report"),
            "render_report": audit["notes"], "pdf_pages": audit["pages"]}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("research", node_research)
    graph.add_node("planner", node_planner)
    graph.add_node("visual", node_visual)
    graph.add_node("writer", node_writer)
    graph.add_node("fact_check", node_fact_check)
    graph.add_node("editor", node_editor)
    graph.add_node("seo", node_seo)
    graph.add_node("critic", node_critic)
    graph.add_node("prepare_revision", node_prepare_revision)
    graph.add_node("render", node_render)

    graph.set_entry_point("research")
    graph.add_edge("research", "planner")
    graph.add_edge("planner", "visual")
    graph.add_edge("visual", "writer")
    graph.add_edge("writer", "fact_check")
    graph.add_edge("fact_check", "editor")
    graph.add_edge("editor", "seo")
    graph.add_edge("seo", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"revise": "prepare_revision", "done": "render"}
    )
    graph.add_edge("prepare_revision", "writer")
    graph.add_edge("render", END)

    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH  # noqa: PLW0603
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_blog_workflow(blog_dir: Path) -> BlogGenerationRecord:
    """Runs the full blog pipeline for the brief stored at `blog_dir` (e.g.
    data/blogs/BLOG-001) and returns the persisted BlogGenerationRecord."""
    repository.init_db()

    blog_input = load_blog_input(blog_dir)
    images = discover_blog_images(blog_dir)
    generation_id = f"blog_{uuid.uuid4().hex[:10]}"

    record = BlogGenerationRecord(
        generation_id=generation_id, blog_id=blog_input.blog_id, status="running"
    )
    repository.save_generation(record)

    initial_state: BlogState = {
        "generation_id": generation_id,
        "blog_input": blog_input,
        "images": images,
        "revision_count": 0,
        "revision_note": "",
        "revision_history": [],
        "research_pack": ResearchPack(),
    }

    step_log: dict[str, AgentStepResult] = {}
    token = _step_log_var.set(step_log)
    llm_token = llm.reset()
    final_state: dict = {}

    try:
        try:
            with tracing.trace_generation(generation_id, blog_input.blog_id):
                final_state = get_graph().invoke(initial_state, {"recursion_limit": 60})

            record.status = "completed"
            _fill_record(record, final_state)
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error = str(exc)
            # Whatever was written before the failure is still useful.
            gen_dir = GENERATED_DIR / generation_id
            for attr, name in (("markdown_path", "content.md"), ("html_path", "preview.html")):
                if (gen_dir / name).exists():
                    setattr(record, attr, str(gen_dir / name))
    finally:
        llm.restore(llm_token)
        _step_log_var.reset(token)

    record.workflow_evaluation = WorkflowEvaluation(
        steps=[step_log[name] for name in _STEP_NAMES if name in step_log]
    )

    repository.save_generation(record)
    return record


def _fill_record(record: BlogGenerationRecord, final_state: dict) -> None:
    from src.agents.blog.article import full_markdown, plain

    record.markdown_path = final_state.get("markdown_path")
    record.html_path = final_state.get("html_path")
    record.pdf_path = final_state.get("pdf_path")
    record.evaluation = final_state.get("evaluation")
    hero = final_state.get("hero_image")
    record.hero_image_path = hero.path if hero else None
    record.image_notes = final_state.get("image_notes", []) or []
    # What research could not find belongs with the post, not in a log.
    pack = final_state.get("research_pack")
    record.research_notes = list(pack.notes) if pack else []
    record.sources = list(dict.fromkeys(pack.sources)) if pack else []
    content = final_state.get("content")
    outline = final_state.get("outline")
    if content is not None:
        record.title = content.title
        record.slug = content.slug
        record.meta_title = content.meta_title
        record.meta_description = content.meta_description
        record.archetype = content.archetype
        record.writer_mode = content.writer_mode
        record.word_count = len(plain(full_markdown(content)).split())
    record.planner_mode = outline.planner_mode if outline is not None else ""
    record.revision_history = list(final_state.get("revision_history") or [])
    record.validation = final_state.get("validation_report")
    record.render_report = list(final_state.get("render_report") or [])
    record.pdf_pages = int(final_state.get("pdf_pages") or 0)
