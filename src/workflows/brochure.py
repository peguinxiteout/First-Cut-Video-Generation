"""The brochure workflow — one LangGraph StateGraph.

    property_intelligence -> vision -> asset_prep -> visual_enrichment
        -> creative_director -> art_director -> copywriter -> render_qa -> critic
                                                    ^                      |
                                                    +---- revise ----------+

    property_intelligence  the fact ledger: every fact the brochure may state,
                           from the form (and any supplied documents)
    vision                 what each upload shows, how good it is, and any
                           text printed inside it (a plan's room names)
    asset_prep             transparency flattened, blank canvas trimmed off
                           drawings, the category's grade on photographs
    visual_enrichment      optional supporting AI visuals (lifestyle,
                           neighbourhood) — never the property, never the
                           cover, never labelled "artistic impression"
    creative_director      the palette, from the category's contrast-checked set
    art_director           the whole book planned from the material BEFORE a
                           word is written: which pages, which layout, which
                           photograph in which slot (planner.py, layouts.py)
    copywriter             every page written from the project's facts in the
                           category's voice (copywriter.py, voice.py, facts.py)
    render_qa              rendered in Chromium, measured, repaired by whole
                           sentences, and rendered again (qa.py, production.py)
    critic                 the evaluation: a model's judgement when one is
                           available, always blended with the measurements

Every step that calls a model has a deterministic fallback, so a brochure
is produced with no API key at all; the record says which steps fell back.

Call `run_brochure_workflow(property_dir, template_name)` — the API, the CLI
and the Streamlit app all use it.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.brochure import (
    catalog, copywriter, creative_director, critic, palettes, planner, production,
    property_intelligence, strategy, vision, vision_offline, visual_enrichment,
)
from src.core.config import settings
from src.core.state import BrochureState
from src.core.types import QualityGateResult
from src.data import repository
from src.data.documents import load_documents
from src.data.ingestion import discover_images, load_property_input
from src.data.schemas import GenerationRecord, WorkflowEvaluation
from src.evaluation import trust_checks
from src.models.image_gen import get_image_service
from src.observability import tracing
from src.workflows._instrumentation import instrumented, make_step_log_var

GENERATED_DIR = settings.data_dir_resolved / "generated"
logger = logging.getLogger(__name__)

_STEP_NAMES = [
    "property_intelligence", "vision", "asset_prep", "visual_enrichment",
    "creative_director", "art_director", "copywriter", "render_qa", "critic",
]
_step_log_var = make_step_log_var("brochure_step_log")
_instrumented = lambda step_name: instrumented(step_name, _step_log_var)  # noqa: E731


def _report(state: BrochureState, agent: str, *lines: str) -> dict:
    """Append what an agent decided to the run's agent reports."""
    reports = dict(state.get("agent_reports") or {})
    reports[agent] = list(reports.get(agent, [])) + [line for line in lines if line]
    return {"agent_reports": reports}


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

@_instrumented("property_intelligence")
def node_property_intelligence(state: BrochureState) -> dict:
    knowledge = property_intelligence.run(state["property_input"], state.get("documents_text", ""))
    out = {"property_knowledge": knowledge}
    out.update(_report(state, "property_intelligence",
                       f"{len(knowledge.facts)} facts in the ledger",
                       f"{len(knowledge.selling_points)} selling points"))
    return out


@_instrumented("vision")
def node_vision(state: BrochureState) -> dict:
    images = state["images"]
    try:
        analyses = vision.run(images)
        mode = "model"
    except Exception as exc:  # noqa: BLE001 - the planner can work from the uploads themselves
        logger.info("Vision unavailable (%s); reading the uploads offline", type(exc).__name__)
        analyses = {}
        mode = "offline"
    missing = [i for i in images if i.image_id not in analyses]
    if missing:
        analyses.update(vision_offline.run(missing))
    out = {"image_analyses": analyses}
    out.update(_report(state, "vision", f"{len(images)} images read ({mode})",
                       f"{len(missing)} read from the upload alone" if missing and mode == "model" else ""))
    return out


@_instrumented("asset_prep")
def node_asset_prep(state: BrochureState) -> dict:
    out_dir = GENERATED_DIR / state["generation_id"] / "normalised"
    images, notes = production.prepare_assets(state["images"], state.get("image_analyses") or {},
                                              out_dir, state["template_name"])
    out = {"images": images, "asset_notes": notes}
    out.update(_report(state, "asset_prep", f"{len(images)} images prepared", *notes[:6]))
    return out


@_instrumented("visual_enrichment")
def node_visual_enrichment(state: BrochureState) -> dict:
    """Optional supporting visuals. Soft by contract: no visual is worth a
    failed brochure. Offline placeholder images are never placed in a
    brochure."""
    try:
        images, analyses, report = visual_enrichment.run(
            state["property_input"], state.get("property_knowledge"), state["images"],
            state.get("image_analyses") or {}, get_image_service(),
            GENERATED_DIR / state["generation_id"] / "ai_images",
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("Visual enrichment skipped: %s", exc)
        return _report(state, "visual_enrichment", f"skipped ({type(exc).__name__})")
    if report.is_mock or not report.visuals:
        return _report(state, "visual_enrichment",
                       "no supporting visuals added" + (" (image generation offline)" if report.is_mock else ""))
    out = {"images": images, "image_analyses": analyses,
           "ai_image_paths": [v.path for v in report.visuals]}
    out.update(_report(state, "visual_enrichment", f"{len(report.visuals)} supporting visual(s) generated"))
    return out


@_instrumented("creative_director")
def node_creative_director(state: BrochureState) -> dict:
    spec = catalog.resolve(state["template_name"])
    palette = creative_director.choose(state.get("property_knowledge"), spec.category)
    out = {"palette_name": palette.name}
    out.update(_report(state, "creative_director", f"palette: {palette.name} — {palette.mood}"))
    return out


@_instrumented("art_director")
def node_art_director(state: BrochureState) -> dict:
    spec = catalog.resolve(state["template_name"])
    positioning = strategy.for_category(spec.category)
    book = planner.plan(state["property_input"], state["images"], state.get("image_analyses") or {},
                        spec, positioning)
    book.palette = state.get("palette_name", "")
    archetypes = {}
    for page in book.pages:
        archetypes[page.kind] = archetypes.get(page.kind, 0) + 1
    out = {"book": book}
    out.update(_report(state, "art_director",
                       f"{spec.name}: {len(book.pages)} pages planned",
                       ", ".join(f"{n} {k}" for k, n in archetypes.items()), *book.notes))
    return out


@_instrumented("copywriter")
def node_copywriter(state: BrochureState) -> dict:
    book = state["book"]
    spec = catalog.resolve(state["template_name"])
    positioning = strategy.for_category(spec.category)
    copywriter.write(book, state["property_input"], positioning,
                     seed=state.get("revision_count", 0))
    sources = {}
    for page in book.pages:
        sources[page.words.source or "none"] = sources.get(page.words.source or "none", 0) + 1
    out = {"book": book}
    out.update(_report(state, "copywriter", f"voice: {positioning.label} — {positioning.promise}",
                       "copy written by: " + ", ".join(f"{k} ({n} pages)" for k, n in sources.items())))
    return out


@_instrumented("render_qa")
def node_render_qa(state: BrochureState) -> dict:
    spec = catalog.resolve(state["template_name"])
    palette = palettes.with_brand(palettes.for_category(spec.category, state.get("palette_name")),
                                  getattr(state["property_input"], "brand_color", None))
    out_dir = GENERATED_DIR / state["generation_id"]
    result = production.produce(state["property_input"], state["images"],
                                state.get("image_analyses") or {}, spec, palette, out_dir,
                                book=state["book"])
    rep = result.report
    out = {
        "book": result.book,
        "html_path": str(result.html_path),
        "pdf_path": str(result.pdf_path),
        "screenshot_path": str(result.screenshot_path),
        "defects": rep.defects,
        "qa_report": rep,
    }
    for line in result.repairs:
        logger.info("render_qa %s: %s", state["generation_id"], line)
    out.update(_report(state, "render_qa",
                       f"{len(result.book.pages)} pages rendered and measured",
                       f"refined over {result.rounds} round(s)" if result.rounds else "",
                       f"average open space {rep.mean_whitespace():.0%}",
                       f"{len(rep.defects)} defect(s) after measuring" if rep.defects else "no defects measured"))
    return out


@_instrumented("critic")
def node_critic(state: BrochureState) -> dict:
    evaluation = critic.run_book(
        pages_dir=Path(state["screenshot_path"]).parent / "pages",
        knowledge=state["property_knowledge"], book=state["book"], report=state["qa_report"])
    tracing.log_final_scores(evaluation, step=state.get("revision_count", 0))
    out = {"evaluation": evaluation}
    out.update(_report(state, "critic", f"judged by: {evaluation.judge}",
                       f"quality score {evaluation.quality_score:.0f} ({evaluation.gate_result.value})"))
    return out


@_instrumented("revise")
def node_prepare_revision(state: BrochureState) -> dict:
    return {"revision_count": state.get("revision_count", 0) + 1}


def route_after_critic(state: BrochureState) -> str:
    evaluation = state["evaluation"]
    if (evaluation.gate_result == QualityGateResult.REVISE and evaluation.judge == "model"
            and state.get("revision_count", 0) < settings_max_revisions()):
        return "revise"
    return "done"


def settings_max_revisions() -> int:
    from src.core.config import load_yaml_config

    return int(load_yaml_config("app.yaml").get("quality_gate", {}).get("max_revisions", 1))


def build_graph():
    graph = StateGraph(BrochureState)
    for name, fn in (("property_intelligence", node_property_intelligence), ("vision", node_vision),
                     ("asset_prep", node_asset_prep), ("visual_enrichment", node_visual_enrichment),
                     ("creative_director", node_creative_director), ("art_director", node_art_director),
                     ("copywriter", node_copywriter), ("render_qa", node_render_qa),
                     ("critic", node_critic), ("prepare_revision", node_prepare_revision)):
        graph.add_node(name, fn)
    graph.set_entry_point("property_intelligence")
    for a, b in (("property_intelligence", "vision"), ("vision", "asset_prep"),
                 ("asset_prep", "visual_enrichment"), ("visual_enrichment", "creative_director"),
                 ("creative_director", "art_director"), ("art_director", "copywriter"),
                 ("copywriter", "render_qa"), ("render_qa", "critic"),
                 ("prepare_revision", "copywriter")):
        graph.add_edge(a, b)
    graph.add_conditional_edges("critic", route_after_critic, {"revise": "prepare_revision", "done": END})
    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH  # noqa: PLW0603
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_brochure_workflow(property_dir: Path, template_name: str = catalog.DEFAULT_TEMPLATE) -> GenerationRecord:
    """Generate the brochure for the property stored at `property_dir` in one
    of the six templates (an id such as "luxury_portrait", a display name
    such as "Luxury - Portrait", or an old id such as "luxury_modern")."""
    repository.init_db()
    property_input = load_property_input(property_dir)
    spec = catalog.resolve(template_name, property_input.category)
    images = discover_images(property_dir)
    documents_text, document_notes = load_documents(property_dir)
    generation_id = f"gen_{uuid.uuid4().hex[:10]}"
    record = GenerationRecord(generation_id=generation_id, property_id=property_input.property_id,
                              status="running", template=spec.name)
    repository.save_generation(record)

    initial: BrochureState = {
        "generation_id": generation_id, "property_input": property_input,
        "template_name": spec.id, "images": images, "documents_text": documents_text,
        "revision_count": 0, "agent_reports": {},
    }
    step_log: dict = {}
    token = _step_log_var.set(step_log)
    final: dict = {}
    try:
        try:
            with tracing.trace_generation(generation_id, property_input.property_id):
                final = get_graph().invoke(initial)
            record.status = "completed"
            record.html_path = final.get("html_path")
            record.pdf_path = final.get("pdf_path")
            record.screenshot_path = final.get("screenshot_path")
            record.evaluation = final.get("evaluation")
            record.defects = final.get("defects") or []
            book = final.get("book")
            record.page_count = len(book.pages) if book else 0
            record.notes = (trust_checks.check(property_input, final.get("property_knowledge"))
                            + document_notes + list(book.notes if book else [])
                            + list(final.get("asset_notes") or []))
            record.ai_image_paths = final.get("ai_image_paths") or []
            record.agent_reports = final.get("agent_reports") or {}
        except Exception as exc:  # noqa: BLE001 - record the failure; the UI shows a plain message
            logger.exception("Brochure generation failed")
            record.status = "failed"
            record.error = str(exc)
    finally:
        _step_log_var.reset(token)
    record.workflow_evaluation = WorkflowEvaluation(
        steps=[step_log[name] for name in _STEP_NAMES if name in step_log])
    repository.save_generation(record)
    return record
