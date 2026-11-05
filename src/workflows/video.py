"""
The video generation workflow — LangGraph StateGraph, in two halves:

    PLAN      n_brief -> n_fact_ledger -> n_asset_audit -> n_script -> n_storyboard
    (approve) a person, or the CLI, sees the storyboard and may change it
    PRODUCE   n_voice_music -> n_motion -> n_overlays -> n_compose -> n_gate

Image first, then motion: every shot starts from one of the person's own
photographs and a video model (or pan-and-zoom) only adds camera movement.
Our code draws every word on screen. Every number said or shown must be a
fact of the property (agents/video/guard.py). Every agent has a path that
needs no model, so a video can be made with no key and no network.

Graph node ids carry an n_ prefix because LangGraph refuses a node named like
a state key (brief, script and storyboard are both); the names the Workflow
tab shows are the step names in _STEP_NAMES.

`run_plan_workflow()` runs the first half and saves a record with the status
"planned"; script.json and storyboard.json are written beside it, and
plan_state.json (data/video_plan_state.py) with what the second half needs that
the record does not hold. `run_produce_workflow(video_id)` runs the second half
from the saved record and that file, in another process if need be: it never
plans again. `run_video_workflow()` runs both halves and saves the gate's
verdict as the status: "passed", "warned" or "failed" (a failed gate still
keeps the MP4; qa.json beside it lists every check). While a run goes its
record says "planning" or "producing", and after every node the repository's
cheap update_progress saves the current node and the steps done, so another
process can poll. A soft failure (a clip that falls back
to pan-and-zoom, text drawn by the fallback renderer, a narration that does
not fit) is recorded and never fails the run; only "no MP4 at all" does.

A run reaches only what configs/video/policy.yaml allows (agents/video/
policy.py): the other vendors' keys are blanked for its length and a call that
would go to one is refused and written to the record's notes.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from src.agents.video import assets as asset_audit
from src.agents.video import brief as brief_agent
from src.agents.video import fact_ledger
from src.agents.video import gate as gate_agent
from src.agents.video import motion as motion_agent
from src.agents.video import overlays as overlays_agent
from src.agents.video import policy
from src.agents.video import script as script_agent
from src.agents.video import storyboard as storyboard_agent
from src.agents.video import voice as voice_agent
from src.core.exceptions import WorkflowError
from src.core.video_state import VideoState
from src.data import video_plan_state
from src.data import video_repository as repository
from src.data.schemas import AgentStepResult, WorkflowEvaluation
from src.data.video_schemas import (
    STATUS_PLANNED, STATUS_PLANNING, STATUS_PRODUCING, GateCheck, Storyboard, VideoEvaluationResult, VideoGate,
    VideoGenerationRecord, VideoMeasurements, VideoRequest,
)
from src.observability import tracing
from src.rendering import ffmpeg, video_compose
from src.workflows._instrumentation import instrumented, make_step_log_var

logger = logging.getLogger(__name__)

_PLAN_STEPS = ["brief", "fact_ledger", "asset_audit", "script", "storyboard"]
_PRODUCE_STEPS = ["voice_music", "motion", "overlays", "compose", "gate"]
_STEP_NAMES = _PLAN_STEPS + _PRODUCE_STEPS

_step_log_var = make_step_log_var("video_step_log")
# Called with ("start" | "done", step name) around every node by the run that set it.
_progress_var: contextvars.ContextVar[Optional[Callable[[str, str], None]]] = contextvars.ContextVar(
    "video_progress", default=None)


def _progress(phase: str, step_name: str) -> None:
    """Tells the run's progress saver a node starts or ends. A failure to save
    progress never fails the run."""
    saver = _progress_var.get()
    if saver is None:
        return
    try:
        saver(phase, step_name)
    except Exception:  # noqa: BLE001 - progress is a courtesy; the run goes on
        logger.warning("Could not save the progress of step %s", step_name, exc_info=True)


def _instrumented(step_name: str):
    """The shared step log and trace (_instrumentation.py), plus a progress save before and after the node."""
    record_step = instrumented(step_name, _step_log_var)

    def decorator(fn):
        node = record_step(fn)

        def wrapper(state):
            _progress("start", step_name)
            try:
                return node(state)
            finally:
                _progress("done", step_name)

        return wrapper

    return decorator


def _progress_saver(video_id: str, step_log: dict[str, AgentStepResult]) -> Callable[[str, str], None]:
    def save(phase: str, step_name: str) -> None:
        if phase == "start":
            repository.update_progress(video_id, current_node=step_name)
        else:
            repository.update_progress(video_id, current_node=step_name, steps=_steps(step_log))
    return save


def _steps(step_log: dict[str, AgentStepResult]) -> list[AgentStepResult]:
    return [step_log[name] for name in _STEP_NAMES if name in step_log]


def _report(state: VideoState, agent: str, *lines: str) -> dict:
    """Append what an agent decided to the run's agent reports."""
    reports = dict(state.get("agent_reports") or {})
    reports[agent] = list(reports.get(agent, [])) + [line for line in lines if line]
    return {"agent_reports": reports}


def _more(state: VideoState, key: str, items: list[str]) -> list[str]:
    return list(state.get(key) or []) + list(items)


def _out_dir(state: VideoState) -> Path:
    return repository.video_output_dir(state["video_id"])


# ---------------------------------------------------------------------------
# Nodes: plan
# ---------------------------------------------------------------------------

@_instrumented("brief")
def node_brief(state: VideoState) -> dict:
    loaded = brief_agent.run(Path(state["property_dir"]), state.get("video_overrides"),
                             draft=state.get("draft", False))
    b = loaded.brief
    out = {"property_input": loaded.property_input, "images": loaded.images,
           "documents_text": loaded.documents_text, "video_input": loaded.video_input,
           "brief": b, "brand_kit": loaded.brand_kit}
    out.update(_report(
        state, "brief",
        f"{b.duration_s} s, {b.aspect}, {b.width}x{b.height} ({b.quality}); {b.objective}, {b.tone} tone",
        f"{b.n_shots} shots, the closing card included; about {b.target_words} words of narration",
        f"{len(loaded.images)} uploaded pictures", *loaded.notes))
    return out


@_instrumented("fact_ledger")
def node_fact_ledger(state: VideoState) -> dict:
    knowledge, notes = fact_ledger.run(state["property_input"], state.get("documents_text", ""),
                                       state.get("offline", False), refresh=state.get("refresh_ledger", False))
    said = [n for n in notes if n.startswith((fact_ledger.CACHE_NOTE, fact_ledger.REFRESH_NOTE))]
    out = {"property_knowledge": knowledge, "notes": _more(state, "notes", said)}
    out.update(_report(state, "fact_ledger", f"{len(knowledge.facts)} facts, "
                       f"{len(knowledge.selling_points)} selling points", *notes))
    return out


@_instrumented("asset_audit")
def node_asset_audit(state: VideoState) -> dict:
    analyses, ranked, plan, notes = asset_audit.run(state["images"], state.get("offline", False),
                                                    state["brief"].n_shots - 1)
    out = {"image_analyses": analyses, "ranked_assets": ranked, "image_plan": plan, "asset_notes": notes}
    out.update(_report(state, "asset_audit", *notes))
    return out


@_instrumented("script")
def node_script(state: VideoState) -> dict:
    result = script_agent.run(state["property_input"], state["brief"], state["property_knowledge"],
                              state.get("ranked_assets"))
    s = result.script
    out = {"script": s, "concept": result.concept}
    out.update(_report(state, "script", f"{len(s.lines)} lines, {s.word_count} words "
                       f"(target {result.concept['target_words']})", *s.notes))
    return out


@_instrumented("storyboard")
def node_storyboard(state: VideoState) -> dict:
    board = storyboard_agent.run(state["brief"], state["script"], state["ranked_assets"],
                                 state["image_analyses"], state["image_plan"])
    out = {"storyboard": board}
    out.update(_report(state, "storyboard", f"{len(board.shots)} shots, {board.total_seconds} s "
                       f"({board.planner_mode} planner)", *board.notes))
    return out


# ---------------------------------------------------------------------------
# Nodes: produce
# ---------------------------------------------------------------------------

@_instrumented("voice_music")
def node_voice_music(state: VideoState) -> dict:
    result = voice_agent.run(state["script"], state["storyboard"], state["brief"], _out_dir(state))
    out = {"storyboard": result.storyboard, "line_timings": result.line_timings,
           "voice_tracks": result.voice_tracks, "voice_name": result.voice_name,
           "music_track": result.music_track, "music_info": result.music_info, "beat_times": result.beat_times,
           "defects": _more(state, "defects", result.defects), "notes": _more(state, "notes", result.notes)}
    music = result.music_info
    out.update(_report(state, "voice_music", f"voice: {result.voice_name}",
                       f"music: {music.provider}" + (f" ({music.model_id}, ${music.cost_usd:.2f})" if music.model_id
                                                     else ""),
                       f"{len(result.line_timings)} lines spoken; the video is {result.total_s:.1f} s",
                       *result.notes, *result.defects))
    return out


@_instrumented("motion")
def node_motion(state: VideoState) -> dict:
    result = motion_agent.run(state["storyboard"], state["brief"], state["brand_kit"], _out_dir(state))
    modes = Counter(c.provider if c.mode != "card" else "card" for c in result.clips)
    layouts = ", ".join(f"shot {c.shot_index + 1}: {c.layout} x{c.peak_upscale:.2f}" for c in result.clips if c.layout)
    out = {"storyboard": result.storyboard, "keyframes": result.keyframes, "clips": result.clips,
           "notes": _more(state, "notes", result.notes)}
    out.update(_report(state, "motion", "clips: " + ", ".join(f"{n} x {m}" for m, n in modes.items()),
                       f"layout and peak enlargement: {layouts}" if layouts else "", *result.notes))
    return out


@_instrumented("overlays")
def node_overlays(state: VideoState) -> dict:
    result = overlays_agent.run(state["storyboard"], state["script"], state["line_timings"], state["brief"],
                                state["brand_kit"], _out_dir(state))
    frames: dict[int, list[str]] = {}
    for item in result.items:
        frames.setdefault(item.shot_index, []).append(item.path)
    out = {"overlay_items": result.items, "overlay_frames": frames, "captions_path": result.captions_path,
           "defects": _more(state, "defects", result.defects), "notes": _more(state, "notes", result.notes)}
    renderers = sorted({i.renderer for i in result.items}) or ["none"]
    out.update(_report(state, "overlays", f"{len(result.items)} overlays ({', '.join(renderers)}); captions written",
                       *result.notes, *result.defects))
    return out


@_instrumented("compose")
def node_compose(state: VideoState) -> dict:
    brief, board, out_dir = state["brief"], state["storyboard"], _out_dir(state)
    frames = video_compose.shot_frames([s.seconds for s in board.shots], brief.fps)
    clip_of = {c.shot_index: c for c in state["clips"]}
    result = video_compose.compose(
        clips=[video_compose.ClipPart(Path(clip_of[s.index].path), n, grade=clip_of[s.index].mode != "card")
               for s, n in zip(board.shots, frames)],
        overlays=[video_compose.OverlayPart(Path(o.path), o.start_s, o.end_s, o.fade_s, o.slide_px)
                  for o in state.get("overlay_items", [])],
        voices=[video_compose.VoicePart(Path(t.path), t.start_s) for t in state["line_timings"] if t.mode == "speech"],
        out_path=out_dir / "video.mp4", thumbnail_path=out_dir / "thumbnail.jpg", width=brief.width,
        height=brief.height, fps=brief.fps, draft=state.get("draft", False), work_dir=out_dir / "work",
        hook_seconds=board.shots[0].seconds, grade=brief.grade,
        music_path=Path(state["music_track"]) if state.get("music_track") else None)
    info = ffmpeg.probe(result.video_path)
    loud = result.loudness
    measurements = VideoMeasurements(
        duration_s=info.duration_s, width=info.width, height=info.height, fps=info.fps, vcodec=info.vcodec,
        acodec=info.acodec, has_audio=info.has_audio, size_bytes=info.size_bytes,
        loudness_lufs=loud.integrated_lufs if loud else None, true_peak_dbfs=loud.true_peak_dbfs if loud else None)
    out = {"video_path": str(result.video_path), "thumbnail_path": str(result.thumbnail_path),
           "measurements": measurements, "notes": _more(state, "notes", result.notes)}
    out.update(_report(state, "compose", f"{info.width}x{info.height} {info.vcodec}/{info.acodec} at {info.fps:g} fps, "
                       f"{info.duration_s:.2f} s, {info.size_bytes / 1e6:.1f} MB",
                       f"loudness {loud.integrated_lufs:.1f} LUFS" if loud else "", *result.notes))
    return out


@_instrumented("gate")
def node_gate(state: VideoState) -> dict:
    """Measure the finished video and the plan (agents/video/gate.py). The gate never fails
    the run by failing itself: a crash in it is recorded as a warning."""
    out_dir = _out_dir(state)
    try:
        inputs = gate_agent.GateInputs(
            video_path=Path(state["video_path"]),
            thumbnail_path=Path(state["thumbnail_path"]) if state.get("thumbnail_path") else None,
            brief=state["brief"], script=state["script"], storyboard=state["storyboard"], clips=state.get("clips") or [],
            line_timings=state.get("line_timings") or [], overlay_items=state.get("overlay_items") or [],
            music_info=state.get("music_info"),
            guard=gate_agent.build_guard(state["property_input"], state["property_knowledge"], state["brief"]))
        evaluation, measured = gate_agent.evaluate(inputs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("The quality gate could not run")
        measured = state.get("measurements")
        reason = f"the gate could not run ({type(exc).__name__}: {str(exc)[:160]})"
        evaluation = VideoEvaluationResult(
            items=[GateCheck(name="gate", status="warn", detail=reason)], defects=[f"[WARN] gate: {reason}"],
            checks={"gate": f"warn: {reason}"}, gate_result=VideoGate.WARN)
    _write_qa(out_dir, state, evaluation, measured)
    out = {"evaluation": evaluation, "failed_shots": evaluation.failed_shots}
    if measured is not None:
        out["measurements"] = measured
    out.update(_report(state, "gate", f"{evaluation.gate_result.value.upper()}, score {evaluation.quality_score:.0f}/100",
                       *evaluation.defects))
    return out


def _write_qa(out_dir: Path, state: VideoState, evaluation: VideoEvaluationResult,
              measured: Optional[VideoMeasurements]) -> None:
    """qa.json: the verdict, every check with what was measured, and each shot's photograph."""
    clip_of = {c.shot_index: c for c in state.get("clips") or []}
    shots = [{"shot": s.index + 1, "kind": s.kind, "photo": Path(s.source_path).name if s.source_path else "",
              "seconds": s.seconds, "layout": clip_of[s.index].layout if s.index in clip_of else "",
              "peak_upscale": clip_of[s.index].peak_upscale if s.index in clip_of else 0.0,
              "tags": s.tags, "overlay_text": s.overlay_text} for s in state["storyboard"].shots]
    payload = {"video_id": state["video_id"], "gate_result": evaluation.gate_result.value,
               "score": evaluation.quality_score, "judge": evaluation.judge,
               "defects": evaluation.defects, "not_measured": evaluation.feedback,
               "checks": [c.model_dump() for c in evaluation.items],
               "measurements": measured.model_dump() if measured is not None else None, "shots": shots}
    (out_dir / "qa.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _chain(nodes: list[tuple[str, Callable]]):
    graph = StateGraph(VideoState)
    for node_id, fn in nodes:
        graph.add_node(node_id, fn)
    graph.set_entry_point(nodes[0][0])
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        graph.add_edge(a, b)
    graph.add_edge(nodes[-1][0], END)
    return graph.compile()


def build_plan_graph():
    return _chain([("n_brief", node_brief), ("n_fact_ledger", node_fact_ledger), ("n_asset_audit", node_asset_audit),
                   ("n_script", node_script), ("n_storyboard", node_storyboard)])


def build_produce_graph():
    return _chain([("n_voice_music", node_voice_music), ("n_motion", node_motion), ("n_overlays", node_overlays),
                   ("n_compose", node_compose), ("n_gate", node_gate)])


_PLAN_GRAPH = None
_PRODUCE_GRAPH = None


def get_plan_graph():
    global _PLAN_GRAPH  # noqa: PLW0603
    if _PLAN_GRAPH is None:
        _PLAN_GRAPH = build_plan_graph()
    return _PLAN_GRAPH


def get_produce_graph():
    global _PRODUCE_GRAPH  # noqa: PLW0603
    if _PRODUCE_GRAPH is None:
        _PRODUCE_GRAPH = build_produce_graph()
    return _PRODUCE_GRAPH


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def auto_approve(storyboard: Storyboard) -> Storyboard:
    """The approval step for a run no one is watching: take the plan as it is."""
    return storyboard


def run_plan_workflow(property_dir: Path, overrides: Optional[dict] = None, *, offline: bool = False,
                      draft: bool = False, refresh_ledger: bool = False,
                      approve: Callable[[Storyboard], Storyboard] = auto_approve,
                      video_id: Optional[str] = None) -> VideoGenerationRecord:
    """Plans the video for the property at `property_dir` (e.g.
    data/properties/ANAMIKA-HIGH-POINT): brief, ledger, asset audit, script
    and storyboard. `overrides` are options laid over video_input.json.
    `approve` gets the storyboard and returns the one to use. Saves and
    returns a record with status "planned" (or "failed", with the reason).
    `video_id` names a record that already exists (a background job creates
    it before it starts); without one a new record is made."""
    return _run(property_dir, overrides, offline, draft, approve, produce=False, refresh_ledger=refresh_ledger,
                video_id=video_id)


def run_video_workflow(property_dir: Path, overrides: Optional[dict] = None, *, offline: bool = False,
                       draft: bool = False, refresh_ledger: bool = False,
                       approve: Callable[[Storyboard], Storyboard] = auto_approve,
                       video_id: Optional[str] = None) -> VideoGenerationRecord:
    """Plan, approve, then produce: voice, motion, overlays and compose.
    Saves and returns a record with the gate's verdict as its status
    (passed, warned or failed)."""
    return _run(property_dir, overrides, offline, draft, approve, produce=True, refresh_ledger=refresh_ledger,
                video_id=video_id)


def run_produce_workflow(video_id: str) -> VideoGenerationRecord:
    """Produces the video of a planned record, in whatever process this is.

    Reads the record (brief, script, storyboard, the steps already run) and
    plan_state.json (everything else the plan graph held in memory) and runs
    the produce graph into the same folder, saving the same record. It never
    plans again: when plan_state.json is missing or from another version the
    record is saved as failed, with the reason. Raises WorkflowError when
    there is no such record, or it is not waiting to be produced."""
    repository.init_db()
    record = repository.get_generation(video_id)
    if record is None:
        raise WorkflowError(f"There is no video {video_id}.")
    if record.status == STATUS_PLANNED:
        record = repository.begin_produce(video_id)           # one job at a time, claimed in one transaction
    elif record.status != STATUS_PRODUCING:
        raise WorkflowError(f"Video {video_id} is '{record.status}'; only a planned video can be produced.")
    record.pid = os.getpid()
    repository.set_pid(video_id, record.pid)

    step_log: dict[str, AgentStepResult] = {s.node: s for s in (record.workflow_evaluation.steps
                                                                if record.workflow_evaluation else [])}
    token = _step_log_var.set(step_log)
    progress_token = _progress_var.set(_progress_saver(video_id, step_log))
    blocked: list[str] = []
    try:
        try:
            if record.brief is None or record.script is None or record.storyboard is None:
                raise WorkflowError(f"The plan of {video_id} is incomplete (no brief, script or storyboard).")
            saved = video_plan_state.load(video_id)
            state: VideoState = {
                "video_id": video_id, "property_dir": saved.property_dir, "video_overrides": saved.overrides,
                "offline": saved.offline, "draft": saved.draft, "refresh_ledger": saved.refresh_ledger,
                "property_input": saved.property_input, "video_input": saved.video_input,
                "brief": record.brief, "brand_kit": saved.brand_kit, "property_knowledge": saved.property_knowledge,
                "image_analyses": saved.image_analyses, "ranked_assets": saved.ranked_assets,
                "image_plan": video_plan_state.plan_from_data(saved.image_plan), "asset_notes": saved.asset_notes,
                "script": record.script, "concept": saved.concept, "storyboard": record.storyboard,
                "agent_reports": dict(record.agent_reports)}
            with policy.video_run_policy() as call_policy, tracing.trace_generation(video_id, record.property_id):
                state = get_produce_graph().invoke(state, {"recursion_limit": 25})
                _fill_produce(record, state)
                blocked = list(call_policy.blocked) if call_policy else []
            _set_verdict(record)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Producing video %s failed", video_id)
            record.status = "failed"
            record.error = str(exc)
    finally:
        _progress_var.reset(progress_token)
        _step_log_var.reset(token)
    return _finish(record, step_log, blocked)


def _run(property_dir: Path, overrides: Optional[dict], offline: bool, draft: bool,
         approve: Callable[[Storyboard], Storyboard], produce: bool, refresh_ledger: bool = False,
         video_id: Optional[str] = None) -> VideoGenerationRecord:
    repository.init_db()
    property_dir = Path(property_dir)
    if video_id:
        record = repository.get_generation(video_id)
        if record is None:
            raise WorkflowError(f"There is no video {video_id}.")
        record.status = STATUS_PLANNING
    else:
        video_id = f"vid_{uuid.uuid4().hex[:10]}"
        record = VideoGenerationRecord(
            video_id=video_id, property_id=property_dir.name, status=STATUS_PLANNING,
            request=VideoRequest(property_id=property_dir.name, overrides=dict(overrides or {}), offline=offline,
                                 draft=draft, refresh_ledger=refresh_ledger))
    record.pid = os.getpid()
    repository.save_generation(record)

    initial: VideoState = {"video_id": video_id, "property_dir": str(property_dir),
                           "video_overrides": dict(overrides or {}), "offline": offline, "draft": draft,
                           "refresh_ledger": refresh_ledger, "agent_reports": {}}
    step_log: dict[str, AgentStepResult] = {}
    token = _step_log_var.set(step_log)
    progress_token = _progress_var.set(_progress_saver(video_id, step_log))
    blocked: list[str] = []
    try:
        try:
            with policy.video_run_policy() as call_policy, tracing.trace_generation(video_id, record.property_id):
                state = get_plan_graph().invoke(initial, {"recursion_limit": 25})
                board = approve(state["storyboard"])
                if board is not state["storyboard"]:
                    problems = storyboard_agent.validate_plan(board, state["brief"], state["ranked_assets"],
                                                              state["image_plan"])
                    if problems:
                        raise WorkflowError("The approved storyboard is not valid: " + "; ".join(problems))
                state = {**state, "storyboard": board}
                _fill_plan(record, state)
                video_plan_state.save(video_plan_state.build(
                    video_id, state, property_dir=property_dir, offline=offline, draft=draft,
                    refresh_ledger=refresh_ledger))
                if produce:
                    repository.update_progress(video_id, status=STATUS_PRODUCING)
                    state = get_produce_graph().invoke(state, {"recursion_limit": 25})
                    _fill_produce(record, state)
                blocked = list(call_policy.blocked) if call_policy else []
            if produce:
                _set_verdict(record)
            else:
                record.status = STATUS_PLANNED
        except Exception as exc:  # noqa: BLE001
            logger.exception("Video generation failed for %s", property_dir.name)
            record.status = "failed"
            record.error = str(exc)
    finally:
        _progress_var.reset(progress_token)
        _step_log_var.reset(token)
    return _finish(record, step_log, blocked)


def _set_verdict(record: VideoGenerationRecord) -> None:
    record.status = _status(record)
    if record.status == "failed":
        record.error = "The quality gate failed: " + "; ".join(
            i.detail for i in record.evaluation.items if i.status == "fail")[:400]


def _finish(record: VideoGenerationRecord, step_log: dict[str, AgentStepResult],
            blocked: list[str]) -> VideoGenerationRecord:
    record.notes = list(dict.fromkeys([*record.notes, *(f"Blocked by the run's policy: {b}." for b in blocked)]))
    record.workflow_evaluation = WorkflowEvaluation(steps=_steps(step_log))
    record.steps_done = sum(1 for s in record.workflow_evaluation.steps if s.status == "success")
    record.current_node = ""
    repository.save_generation(record)
    return record


def _status(record: VideoGenerationRecord) -> str:
    """The gate's verdict as the record's status."""
    gate = record.evaluation.gate_result if record.evaluation else VideoGate.WARN
    return {VideoGate.PASS: "passed", VideoGate.WARN: "warned", VideoGate.FAIL: "failed"}[gate]


def _fill_plan(record: VideoGenerationRecord, state: dict) -> None:
    brief, script, board = state["brief"], state["script"], state["storyboard"]
    out_dir = repository.video_output_dir(record.video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "storyboard.json").write_text(board.model_dump_json(indent=2), encoding="utf-8")

    record.property_id = state["property_input"].property_id
    record.output_dir = str(out_dir)
    record.duration_s, record.aspect, record.provider = brief.duration_s, brief.aspect, brief.model
    record.brief, record.script, record.storyboard = brief, script, board
    record.generated_shots = board.generated_shots
    record.agent_reports = dict(state.get("agent_reports") or {})
    record.notes = list(dict.fromkeys([*(state.get("notes") or []), *(state.get("asset_notes") or []),
                                       *script.notes, *board.notes]))


def _fill_produce(record: VideoGenerationRecord, state: dict) -> None:
    board = state["storyboard"]                         # re-timed to the voice
    out_dir = repository.video_output_dir(record.video_id)
    (out_dir / "storyboard.json").write_text(board.model_dump_json(indent=2), encoding="utf-8")

    clips = state.get("clips") or []
    record.storyboard = board
    record.video_path, record.thumbnail_path = state.get("video_path"), state.get("thumbnail_path")
    record.captions_path = state.get("captions_path")
    record.line_timings, record.clips = state.get("line_timings") or [], clips
    record.voice, record.music = state.get("voice_name", ""), state.get("music_track") or ""
    record.music_info = state.get("music_info")
    record.providers_used = dict(Counter(c.provider for c in clips if c.provider))
    record.cost_usd = round(sum(c.cost_usd for c in clips) + (record.music_info.cost_usd if record.music_info else 0)
                            + sum(t.cost_usd for t in record.line_timings), 4)
    record.measurements = state.get("measurements")
    record.evaluation = state.get("evaluation")
    record.actual_duration_s = record.measurements.duration_s if record.measurements else 0.0
    record.defects = list(dict.fromkeys(state.get("defects") or []))
    record.agent_reports = dict(state.get("agent_reports") or {})
    record.notes = list(dict.fromkeys([*record.notes, *(state.get("notes") or [])]))
