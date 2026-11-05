"""
plan_state.json: what the produce half of a video run needs and the record
does not hold. The plan graph keeps these in memory (the fact ledger, the
brand kit, the ranked photographs, which pictures look alike, the property as
read); once it returns they are gone. A produce job starts in another process,
possibly hours later, and must make the video the person approved, so they
are written beside script.json and read back here, never worked out again: a
second fact ledger could differ from the first, and the gate checks the
numbers against it.

Paths under the project are stored relative to it, so the folder can move.
The file holds no keys.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from src.agents.brochure.image_plan import Plan
from src.core.config import PROJECT_ROOT
from src.core.exceptions import WorkflowError
from src.data import video_repository
from src.data.video_schemas import PLAN_STATE_FILE, PLAN_STATE_VERSION, PlanState

logger = logging.getLogger(__name__)


def plan_state_path(video_id: str) -> Path:
    return video_repository.video_output_dir(video_id) / PLAN_STATE_FILE


def _relative(path: str | None) -> str | None:
    """`path` under the project root as a relative path (forward slashes); anything else as it is."""
    if not path:
        return path
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path


def _absolute(path: str | None) -> str | None:
    if not path or Path(path).is_absolute():
        return path
    return str((PROJECT_ROOT / path).resolve())


def plan_to_data(plan: Plan) -> dict:
    data = asdict(plan)
    data["members"] = {str(k): list(v) for k, v in plan.members.items()}      # JSON keys are text
    return data


def plan_from_data(data: dict) -> Plan:
    return Plan(cluster_of=dict(data.get("cluster_of") or {}),
                members={int(k): list(v) for k, v in (data.get("members") or {}).items()},
                notes=list(data.get("notes") or []))


def build(video_id: str, state: dict, *, property_dir: Path, offline: bool, draft: bool,
          refresh_ledger: bool) -> PlanState:
    """The plan state from the plan graph's final state."""
    brand = state["brand_kit"].model_copy(update={"logo_path": _relative(state["brand_kit"].logo_path)})
    ranked = [a.model_copy(update={"path": _relative(a.path)}) for a in state["ranked_assets"]]
    return PlanState(
        video_id=video_id, property_dir=_relative(str(Path(property_dir).resolve())), offline=offline, draft=draft,
        refresh_ledger=refresh_ledger, overrides=dict(state.get("video_overrides") or {}),
        video_input=state["video_input"], property_input=state["property_input"],
        property_knowledge=state["property_knowledge"], brand_kit=brand, image_analyses=dict(state["image_analyses"]),
        ranked_assets=ranked, image_plan=plan_to_data(state["image_plan"]),
        asset_notes=list(state.get("asset_notes") or []), concept=dict(state.get("concept") or {}))


def save(plan_state: PlanState) -> Path:
    path = plan_state_path(plan_state.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load(video_id: str) -> PlanState:
    """The saved plan state, paths made absolute. Raises WorkflowError, in
    words, when the file is missing, unreadable or from another version: the
    caller fails the job; it does not plan again."""
    path = plan_state_path(video_id)
    if not path.is_file():
        raise WorkflowError(f"{PLAN_STATE_FILE} is missing for {video_id} ({path}); the plan has to be made again.")
    try:
        loaded = PlanState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise WorkflowError(f"{PLAN_STATE_FILE} for {video_id} cannot be read: {' '.join(str(exc).split())[:300]}; "
                            "the plan has to be made again.") from exc
    if loaded.version != PLAN_STATE_VERSION:
        raise WorkflowError(f"{PLAN_STATE_FILE} for {video_id} is version {loaded.version}; this code reads "
                            f"version {PLAN_STATE_VERSION}. The plan has to be made again.")
    if loaded.video_id != video_id:
        raise WorkflowError(f"{PLAN_STATE_FILE} in the folder of {video_id} belongs to {loaded.video_id}.")
    brand = loaded.brand_kit.model_copy(update={"logo_path": _absolute(loaded.brand_kit.logo_path)})
    ranked = [a.model_copy(update={"path": _absolute(a.path)}) for a in loaded.ranked_assets]
    return loaded.model_copy(update={"brand_kit": brand, "ranked_assets": ranked,
                                     "property_dir": _absolute(loaded.property_dir)})
