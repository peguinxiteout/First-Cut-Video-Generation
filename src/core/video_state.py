"""
The state LangGraph passes between video workflow nodes. Each node reads what
it needs and returns the keys it changes.

Every key a node returns MUST be declared here: LangGraph drops undeclared
keys silently. Graph node ids are prefixed (n_brief, n_script, ...) because
LangGraph also refuses a node whose id equals a state key; the step names
the Workflow tab shows (brief, script, storyboard, ...) are set separately,
in workflows/video.py.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.data.schemas import ImageAnalysis, ImageAsset, PropertyInput, PropertyKnowledge
from src.data.video_schemas import (
    BrandKit, ClipResult, LineTiming, MusicInfo, OverlayItem, RankedAsset, Storyboard, VideoBrief, VideoEvaluationResult,
    VideoInput, VideoMeasurements, VideoScript,
)


class VideoState(TypedDict, total=False):
    # Input
    video_id: str
    property_dir: str               # data/properties/<id>, read by the brief node
    video_overrides: dict[str, Any]  # command-line options, laid over video_input.json
    offline: bool                   # take every fallback; call no model
    draft: bool                     # render small and fast, for previews
    refresh_ledger: bool            # ask the model for the fact ledger again, whatever the cache holds
    property_input: PropertyInput
    images: list[ImageAsset]
    video_input: VideoInput
    documents_text: str

    # Plan
    brief: VideoBrief
    brand_kit: BrandKit
    property_knowledge: PropertyKnowledge
    image_analyses: dict[str, ImageAnalysis]
    ranked_assets: list[RankedAsset]   # photographs best first, with measured sharpness and size
    image_plan: Any                 # image_plan.Plan: which pictures look alike
    asset_notes: list[str]          # including the missing-photo flag
    script: VideoScript
    concept: dict[str, Any]         # pacing and music mood
    storyboard: Storyboard

    # Produce
    voice_tracks: list[str]
    line_timings: list[LineTiming]
    music_track: Optional[str]
    music_info: MusicInfo
    beat_times: list[float]
    keyframes: dict[int, str]       # shot index -> still the clip starts from
    clips: list[ClipResult]
    overlay_frames: dict[int, list[str]]
    overlay_items: list[OverlayItem]
    voice_name: str                 # who spoke: the provider and voice, or 'silent'
    captions_path: str

    # Output and check
    video_path: str
    thumbnail_path: str
    measurements: VideoMeasurements
    evaluation: VideoEvaluationResult
    defects: list[str]
    failed_shots: list[int]
    agent_reports: dict[str, list[str]]
    notes: list[str]                # said to the person with the result
    error: Optional[str]
