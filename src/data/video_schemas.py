"""
Pydantic schemas for the video workflow: what a person asks for
(`VideoInput`, the `video_input.json` next to a property's `property.json`),
what the workflow works out from it (brief, script, storyboard, timings,
clips), what was measured on the finished file, and the record that is saved.

Kept apart from schemas.py, which brochure and blog share. The generic
pieces (AgentStepResult, WorkflowEvaluation, HumanReview) are imported from
there, not copied.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.core.types import QualityGateResult
from src.data.schemas import HumanReview, WorkflowEvaluation

VIDEO_DURATIONS = (10, 15, 30, 45, 60)
VIDEO_ASPECTS = ("9:16", "16:9", "1:1")
VIDEO_OBJECTIVES = ("awareness", "site_visit", "launch", "enquiry")
VIDEO_QUALITIES = ("draft", "final")

# What a shot's camera does to its still. The video model (or pan-and-zoom)
# only adds this movement; it never changes what the picture shows.
CAMERA_MOVES = ("push_in", "pull_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "still")
# real: a photograph the person supplied. generated: a frame made for the
# shot, allowed only where configs/video/styles.yaml says so. card: palette,
# logo and text, no photograph (the end card, or a generated shot with no
# image model).
SHOT_KINDS = ("real", "generated", "card")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class VideoInput(BaseModel):
    """data/properties/<id>/video_input.json. Every field is optional so a
    property with no file still produces a video."""

    duration_s: int = 15
    aspect: str = "9:16"
    audience: str = ""
    objective: str = "awareness"          # awareness | site_visit | launch | enquiry
    highlights: list[str] = Field(default_factory=list)
    tone: str = ""
    voice: str = ""                       # a key of configs/video/speech.yaml; blank = its default
    music_mood: str = ""
    model: str = "auto"                   # auto | a provider id from configs/video/providers.yaml
    cta: str = ""
    creative_direction: str = ""
    allow_generated: bool = False
    hero_image: str = ""                  # an image id or file name: the picture the video opens on

    @field_validator("duration_s")
    @classmethod
    def _duration(cls, value: int) -> int:
        if value not in VIDEO_DURATIONS:
            raise ValueError(f"duration_s must be one of {list(VIDEO_DURATIONS)}, got {value}")
        return value

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, value: str) -> str:
        value = value.strip().replace("x", ":").replace("/", ":")
        if value not in VIDEO_ASPECTS:
            raise ValueError(f"aspect must be one of {list(VIDEO_ASPECTS)}, got {value!r}")
        return value

    @field_validator("objective")
    @classmethod
    def _objective(cls, value: str) -> str:
        value = value.strip().lower().replace(" ", "_").replace("-", "_")
        if value not in VIDEO_OBJECTIVES:
            raise ValueError(f"objective must be one of {list(VIDEO_OBJECTIVES)}, got {value!r}")
        return value

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        return value.strip().lower() or "auto"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class VideoBrief(VideoInput):
    """The input with every default resolved and the property's identity
    attached: what every later step reads instead of re-deriving it."""

    property_id: str
    name: str
    developer: str = ""
    category: str = "mid"                 # affordable | mid | luxury
    lang: str = "en"
    width: int = 1080
    height: int = 1920
    fps: int = 25
    quality: str = "final"                # draft renders small and fast
    target_words: int = 32                # about 2.1 words a second at a natural pace
    n_shots: int = 5
    palette_name: str = ""
    cta_text: str = ""                    # the call to action as it is said and printed

    @field_validator("quality")
    @classmethod
    def _quality(cls, value: str) -> str:
        if value not in VIDEO_QUALITIES:
            raise ValueError(f"quality must be one of {list(VIDEO_QUALITIES)}, got {value!r}")
        return value


class BrandKit(BaseModel):
    """What our code draws on every frame: the real logo file, palette
    tokens and the identifiers a buyer expects to see."""

    name: str
    logo_path: Optional[str] = None
    palette_name: str = ""
    palette: dict[str, str] = Field(default_factory=dict)
    accent: str = ""
    font_family: str = ""
    rera_number: str = ""
    phone: str = ""
    website: str = ""


class RankedAsset(BaseModel):
    """One uploaded picture as the video sees it: what it is, how sharp and
    how big, and whether it can carry a shot."""

    image_id: str
    filename: str
    path: str
    width: int = 0
    height: int = 0
    sharpness: float = 0.0                # variance of the Laplacian on a small greyscale copy
    quality: float = 0.0                  # 0-10, the vision reading blended with sharpness
    category: str = ""                    # a vision category: exterior, lifestyle, floorplan, ...
    topic: str = ""                       # what it shows, a key of brochure/topics.py (pool, exterior, ...)
    cluster: int = -1                     # pictures that look alike share one (image_plan)
    duplicate_of: str = ""                # the better picture of its cluster, when this is not it
    usable: bool = True
    note: str = ""                        # why it is not usable, or a caution

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0


class ScriptLine(BaseModel):
    text: str
    role: str = "body"                    # hook | feature | proof | cta
    topic: str = ""                       # a key of src/agents/brochure/topics.py, for choosing the photograph

    @property
    def words(self) -> int:
        return len(self.text.split())


class VideoScript(BaseModel):
    lines: list[ScriptLine] = Field(default_factory=list)
    on_screen: list[str] = Field(default_factory=list)     # the title block: name, and price if the category shows it
    chips: list[str] = Field(default_factory=list)         # one or two short highlights, drawn as they are spoken
    chip_lines: list[int] = Field(default_factory=list)    # the line each chip belongs to, so it shows with that line
    end_card: list[str] = Field(default_factory=list)      # RERA and contact, drawn on the closing card
    cta: str = ""
    word_count: int = 0
    mode: str = "template"                # template | model
    notes: list[str] = Field(default_factory=list)


class Shot(BaseModel):
    index: int
    kind: str = "real"
    image_id: str = ""
    source_path: str = ""
    subject: str = ""                     # what the photograph shows (its topic), for the person reading the plan
    image_prompt: str = ""                # generated shots only
    generated_kind: str = ""              # one of the sector's allowed kinds in configs/video/styles.yaml
    camera: str = "push_in"
    motion_prompt: str = ""
    seconds: float = 3.0
    focal: str = "50% 50%"                # object-position style, as focal.focal_point returns it
    overlay_text: list[str] = Field(default_factory=list)
    line_ids: list[int] = Field(default_factory=list)
    reuse: bool = False                   # a second crop of a photograph already used

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        if value not in SHOT_KINDS:
            raise ValueError(f"kind must be one of {list(SHOT_KINDS)}, got {value!r}")
        return value

    @field_validator("camera")
    @classmethod
    def _camera(cls, value: str) -> str:
        if value not in CAMERA_MOVES:
            raise ValueError(f"camera must be one of {list(CAMERA_MOVES)}, got {value!r}")
        return value


class Storyboard(BaseModel):
    shots: list[Shot] = Field(default_factory=list)
    planner_mode: str = "rules"           # rules | model
    notes: list[str] = Field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return round(sum(s.seconds for s in self.shots), 3)

    @property
    def generated_shots(self) -> list[int]:
        return [s.index for s in self.shots if s.kind == "generated"]


# ---------------------------------------------------------------------------
# Produce
# ---------------------------------------------------------------------------

class LineTiming(BaseModel):
    line_index: int
    start_s: float
    end_s: float
    path: str = ""
    provider: str = ""                    # the speech provider that made it
    voice: str = ""
    mode: str = "speech"                  # speech | silent (no voice available; captions carry the line)

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class OverlayItem(BaseModel):
    """One transparent picture of on-screen text, shown over one shot."""

    shot_index: int
    kind: str = "title"                   # title (name, price, chips) | end_card
    path: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    texts: list[str] = Field(default_factory=list)
    renderer: str = "browser"             # browser | fallback


class ClipResult(BaseModel):
    """What happened to one shot's clip. A failed clip is a record, not an
    error: the shot is still filled by pan-and-zoom."""

    shot_index: int
    path: str = ""
    provider: str = ""
    model_id: str = ""
    mode: str = "kenburns"                # model | kenburns | card
    ok: bool = True
    reason: str = ""                      # why a requested provider was not used
    from_cache: bool = False
    duration_s: float = 0.0
    cost_usd: float = 0.0
    first_frame_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

class VideoMeasurements(BaseModel):
    """Read from the finished file, not from what the workflow intended."""

    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    has_audio: bool = False
    size_bytes: int = 0
    loudness_lufs: Optional[float] = None
    true_peak_dbfs: Optional[float] = None
    black_segments: int = 0
    frozen_s: float = 0.0


class VideoEvaluationResult(BaseModel):
    checks: dict[str, str] = Field(default_factory=dict)       # check -> "pass" | plain-language result
    defects: list[str] = Field(default_factory=list)           # reasons the video should not go out
    feedback: list[str] = Field(default_factory=list)
    failed_shots: list[int] = Field(default_factory=list)
    quality_score: float = 0
    gate_result: QualityGateResult = QualityGateResult.FAIL
    judge: str = "measured"               # measured | model


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

class VideoGenerationRecord(BaseModel):
    video_id: str
    property_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"               # pending | running | planned | completed | failed
    error: Optional[str] = None

    output_dir: Optional[str] = None
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    captions_path: Optional[str] = None

    duration_s: int = 15                  # asked for
    actual_duration_s: float = 0.0        # measured
    aspect: str = "9:16"
    provider: str = "auto"                # the choice made; providers_used says what ran
    providers_used: dict[str, int] = Field(default_factory=dict)   # provider id -> shots
    cost_usd: float = 0.0
    generated_shots: list[int] = Field(default_factory=list)

    brief: Optional[VideoBrief] = None
    script: Optional[VideoScript] = None
    storyboard: Optional[Storyboard] = None
    line_timings: list[LineTiming] = Field(default_factory=list)
    clips: list[ClipResult] = Field(default_factory=list)
    voice: str = ""                       # speech provider and voice that spoke
    music: str = ""

    measurements: Optional[VideoMeasurements] = None
    evaluation: Optional[VideoEvaluationResult] = None
    workflow_evaluation: Optional[WorkflowEvaluation] = None
    human_review: HumanReview = Field(default_factory=HumanReview)
    notes: list[str] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)
    agent_reports: dict[str, list[str]] = Field(default_factory=dict)
