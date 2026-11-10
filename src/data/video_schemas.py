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
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.schemas import HumanReview, ImageAnalysis, PropertyInput, PropertyKnowledge, WorkflowEvaluation

VIDEO_DURATIONS = (10, 15, 30, 45, 60)
VIDEO_ASPECTS = ("9:16", "16:9", "1:1")
VIDEO_OBJECTIVES = ("awareness", "site_visit", "launch", "enquiry")
VIDEO_QUALITIES = ("draft", "final")

# Where a record is in its life. planning and producing are the two states in
# which a job is (or should be) running; planned is "waiting for approval".
STATUS_PLANNING, STATUS_PLANNED, STATUS_PRODUCING = "planning", "planned", "producing"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = (STATUS_PLANNING, STATUS_PRODUCING)

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
    burn_captions: bool = True            # print each spoken line on the picture (captions.srt is written either way)
    grade: bool = True                    # the light colour grade; pacing.yaml can switch it off for every video
    pronunciations: dict[str, str] = Field(default_factory=dict)   # spoken respellings for this property, over speech.yaml's

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

    @field_validator("pronunciations")
    @classmethod
    def _pronunciations(cls, value: dict[str, str]) -> dict[str, str]:
        return {k.strip(): str(v).strip() for k, v in value.items() if k.strip() and str(v).strip()}


class VideoRecipe(BaseModel):
    """One entry of configs/video/recipes.yaml: a ready-made start for the Video tab. It sets brief fields and the
    prompt box's words and adds no fact. `prompt` holds {property}, replaced by the chosen property's name."""

    id: str
    name: str
    category: str
    description: str
    duration_s: int
    aspect: str
    objective: str
    tone: str
    music_mood: str
    cta: str = ""
    prompt: str
    thumbnail: str

    @field_validator("id", "name", "category", "description", "prompt", "music_mood", "thumbnail")
    @classmethod
    def _text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("duration_s")
    @classmethod
    def _duration(cls, value: int) -> int:
        if value not in VIDEO_DURATIONS:
            raise ValueError(f"duration_s must be one of {list(VIDEO_DURATIONS)}, got {value}")
        return value

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, value: str) -> str:
        if value not in VIDEO_ASPECTS:
            raise ValueError(f"aspect must be one of {list(VIDEO_ASPECTS)}, got {value!r}")
        return value

    @field_validator("objective")
    @classmethod
    def _objective(cls, value: str) -> str:
        if value not in VIDEO_OBJECTIVES:
            raise ValueError(f"objective must be one of {list(VIDEO_OBJECTIVES)}, got {value!r}")
        return value


class PromptBrief(BaseModel):
    """What a free-text prompt sets (the Video tab's prompt box): the options of VideoInput it
    can steer, each None when the prompt does not say it. Length and shape come from rules only.
    The voice, the opening picture, generated frames and the grade are not here: a prompt never
    sets them (a key a model sends for one is ignored)."""

    model_config = ConfigDict(extra="ignore")

    duration_s: Optional[int] = None
    aspect: Optional[str] = None
    objective: Optional[str] = None
    audience: Optional[str] = None
    highlights: Optional[list[str]] = None
    tone: Optional[str] = None
    music_mood: Optional[str] = None
    cta: Optional[str] = None
    creative_direction: Optional[str] = None
    burn_captions: Optional[bool] = None
    pronunciations: Optional[dict[str, str]] = None

    @field_validator("duration_s")
    @classmethod
    def _duration(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value not in VIDEO_DURATIONS:
            raise ValueError(f"duration_s must be one of {list(VIDEO_DURATIONS)}, got {value}")
        return value

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in VIDEO_ASPECTS:
            raise ValueError(f"aspect must be one of {list(VIDEO_ASPECTS)}, got {value!r}")
        return value

    @field_validator("objective")
    @classmethod
    def _objective(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in VIDEO_OBJECTIVES:
            raise ValueError(f"objective must be one of {list(VIDEO_OBJECTIVES)}, got {value!r}")
        return value


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


class PhotoAnalysis(ImageAnalysis):
    """The vision reading of a picture for a video: the brochure's reading, plus a plain
    caption and the tags of what is clearly visible (src/agents/video/photo_tags.py)."""

    caption: str = ""
    tags: list[str] = Field(default_factory=list)


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
    caption: str = ""                     # what is visible, in a sentence (empty without a vision reading)
    tags: list[str] = Field(default_factory=list)   # what is clearly visible, as photo_tags names (empty without vision)
    # Amenities the model tagged that the caption does not say: not evidence for a line or a chip.
    weak_tags: list[str] = Field(default_factory=list)
    room: str = ""                        # the room of a home it shows (living, bedroom ...), from category and caption
    cluster: int = -1                     # pictures that look alike share one (image_plan)
    duplicate_of: str = ""                # the better picture of its cluster, when this is not it
    usable: bool = True
    note: str = ""                        # why it is not usable, or a caution

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    @property
    def read_by_vision(self) -> bool:
        """The picture was read by a vision model (it has tags, whether or not a caption backs them)."""
        return bool(self.tags or self.weak_tags)


class ScriptLine(BaseModel):
    text: str
    role: str = "body"                    # hook | feature | proof | cta
    topic: str = ""                       # a key of src/agents/brochure/topics.py, for choosing the photograph
    tags: list[str] = Field(default_factory=list)   # what a photograph must show to carry this line (photo_tags names)
    photo: str = ""                       # the picture the script set aside for it, when it needs one
    source: str = ""                      # the fact the line comes from ("unit types", "selling point: ..."), to check it
    scene: str = ""                       # the kind of picture it wants: exterior | interior | amenity | view | any

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
    caption: str = ""                     # what is visible in it, in a sentence (vision reading; empty offline)
    tags: list[str] = Field(default_factory=list)   # what is clearly visible in it (photo_tags names; empty offline)
    image_prompt: str = ""                # generated shots only
    generated_kind: str = ""              # one of the sector's allowed kinds in configs/video/styles.yaml
    camera: str = "push_in"
    motion_prompt: str = ""
    seconds: float = 3.0
    focal: str = "50% 50%"                # object-position style, as focal.focal_point returns it
    overlay_text: list[str] = Field(default_factory=list)
    line_ids: list[int] = Field(default_factory=list)
    reuse: bool = False                   # a second crop of a photograph already used
    layout: str = ""                      # fill (cropped to the frame) | fit (whole photo on a blurred copy); set when the clip is made
    peak_upscale: float = 0.0             # the most the photograph's pixels are enlarged at any moment of the shot

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
    spoken: str = ""                      # the text the voice was given (respelled where speech.yaml says), not what is printed
    cost_usd: float = 0.0                 # what the line cost to speak; 0 from the cache

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class OverlayItem(BaseModel):
    """One transparent picture of on-screen text (or of the scrim behind it),
    shown for one stretch of the video. A title, a chip or a card line is its
    own item so each can come in on its own."""

    shot_index: int
    kind: str = "title"                   # title (name, price, chips) | end_card | caption
    key: str = ""                         # which piece of its kind: heading, chip0, cta, caption, ...
    path: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    texts: list[str] = Field(default_factory=list)
    renderer: str = "browser"             # browser | fallback
    fade_s: float = 0.3                   # fade in and out
    slide_px: int = 0                     # slides up from this far below while it fades in
    rects: list[dict[str, float | str]] = Field(default_factory=list)   # where its text landed: x, y, w, h in frame pixels


class ClipResult(BaseModel):
    """What happened to one shot's clip. A failed clip is a record, not an
    error: the shot is still filled by pan-and-zoom."""

    shot_index: int
    path: str = ""
    provider: str = ""
    model_id: str = ""
    mode: str = "kenburns"                # model | kenburns | card
    layout: str = ""                      # fill | fit (real shots)
    peak_upscale: float = 0.0
    ok: bool = True
    reason: str = ""                      # why a requested provider was not used
    from_cache: bool = False
    duration_s: float = 0.0
    cost_usd: float = 0.0
    first_frame_score: Optional[float] = None


class MusicInfo(BaseModel):
    """What the background music is and how it was had. `provider` is "none"
    when there is no music; `notes` says why each provider ahead of the one
    that answered did not."""

    provider: str = "none"                # beatoven | lyria | library | none
    model_id: str = ""
    track_id: str = ""                    # the generator's own id for the track
    prompt: str = ""                      # what was asked of a generator (empty for a library track)
    path: str = ""
    title: str = ""                       # a library track's name
    licence: str = ""                     # a library track's licence.txt first line; for a generated track, what is known of its terms
    cost_usd: float = 0.0
    from_cache: bool = False
    duration_s: float = 0.0               # of the track as the provider made it
    notes: list[str] = Field(default_factory=list)


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
    pix_fmt: str = ""
    loudness_lufs: Optional[float] = None
    true_peak_dbfs: Optional[float] = None
    black_segments: int = 0
    black_s: float = 0.0                  # in all
    frozen_s: float = 0.0                 # in all, the still closing card included
    black_spans: list[list[float]] = Field(default_factory=list)     # [start, end] seconds
    frozen_spans: list[list[float]] = Field(default_factory=list)


class VideoGate(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GateCheck(BaseModel):
    """One measured check of the quality gate (configs/video/evaluation.yaml)."""

    name: str
    status: str = "pass"                  # pass | warn | fail | skipped (could not be, or need not be, measured)
    detail: str = ""                      # what was measured, in words
    weight: float = 0.0
    hard: bool = False                    # a failure fails the whole gate
    shots: list[int] = Field(default_factory=list)      # the shots (1-based) it is about
    values: dict[str, Any] = Field(default_factory=dict)


class VideoEvaluationResult(BaseModel):
    checks: dict[str, str] = Field(default_factory=dict)       # check -> "pass" | plain-language result
    items: list[GateCheck] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)           # every check that did not pass, "[FAIL] ..." or "[WARN] ..."
    feedback: list[str] = Field(default_factory=list)
    failed_shots: list[int] = Field(default_factory=list)
    quality_score: float = 0                                   # 0-100, the weighted share of passes
    gate_result: VideoGate = VideoGate.FAIL
    judge: str = "measured"               # measured | model


# ---------------------------------------------------------------------------
# Request and plan state
# ---------------------------------------------------------------------------

class VideoRequest(BaseModel):
    """What was asked for, kept on the record before the job starts so the job
    reads it from the database and not from its command line."""
    property_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)   # laid over video_input.json
    offline: bool = False
    draft: bool = False
    refresh_ledger: bool = False
    # True: plan and produce in one job, with no stop for approval (the Video tab's Plan toggle off).
    auto_approve: bool = False
    # The prompt box: the text as typed, how its options were worked out ("model" or "rules"), and what
    # was taken from it or left out. All empty for a video made from the form.
    prompt: str = ""
    brief_source: str = ""
    brief_notes: list[str] = Field(default_factory=list)
    # Where each option of the brief came from: the toolbar, the form, the prompt, the property's saved
    # defaults (video_input.json) or the built-in default. Empty for a video made before these were kept.
    sources: dict[str, str] = Field(default_factory=dict)
    # The recipe the video was started from (configs/video/recipes.yaml), "" when none.
    recipe_id: str = ""


PLAN_STATE_VERSION = 1
PLAN_STATE_FILE = "plan_state.json"


class PlanState(BaseModel):
    """What the produce half needs that the record does not hold: the part of
    the plan graph's state that lives only in memory. Written beside
    script.json when the plan is done; the produce job reads it and never
    builds any of it again. JSON only, no keys. Paths under the project are
    written relative to it (see data/video_plan_state.py)."""
    version: int = PLAN_STATE_VERSION
    video_id: str
    property_dir: str
    offline: bool = False
    draft: bool = False
    refresh_ledger: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict)
    video_input: VideoInput
    property_input: PropertyInput
    property_knowledge: PropertyKnowledge
    brand_kit: BrandKit
    image_analyses: dict[str, PhotoAnalysis] = Field(default_factory=dict)
    ranked_assets: list[RankedAsset] = Field(default_factory=list)
    image_plan: dict[str, Any] = Field(default_factory=dict)     # image_plan.Plan as plain data
    asset_notes: list[str] = Field(default_factory=list)
    concept: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_analyses", mode="before")
    @classmethod
    def _photo_readings(cls, value):
        """The offline reading is a plain ImageAnalysis; it is kept as the PhotoAnalysis it is a part of."""
        if not isinstance(value, dict):
            return value
        return {k: PhotoAnalysis(**v.model_dump()) if isinstance(v, ImageAnalysis) and not isinstance(v, PhotoAnalysis)
                else v for k, v in value.items()}


class VideoSummary(BaseModel):
    """One row of the history, read from the flat columns only (no record_json)."""
    video_id: str
    property_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None  # when the running job (plan, or produce after approval) started
    status: str
    current_node: str = ""
    steps_done: int = 0
    duration_s: int = 15
    aspect: str = "9:16"
    provider: str = "auto"
    cost_usd: float = 0.0
    gate_result: str = ""                 # pass | warn | fail, once the gate has run
    quality_score: Optional[float] = None
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    human_review_status: str = "pending"


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

class VideoGenerationRecord(BaseModel):
    video_id: str
    property_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"               # pending | planning | planned | producing | passed | warned | failed | cancelled (passed, warned, failed: the gate's result; a gate FAIL keeps the MP4)
    error: Optional[str] = None
    request: Optional[VideoRequest] = None
    # Job state, kept in their own columns and updated after every node
    # (video_repository.update_progress); the record_json copy can be older.
    current_node: str = ""                # the step running now, or the last one run
    steps_done: int = 0
    updated_at: Optional[datetime] = None
    pid: Optional[int] = None             # the job's process, while it runs

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
    music_info: Optional[MusicInfo] = None

    measurements: Optional[VideoMeasurements] = None
    evaluation: Optional[VideoEvaluationResult] = None
    workflow_evaluation: Optional[WorkflowEvaluation] = None
    human_review: HumanReview = Field(default_factory=HumanReview)
    notes: list[str] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)
    agent_reports: dict[str, list[str]] = Field(default_factory=dict)
