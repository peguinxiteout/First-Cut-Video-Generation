"""
Pydantic schemas for every structured object that flows through the brochure
workflow. Agents produce/consume these; the renderer and evaluators read them.
Keeping these centralized means every agent, the API, and the tests agree on
one shape.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.core.types import ImageCategory, ImageRole, QualityGateResult


# ---------------------------------------------------------------------------
# Property input (this is the file YOU fill in for each property)
# ---------------------------------------------------------------------------

class PropertyLocation(BaseModel):
    city: str
    area: Optional[str] = None
    address: Optional[str] = None


class PropertyContact(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    # Where a buyer visits: a site office or experience centre. Falls back
    # to the property's own address when blank.
    address: Optional[str] = None


# ---------------------------------------------------------------------------
# Structured property data — the modules a brochure is MADE of
#
# The print review's list of what the reference brochures carry and the
# generated ones did not was, almost entirely, a list of STRUCTURED
# modules: "a unit schedule with carpet and built-up areas", "a location
# map with distance chips", "an amenities index grouped by zone", "a
# developer capability page with real numbers", "the booking steps".
#
# None of those can be written by a language model, because every one of
# them is a table of facts about a particular building. They are typed
# here so a person can enter them once, and every module below is shown
# ONLY where its data exists — a property that gives no landmarks has no
# location module, rather than a location module full of plausible
# landmarks. That is the same rule the rest of the system runs on and it
# is the reason these are fields rather than prompts.
# ---------------------------------------------------------------------------

class UnitType(BaseModel):
    """One row of the unit schedule: what a buyer is actually choosing between."""
    name: str                      # "3 BHK + Study"
    carpet_area: str = ""          # "1,240 sq ft"
    built_up_area: str = ""
    price: str = ""
    # A floor plan among the uploaded images, by filename. The drawing is
    # never cropped and never invented; naming it here is what ties a plan
    # to its row in the schedule.
    plan_image: str = ""


class Landmark(BaseModel):
    """Somewhere a resident goes, and how far it is."""
    name: str
    distance: str = ""             # "2.4 km"
    minutes: str = ""              # "6 min"
    # One of the categories the icon set draws: transit, school, hospital,
    # retail, work, leisure, airport. Anything else gets a neutral mark.
    category: str = ""


class AmenityGroup(BaseModel):
    """Amenities as a reader uses them — by zone, not as one list of forty."""
    zone: str                      # "Clubhouse", "Outdoors", "Wellness"
    items: list[str] = Field(default_factory=list)


class DeveloperStat(BaseModel):
    """A number about the developer, and what it counts."""
    value: str                     # "28"
    label: str                     # "years building in the NCR"


class BookingStep(BaseModel):
    """One step of how a buyer actually buys."""
    title: str
    detail: str = ""


class Testimonial(BaseModel):
    quote: str
    attribution: str = ""


class SpecGroup(BaseModel):
    """Specifications as the trades group them."""
    heading: str                   # "Flooring", "Kitchen", "Safety"
    items: list[str] = Field(default_factory=list)


class Highlight(BaseModel):
    """One reason to buy, as a short label with an icon.

    Mahindra's cover-adjacent grid of ten of these — "Location with better
    growth potential", "2 min from planned Rajnouli metro station" — is
    the densest page of argument in any of the three reference brochures,
    and the generated books had nothing like it."""
    label: str
    icon: str = ""                 # one of the drawn icon keys; blank is fine


class GreenStat(BaseModel):
    """A sustainability figure and what it measures."""
    value: str                     # "45%"
    label: str                     # "water saving"


class FloorBand(BaseModel):
    """One band of the building, from the ground up.

    The review asked for a "cutaway infographic" — the section drawing
    that shows which floors hold what. A cutaway is a drawing of a
    PARTICULAR building and nothing here can author one truthfully: the
    system has photographs and a site plan, not a section. What it can do
    honestly is draw the stack a person describes, band by band, which is
    the part of a cutaway a reader actually uses.
    """
    label: str                     # "Levels 24-32"
    detail: str = ""               # "Sky villas"
    highlight: bool = False        # the amenity floor, the sky deck


class PropertyInput(BaseModel):
    """Matches data/properties/<id>/property.json"""

    property_id: str
    name: str
    developer: Optional[str] = None
    location: PropertyLocation
    # --- the brochure form (streamlit_app.py) ------------------------------
    # The eleven things the form asks for, in the order it asks. Everything
    # here is OPTIONAL in the schema so a property.json written before the
    # form existed still loads; the form itself makes the essentials
    # mandatory (src/data/property_form.py).
    property_type: Optional[str] = None      # Apartment, Villa, Plot, Penthouse ...
    category: Optional[str] = None           # affordable | mid | luxury
    usps: list[str] = Field(default_factory=list)          # key selling points
    target_audience: Optional[str] = None
    cta: Optional[str] = None                # "Book a site visit"
    tagline: Optional[str] = None
    configuration: Optional[str] = None  # e.g. "3 BHK"
    area: Optional[str] = None  # e.g. "1450 sq ft"
    price: Optional[str] = None
    amenities: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    contact: Optional[PropertyContact] = None
    # RERA registration number. The evaluation singled out the original's
    # RERA and banking details as a key trust signal the generated
    # brochure lacked.
    rera_number: Optional[str] = None

    # --- the structured modules -------------------------------------------
    # Every one of these is optional and every one of them is a gate: the
    # module it feeds is composed only when the list is non-empty. A
    # property that supplies nothing here produces exactly the brochure it
    # produced before these fields existed.
    unit_types: list[UnitType] = Field(default_factory=list)
    landmarks: list[Landmark] = Field(default_factory=list)
    amenity_groups: list[AmenityGroup] = Field(default_factory=list)
    developer_stats: list[DeveloperStat] = Field(default_factory=list)
    booking_steps: list[BookingStep] = Field(default_factory=list)
    testimonials: list[Testimonial] = Field(default_factory=list)
    spec_groups: list[SpecGroup] = Field(default_factory=list)
    # Shown on the snapshot module beside the configuration and the area.
    possession: Optional[str] = None
    project_size: Optional[str] = None       # "12 acres", "4 towers, 560 homes"
    approvals: list[str] = Field(default_factory=list)   # "HDFC", "SBI", "ICICI"
    # For the location module's map. A property with no coordinates gets
    # the distance chips and no map, which is still the useful half.
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # A KEY PLAN: the small drawing that shows where a unit sits in the
    # building or the site, by filename among the uploaded images.
    #
    # It is a field rather than something inferred because the one thing
    # that makes a key plan useful — the mark showing WHICH unit — is not
    # recoverable from a site plan JPEG, and a key plan without that mark
    # is the unlabelled thumbnail the review complained about in the
    # reference brochures. Supply one that carries the mark, or the plan
    # pages carry none.
    key_plan: Optional[str] = None
    # The building from the ground up, for the stack diagram.
    floor_stack: list[FloorBand] = Field(default_factory=list)
    highlights: list[Highlight] = Field(default_factory=list)
    green_stats: list[GreenStat] = Field(default_factory=list)
    green_certification: Optional[str] = None
    # Free-form paragraphs about the developer, printed on the developer
    # page beside its figures.
    developer_story: Optional[str] = None
    # The developer's own brand accent ("#E31837"). Replaces the category
    # palette's accent only where it passes the contrast checks.
    brand_color: Optional[str] = None


# ---------------------------------------------------------------------------
# Property Intelligence Agent output
# ---------------------------------------------------------------------------

class PropertyKnowledge(BaseModel):
    facts: dict[str, str] = Field(default_factory=dict)
    selling_points: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    # Materials, finishes, and fittings — only when a source states them.
    # The evaluation noted even the original brochure lacked these; when
    # documents provide them, they get a dedicated page.
    specifications: list[str] = Field(default_factory=list)
    # Registration details found in the input/documents (never invented).
    rera_number: str = ""
    legal_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Vision Agent output
# ---------------------------------------------------------------------------

class ImageAsset(BaseModel):
    """A single image file discovered during ingestion, or generated."""
    image_id: str
    filename: str
    path: str  # absolute path on disk
    # What the person uploaded it AS, when the form told us: photo,
    # floorplan, masterplan, map or logo. Empty means "work it out" — the
    # Vision Agent's category decides. An upload slot is a stronger signal
    # than any classifier: a floor plan uploaded as a floor plan is one.
    role: str = ""
    # What a photograph shows, when the person (or a caption file) said so:
    # "pool", "bedroom", "exterior", "aerial" ... — a topic key from
    # src/agents/brochure/topics.py. Stronger than any filename guess and
    # used ahead of the Vision Agent's reading when present. Recorded in
    # images/roles.json as "photo:<subject>".
    subject: str = ""
    # True for AI-generated visuals. These are never used as the cover and
    # never as a depiction of the property itself — that is a RERA
    # truthful-advertising rule and it is enforced at selection time, in
    # visual_enrichment.py, not by printing a disclaimer over the picture.
    # Templates carry no "artistic impression" mark: it was being stamped
    # on the person's own photographs as well, where it is simply untrue.
    # A disclaimer that belongs on a page comes from the source content.
    is_ai_generated: bool = False


class ImageAnalysis(BaseModel):
    image_id: str
    category: ImageCategory
    quality: float = Field(ge=0, le=10)
    composition: float = Field(ge=0, le=10)
    hero_score: float = Field(ge=0, le=10)
    # Short freeform descriptor of the SPECIFIC subject/angle, e.g.
    # "entrance_gate", "tower_exterior_wide", "living_room_dusk" — not a
    # fixed enum, since subjects vary a lot property to property. Used by
    # the Image Selection Agent to avoid picking several near-duplicate
    # shots (e.g. two similar entrance renders) into the same brochure.
    scene_tag: str = ""
    notes: str = ""
    # What the Vision Agent saw (see the vision_classify_batch prompt):
    # grounded descriptions of what is visibly IN the photo, never a
    # marketing claim. The planner uses them to choose a page's subject;
    # the copywriter never prints them (a brochure sells the benefit, it
    # does not describe the picture).
    suggested_title: str = ""     # e.g. "Living Room", "Grand Facade"
    suggested_caption: str = ""   # a rich, multi-sentence grounded description (see vision_classify_batch prompt)
    # How the image was MADE, not what it shows. Generated images are
    # prompted in the same medium, so an AI lifestyle photo doesn't sit
    # among architectural CGI renders looking like stock.
    medium: str = ""          # photograph | cgi_render | drawing
    visible_text: str = ""        # text legible IN the image itself (e.g. a floor plan's printed unit number/area) — transcription, not invention; empty if none/illegible


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Layer 2 (Content Quality) for brochures. 3 dimensions — see the
    "why these" note in src/evaluation/scoring.py. A separate, purely
    technical "did the PDF actually render" check exists too, but it's a
    rendering-pipeline reliability guard, not a scored quality dimension —
    see is_pdf_render_valid() in src/rendering/renderer.py and
    workflows/brochure.py's node_render(). A generation whose PDF didn't
    render validly fails outright (status="failed") rather than reaching
    the Critic with a broken file. Each field's `description` is the
    single source of truth for its one-line "what it measures"
    explanation, surfaced in the Streamlit UI as a tooltip rather than
    duplicated across the codebase."""

    factual_consistency: float = Field(
        default=0,
        description="Does the copy avoid CONTRADICTING the known property facts "
        "(not: is every sentence pre-declared in property.json — verbatim "
        "user-provided content is graded on consistency, not exhaustive traceability). "
        "Scored by the VLM/LLM judge reading the copy against the fact ledger.",
    )
    factual_consistency_reasoning: str = Field(
        default="",
        description="The judge's one-to-two-sentence, evidence-citing explanation for the "
        "factual_consistency score — which facts it checked and what it found (or didn't "
        "find) contradicted. Internal/debugging use (e.g. MLflow), not necessarily shown "
        "prominently in the UI — this is a rationale for the score, not a per-item issues "
        "list surfaced to flag the person's own content as wrong.",
    )
    visual_design_quality: float = Field(
        default=0,
        description="Photo quality, layout hierarchy/spacing, and consistent use of "
        "the chosen palette/fonts, judged directly from the rendered screenshot. "
        "Scored by the VLM judge looking at the actual rendered page.",
    )
    visual_design_quality_reasoning: str = Field(default="", description="See factual_consistency_reasoning.")
    content_quality: float = Field(
        default=0,
        description="Writing clarity, flow, and persuasiveness, appropriate for a "
        "premium real-estate audience. Scored by the LLM judge reading the copy text.",
    )
    content_quality_reasoning: str = Field(default="", description="See factual_consistency_reasoning.")
    layout_integrity: Optional[float] = Field(
        default=None,
        description="Deterministic: measured in the real renderer after layout (text running past "
        "a page edge, or clipped). None when the layout could not be measured.",
    )
    # Plain-language problems found by deterministic checks (layout audit).
    issues: list[str] = Field(default_factory=list)
    quality_score: float = 0
    gate_result: QualityGateResult = QualityGateResult.FAIL
    # --- measured on the rendered brochure (src/agents/brochure/qa.py) ---
    # "model" when a vision/language model judged the brochure; "measured"
    # when no model was available and the scores come from the rendered-page
    # measurements alone. Shown with the scores, so a reader knows which.
    judge: str = "model"
    whitespace_mean: Optional[float] = None       # share of a page's frame left empty
    pages_over_whitespace: int = 0               # pages above the 20% limit
    image_integrity: Optional[float] = None      # 100 = every image whole, once
    checks: dict[str, str] = Field(default_factory=dict)   # check -> "pass" | plain-language result


# ---------------------------------------------------------------------------
# Three-layer evaluation, matching the platform's original design:
#   1. Workflow/Agent Evaluation -> "did every agent run successfully"
#   2. Content Quality Evaluation -> EvaluationResult above (the Critic)
#   3. Human Review -> a person's sign-off, layered on top of both
# ---------------------------------------------------------------------------

class AgentStepResult(BaseModel):
    """One LangGraph node's outcome for a single generation run."""
    node: str
    status: str = "success"  # "success" | "failed"
    latency_ms: float = 0
    error: Optional[str] = None


class WorkflowEvaluation(BaseModel):
    """Layer 1: did every agent/step in the pipeline actually run
    successfully? Independent of whether the OUTPUT was any good — that's
    layer 2 (EvaluationResult)."""
    steps: list[AgentStepResult] = Field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(s.status == "success" for s in self.steps)

    @property
    def failed_steps(self) -> list[str]:
        return [s.node for s in self.steps if s.status != "success"]


class HumanReview(BaseModel):
    """Layer 3: a person's sign-off on the finished brochure, recorded
    after the fact — separate from both the automated workflow/content
    checks above."""
    status: str = "pending"  # "pending" | "approved" | "revision_requested"
    comment: str = ""
    reviewed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Generation record (what gets persisted / returned by the API)
# ---------------------------------------------------------------------------

class GenerationRecord(BaseModel):
    generation_id: str
    property_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending | running | completed | failed
    pdf_path: Optional[str] = None
    html_path: Optional[str] = None
    screenshot_path: Optional[str] = None
    workflow_evaluation: Optional[WorkflowEvaluation] = None
    evaluation: Optional[EvaluationResult] = None
    human_review: HumanReview = Field(default_factory=HumanReview)
    error: Optional[str] = None
    # Shown in the UI for the current run: document-reading notes (e.g. a
    # scanned PDF with no text) and image-generation notes.
    notes: list[str] = Field(default_factory=list)
    # Defects that failed preflight. These are not notes: a note is
    # something a reader might like to know, a defect is a reason the
    # brochure should not go out — type over type, text off the page,
    # text under the footer, a clipped URL, type too small to read. The
    # review found three of them in production because the workflow
    # logged them as warnings and shipped anyway.
    defects: list[str] = Field(default_factory=list)
    ai_image_paths: list[str] = Field(default_factory=list)
    # What the brochure is: its template ("Luxury - Portrait"), and what each
    # agent decided — shown on the Agents tab, kept out of the main view.
    template: str = ""
    page_count: int = 0
    agent_reports: dict[str, list[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Blog Generation (second solution). Reuses AgentStepResult/WorkflowEvaluation/
# HumanReview/QualityGateResult from above — those are already generic, not
# brochure-specific.
# ---------------------------------------------------------------------------

class BlogImageAsset(BaseModel):
    """A single image uploaded for a blog post."""
    image_id: str
    filename: str
    path: str


class BlogInput(BaseModel):
    """Matches data/blogs/<id>/blog_input.json — everything the person
    provides about the post they want written.

    ON DISK: one UTF-8 JSON object at `data/blogs/<blog_id>/blog_input.json`
    (optional images in `data/blogs/<blog_id>/images/*.jpg|png|webp`). Only
    `blog_id` and `topic` are REQUIRED; every other field is optional and
    may be omitted, empty, or null-equivalent ("" / []).

    Field reference (all strings unless noted):
      blog_id            REQUIRED. Folder-safe id (see src/core/ids.slugify).
      topic              REQUIRED. What the post is about, as a reader would
                         search it: "Is Shela, Ahmedabad a good place to buy?"
      sections           list[str] of ids from configs/blog/sections.yaml.
      custom_sections    list[str], extra sections in the author's own words.
      local_facts        list[str], facts the author vouches for — the ONLY
                         source of numbers besides research. One per item.
                         A source can be appended: "Metro 1.2 km away
                         (source: GMRC)".
      key_points         list[str], points the post must make.
      tone               e.g. "professional", "conversational".
      target_word_count  int, e.g. 1200.
      audience           who it is for: "first-time buyers", "NRI investors".
      location           the place the post is about: "Shela, Ahmedabad".
                         Optional — derived from the topic when blank.
      primary_keyword    the search phrase to rank for. Derived when blank.
      secondary_keywords list[str].
      article_type       force an archetype instead of auto-detection: one of
                         configs/blog/archetypes.yaml ids (locality_guide,
                         buyers_guide, comparison, investment_analysis,
                         explainer, how_to, market_update).
      brand_name         the publisher / developer the post is written for.
      brand_description  one line of positioning, used sparingly in the CTA.
      project_name       a specific project the post may name.
      author_name        byline. Defaults to brand_name, then "Editorial team".
      cta_text           what readers should do next.
      cta_url / cta_phone / cta_email
                         REAL contact details only. Anything that looks like
                         a placeholder (example.com, XXXX, "https://...") is
                         ignored — the CTA is then written without contacts.
      reference_links    list[str] of URLs the author vouches for, optionally
                         "Title | https://...". Cited in the references.
    """
    blog_id: str
    topic: str
    # Section/perspective ids chosen from configs/blog/sections.yaml
    # (e.g. "strengths", "investment_perspective"). Replaces the old
    # free-text brief as the primary way to shape the post.
    sections: list[str] = Field(default_factory=list)
    # Sections the catalogue does not have a name for, described by the
    # person in their own words: "Why Ahmedabad is becoming a major real
    # estate investment hub", "Comparison with nearby developments".
    #
    # Kept separate from `sections` rather than mixed in, because the two
    # are checked differently. A catalogue id has a known label the
    # coverage check can look for in the finished post; a sentence someone
    # typed does not, so it is matched on its own distinctive words
    # instead (see evaluation/blog_deterministic.uncovered_sections).
    custom_sections: list[str] = Field(default_factory=list)
    # Optional facts the person can vouch for (prices, distances, project
    # names). The writer may use numbers ONLY from here — see the
    # "no invented figures" rule in configs/blog/prompts.yaml. Without
    # this, asking for "hyper-local data" just means inventing it.
    local_facts: list[str] = Field(default_factory=list)
    # Call to action. The evaluation found neither the original nor the
    # generated blog had one; every post now ends with one.
    cta_text: str = ""
    cta_url: str = ""
    cta_phone: str = ""
    # Kept for backward compatibility with existing blog_input.json files.
    key_points: list[str] = Field(default_factory=list)
    tone: str = ""  # e.g. "professional", "conversational", "persuasive"
    target_word_count: Optional[int] = None
    # --- Optional brief fields (added; all default to empty) -------------
    audience: str = ""
    location: str = ""
    primary_keyword: str = ""
    secondary_keywords: list[str] = Field(default_factory=list)
    article_type: str = ""
    brand_name: str = ""
    brand_description: str = ""
    project_name: str = ""
    author_name: str = ""
    cta_email: str = ""
    reference_links: list[str] = Field(default_factory=list)


class BlogReference(BaseModel):
    """One source a post cites or recommends for verification. Only ever
    built from the brief, the research pack, or the curated authority
    registry (configs/blog/editorial.yaml) — never from model output."""
    title: str
    url: str = ""
    kind: str = "research"   # official | brief | research | news | background
    note: str = ""


class NearbyPlace(BaseModel):
    """One real place near the subject, from OpenStreetMap."""
    name: str
    kind: str = ""
    distance_km: float = 0.0


class JourneyTime(BaseModel):
    """A road journey, measured rather than guessed.

    `free_flow_minutes` is what the router computes with no traffic model
    at all. Named for what it is so it cannot be quoted as a commute —
    the one number a reader would actually plan around."""
    destination: str
    distance_km: float = 0.0
    free_flow_minutes: int = 0


class NewsItem(BaseModel):
    """A headline, with where and when it came from.

    Evidence that something was ANNOUNCED, which is not the same as
    evidence that it happened — the writer is told to treat these as
    signals worth mentioning, attributed, not as facts to assert."""
    title: str
    source: str = ""
    published: str = ""
    url: str = ""


class ResearchPack(BaseModel):
    """What the research step found, and the writer's only source of
    outside fact.

    The point of writing this down as a structure — rather than letting
    the writer search while it writes — is that it separates two
    questions that otherwise get tangled: did the research find anything,
    and did the writer use it well. With research as its own recorded
    step, a weak post can be traced to one or the other. With the writer
    browsing mid-sentence it cannot, and the run is not reproducible
    either.

    `notes` is what could not be researched, in plain language, and it
    reaches the person as well as the model: a section that is thin
    because Overpass was down should say so rather than read as
    carelessness."""
    subject: str = ""                 # what the post is about
    place_query: str = ""             # what we asked the geocoder for
    place_name: str = ""              # what it actually resolved to
    latitude: float = 0.0
    longitude: float = 0.0
    located: bool = False

    background: str = ""              # neutral summary of the area
    background_source: str = ""

    nearby: list[NearbyPlace] = Field(default_factory=list)
    journeys: list[JourneyTime] = Field(default_factory=list)
    news: list[NewsItem] = Field(default_factory=list)

    # What pages already ranking for this topic cover. Used to find the
    # angle nobody has taken, NOT as source material — these are
    # competitors' claims, unverified.
    competitor_topics: list[str] = Field(default_factory=list)
    # Question-shaped searches real people make. They make good headings,
    # because a heading someone already typed is a heading that gets read.
    reader_questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    # Search results from authoritative domains (government, RERA, metro
    # and municipal bodies), with their snippets — the only web results
    # treated as evidence rather than as competitor shape.
    official_sources: list[BlogReference] = Field(default_factory=list)
    # The article archetype the research depth was chosen for.
    archetype: str = ""

    @property
    def is_empty(self) -> bool:
        return not any((self.background, self.nearby, self.journeys, self.news,
                        self.competitor_topics, self.reader_questions, self.official_sources))


class FactCheckResult(BaseModel):
    """Which claims in the draft are actually supported.

    Only claims that CAN be checked are checked: numbers, distances,
    dates, named places. An opinion is not a fact-checking failure and
    flagging it as one would train the writer out of having a view."""
    verified: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported


class BlogSectionPlan(BaseModel):
    """One planned section: what it is for and what SHAPE it should take.
    The shape is planned, not left to the writer, because walls of prose
    where a table or checklist belonged was a recurring evaluation finding."""
    heading: str
    purpose: str = ""
    format: str = "prose"     # prose | list | table | checklist | steps | faq | pros_cons
    covers: list[str] = Field(default_factory=list)   # requested section ids / custom texts
    slot: str = ""            # archetype slot id (configs/blog/archetypes.yaml)


class BlogOutline(BaseModel):
    """Planner Agent output: a lightweight skeleton the Writer fleshes out."""
    working_title: str
    section_headings: list[str] = Field(default_factory=list)
    section_briefs: dict[str, str] = Field(default_factory=dict)  # heading -> 1-line brief
    # Added: the article archetype and the reasoning that shapes the post.
    archetype: str = ""
    angle: str = ""
    audience: str = ""
    search_intent: str = ""
    primary_keyword: str = ""
    secondary_keywords: list[str] = Field(default_factory=list)
    sections: list[BlogSectionPlan] = Field(default_factory=list)
    planner_mode: str = ""    # llm | fallback


class BlogSection(BaseModel):
    heading: str
    body_markdown: str
    image_id: Optional[str] = None  # optional image placed in this section
    format: str = ""
    slot: str = ""


class BlogFAQ(BaseModel):
    question: str
    answer_markdown: str


class BlogContent(BaseModel):
    """Writer Agent output, refined by the Editor and the SEO Agent."""
    title: str
    meta_description: str = ""
    seo_keywords: list[str] = Field(default_factory=list)
    intro_markdown: str = ""
    sections: list[BlogSection] = Field(default_factory=list)
    conclusion_markdown: str = ""
    cover_image_id: Optional[str] = None
    # Closing call to action, rendered as its own designed block.
    cta_heading: str = ""
    cta_markdown: str = ""
    # Set when the cover is an AI-generated hero, so it is labelled.
    hero_is_ai_generated: bool = False
    hero_caption: str = ""
    # --- Added ------------------------------------------------------------
    conclusion_heading: str = ""
    key_takeaways: list[str] = Field(default_factory=list)
    faqs: list[BlogFAQ] = Field(default_factory=list)
    # Built deterministically from the brief/research/authority registry.
    references: list[BlogReference] = Field(default_factory=list)
    meta_title: str = ""
    slug: str = ""
    tags: list[str] = Field(default_factory=list)
    primary_keyword: str = ""
    secondary_keywords: list[str] = Field(default_factory=list)
    archetype: str = ""
    author: str = ""
    published_on: str = ""
    writer_mode: str = ""     # llm | fallback


class BlogDimensionScore(BaseModel):
    """One rubric dimension (configs/blog/evaluation.yaml), with the
    measurements behind it and what to change, in words a writer can act on."""
    name: str
    label: str = ""
    score: float = 0
    deterministic_score: float = 0
    llm_score: Optional[float] = None
    weight: float = 0
    threshold: float = 0
    passed: bool = True
    # the score's reason in one line, shown beside it
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class BlogValidationIssue(BaseModel):
    code: str
    severity: str = "warning"   # error | warning | info
    message: str
    location: str = ""


class BlogValidationReport(BaseModel):
    """What the Editor (normaliser + validator) changed, and what it could
    not fix mechanically."""
    fixes: list[str] = Field(default_factory=list)
    issues: list[BlogValidationIssue] = Field(default_factory=list)
    removed_claims: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


class BlogRevisionRound(BaseModel):
    round: int
    quality_score: float = 0
    gate_result: str = ""
    failing_dimensions: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    writer_mode: str = ""
    decision: str = ""          # revise | done | skipped: <reason>


class BlogEvaluationResult(BaseModel):
    """Layer 2 (Content Quality) for blog posts.

    `quality_score` is the weighted sum of the five rubric `dimensions`
    (content quality, relevance, structure & readability, audience
    engagement, SEO optimisation; configs/blog/evaluation.yaml) — see
    src/evaluation/blog_rubric.py. The four legacy top-level scores are
    still populated for older UI code and stored rows."""

    grounding: float = Field(
        default=0,
        description="Sticks strictly to the topic and key points provided — no "
        "invented claims, no off-topic wandering. Scored by the LLM judge reading "
        "the post against the original brief.",
    )
    writing_quality: float = Field(
        default=0,
        description="Clarity, flow, logical structure, and whether it actually reads "
        "in the requested tone, including a strong intro/conclusion. Scored by the "
        "LLM judge reading the full post.",
    )
    structural_completeness: float = Field(
        default=0,
        description="Title, meta description, and keywords present and well-formed; "
        "adequate section count; word count near the target; no empty sections. "
        "Scored automatically from the data structure — not an LLM judgment.",
    )
    editorial_integrity: float = Field(
        default=0,
        description="Deterministic checks for the failure modes the original-vs-generated "
        "evaluation found: every selected section covered, a call to action present, no "
        "keyword stuffing, no figures that are not in the provided facts, and no stock "
        "AI phrasing.",
    )
    # Plain-language reasons behind editorial_integrity, shown in the UI.
    issues: list[str] = Field(default_factory=list)
    quality_score: float = 0
    gate_result: QualityGateResult = QualityGateResult.FAIL
    # --- Added ------------------------------------------------------------
    dimensions: list[BlogDimensionScore] = Field(default_factory=list)
    # Raw deterministic measurements (paragraph lengths, keyword density...).
    metrics: dict[str, float] = Field(default_factory=dict)
    # Dimension-level revision instructions sent back to the writer.
    feedback: list[str] = Field(default_factory=list)
    critic_mode: str = ""       # "llm+deterministic" | "deterministic"


class BlogGenerationRecord(BaseModel):
    generation_id: str
    blog_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending | running | completed | failed
    markdown_path: Optional[str] = None
    html_path: Optional[str] = None
    pdf_path: Optional[str] = None
    hero_image_path: Optional[str] = None
    image_notes: list[str] = Field(default_factory=list)
    # What research could not find, in plain language. Shown with the post
    # rather than buried in a log: a section that is thin because a source
    # was unreachable should say so rather than read as carelessness.
    research_notes: list[str] = Field(default_factory=list)
    # Where the outside facts came from, so a reader (or an editor) can
    # check them.
    sources: list[str] = Field(default_factory=list)
    workflow_evaluation: Optional[WorkflowEvaluation] = None
    evaluation: Optional[BlogEvaluationResult] = None
    human_review: HumanReview = Field(default_factory=HumanReview)
    error: Optional[str] = None
    # --- Added ------------------------------------------------------------
    title: str = ""
    slug: str = ""
    meta_title: str = ""
    meta_description: str = ""
    archetype: str = ""
    word_count: int = 0
    planner_mode: str = ""      # llm | fallback
    writer_mode: str = ""       # llm | fallback
    revision_history: list[BlogRevisionRound] = Field(default_factory=list)
    validation: Optional[BlogValidationReport] = None
    # Findings from the rendered PDF (page count, overflow, leaked markup).
    render_report: list[str] = Field(default_factory=list)
    pdf_pages: int = 0


# ---------------------------------------------------------------------------
# AI image generation (brochure Image Gap Analysis + blog Visual Director)
# ---------------------------------------------------------------------------

# Kinds of image the system is ALLOWED to generate. Deliberately excludes
# anything that would depict the actual property (its facade, towers,
# specific rooms, floor/site plans): a generated render presented as the
# real building misleads buyers, and in India would likely breach RERA's
# rules on truthful advertising. Generated images set mood and context;
# the uploaded photos remain the only depiction of the property itself.
ALLOWED_GENERATED_KINDS = (
    "lifestyle",         # people-free or distant-figure scenes of how life feels here
    "amenity_concept",   # an amenity the property lists but has no photo of, shown generically
    "neighbourhood",     # the city/area mood (skyline at dusk, green boulevard) — not the building
    "design_element",    # abstract architectural texture / pattern used as a design accent
    "blog_hero",         # blog title image
)


class ImageBrief(BaseModel):
    """What to generate and WHY — produced before any image exists, so
    every generated visual has a stated purpose and a destination."""
    kind: str                       # one of ALLOWED_GENERATED_KINDS
    purpose: str                    # the design/narrative gap it fills
    subject: str                    # concrete, depictable subject
    aspect: str = "landscape"       # landscape | portrait | wide | square
    placement: str = ""             # where it's meant to go, e.g. "lifestyle section"
    caption: str = ""               # short evocative caption for the page


class GeneratedVisual(BaseModel):
    brief: ImageBrief
    path: str
    model_id: str
    prompt: str
    width: int
    height: int
    is_ai_generated: bool = True
    from_cache: bool = False
