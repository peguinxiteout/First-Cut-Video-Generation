"""
The Video tab of the Streamlit app (streamlit_app.py calls render_video_tab()).

    write a prompt, pick a property -> send            (Plan off: the video is made without a stop)
    or, with Plan on: fill the brief form -> send -> read the storyboard -> Approve and produce
    -> watch it -> the video

Everything the page does goes through src/workflows/video_jobs.py and the
video repository: it starts a job, polls the database and never talks to the
process. A video is named in the page address (?video=vid_...), so a refresh or
a new session lands on the same one. The pieces that decide something (the
form-to-options mapping, the step list, the elapsed time, the log tail) are
plain functions so they can be tested without a page.

Options come from the code, not from lists kept here: durations, aspects and
objectives from data/video_schemas.py, tones from script_templates.yaml, voices
from speech.yaml, motion models from the selector (providers.yaml).
"""
from __future__ import annotations

import io
import json
import logging
import re
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

import streamlit as st
from pydantic import ValidationError

from src.agents.video import brief as brief_agent
from src.agents.video import layout as layout_agent
from src.core.config import load_yaml_config, settings
from src.core.exceptions import IngestionError, JobBusyError, JobError
from src.data import video_repository as repository
from src.data.ingestion import discover_images
from src.data.video_schemas import (
    ACTIVE_STATUSES, STATUS_PLANNED, VIDEO_OBJECTIVES, Shot, VideoGenerationRecord, VideoInput, VideoRequest,
    VideoSummary,
)
from src.models.video_gen import selector
from src.workflows import video_jobs

logger = logging.getLogger(__name__)

# The ten steps in the order they run (workflows/video.py _STEP_NAMES), as a person would name them.
STEP_TITLES = {
    "brief": "Read the brief",
    "fact_ledger": "Collect the facts",
    "asset_audit": "Look at the photographs",
    "script": "Write the script",
    "storyboard": "Plan the shots",
    "voice_music": "Record the voice and choose the music",
    "motion": "Move the pictures",
    "overlays": "Draw the text and captions",
    "compose": "Put the video together",
    "gate": "Check the result",
}
ACTIVE = (*ACTIVE_STATUSES, "running")
POLL_SECONDS = 2
LOG_LINES = 20
VIDEO_QUERY = "video"
NO_HERO = ""


# ---------------------------------------------------------------------------
# The form: values in, options out
# ---------------------------------------------------------------------------

def form_key(property_id: str, field: str) -> str:
    """A widget key per property, so each property keeps its own entries."""
    return f"vt_{property_id}_{field}"


def properties_dir() -> Path:
    return settings.data_dir_resolved / "properties"


def property_choices() -> list[tuple[str, str]]:
    """(folder name, label) of every folder in data/properties/ that has a property.json;
    the label is the property's name when it can be read."""
    root = properties_dir()
    if not root.is_dir():
        return []
    choices = []
    for folder in sorted(root.iterdir(), key=lambda d: d.name.lower()):
        file = folder / "property.json"
        if not folder.is_dir() or not file.is_file():
            continue
        try:
            name = str(json.loads(file.read_text(encoding="utf-8")).get("name") or "").strip()
        except (OSError, ValueError, AttributeError):
            name = ""
        choices.append((folder.name, f"{name} ({folder.name})" if name and name != folder.name else folder.name))
    return choices


def tone_choices() -> list[str]:
    """"" (the category's own tone) and every tone the script templates have."""
    return ["", *load_yaml_config("video/script_templates.yaml")["tones"]]


def voice_choices() -> dict[str, str]:
    """voice key -> label, from speech.yaml."""
    cfg = load_yaml_config("video/speech.yaml")
    return {key: v.get("label", key) for key, v in cfg["voices"].items()}


def default_voice() -> str:
    return load_yaml_config("video/speech.yaml")["default_voice"]


def hero_choices(folder: Path) -> list[str]:
    """File names of the property's photographs a video could open on."""
    try:
        images = discover_images(folder)
    except IngestionError:
        return []
    return [i.filename for i in images if i.role in ("", "photo")]


def _lines(text: str) -> list[str]:
    return [line.strip(" •-\t") for line in (text or "").splitlines() if line.strip(" •-\t")]


def parse_pronunciations(text: str) -> tuple[dict[str, str], list[str]]:
    """'word=respelling' per line -> ({word: respelling}, the lines that were not that)."""
    pairs, bad = {}, []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        word, sep, spoken = line.partition("=")
        if sep and word.strip() and spoken.strip():
            pairs[word.strip()] = spoken.strip()
        else:
            bad.append(line.strip())
    return pairs, bad


def pronunciations_text(pairs: dict[str, str]) -> str:
    return "\n".join(f"{word}={spoken}" for word, spoken in pairs.items())


def form_defaults(folder: Path) -> tuple[dict, list[str]]:
    """The form's starting values: the property's video_input.json over the defaults of VideoInput.
    Also the notes to show (the file could not be read)."""
    notes: list[str] = []
    try:
        given = brief_agent.load_video_input(folder)
    except IngestionError as exc:
        notes.append(f"{exc} The form starts from the defaults.")
        given = VideoInput()
    return {
        "duration_s": given.duration_s, "aspect": given.aspect, "audience": given.audience,
        "objective": given.objective, "highlights": "\n".join(given.highlights), "tone": given.tone,
        "voice": given.voice or default_voice(), "music_mood": given.music_mood, "cta": given.cta,
        "creative_direction": given.creative_direction, "hero_image": given.hero_image,
        "burn_captions": given.burn_captions, "grade": given.grade, "allow_generated": given.allow_generated,
        "pronunciations": pronunciations_text(given.pronunciations), "model": given.model,
    }, notes


def build_overrides(values: dict) -> dict:
    """The form's values as the options of a video. Every option is sent, an empty one as empty,
    so clearing a field here clears what video_input.json says. `highlights` and `pronunciations`
    are the text areas' text."""
    base = VideoInput()

    def text(field: str) -> str:
        return str(values.get(field, getattr(base, field)) or "").strip()

    return {
        "duration_s": int(values.get("duration_s", base.duration_s)),
        "aspect": text("aspect") or base.aspect,
        "audience": text("audience"),
        "objective": text("objective") or base.objective,
        "highlights": list(dict.fromkeys(_lines(values.get("highlights", "")))),
        "tone": text("tone"),
        "voice": text("voice"),
        "music_mood": text("music_mood"),
        "model": text("model") or base.model,
        "cta": text("cta"),
        "creative_direction": text("creative_direction"),
        "allow_generated": bool(values.get("allow_generated", base.allow_generated)),
        "hero_image": text("hero_image"),
        "burn_captions": bool(values.get("burn_captions", base.burn_captions)),
        "grade": bool(values.get("grade", base.grade)),
        "pronunciations": parse_pronunciations(values.get("pronunciations", ""))[0],
    }


def merge_defaults(folder: Path, overrides: dict) -> Path:
    """Writes `overrides` into the property's video_input.json over what is there, keeping the keys
    the form does not know. Raises IngestionError, in words, if the file or the result is not valid."""
    path = folder / brief_agent.VIDEO_INPUT_FILE
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IngestionError(f"{brief_agent.VIDEO_INPUT_FILE} is not valid JSON ({exc}); nothing was saved.") from exc
        if not isinstance(existing, dict):
            raise IngestionError(f"{brief_agent.VIDEO_INPUT_FILE} must be a JSON object; nothing was saved.")
    merged = {**existing, **overrides}
    try:
        VideoInput.model_validate(merged)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise IngestionError(f"The video options are not valid: {problems}. Nothing was saved.") from exc
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# -- the motion model selector --------------------------------------------------------------------

def money(usd: float) -> str:
    """Dollars to the cent, a half rounded up (0.075 is 0.08; Python's own format gives 0.07)."""
    return str(Decimal(str(usd)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def md(text: str) -> str:
    """Text for a caption or a label: a dollar sign is escaped, or two of them in one line are read as maths."""
    return text.replace("$", "\\$")


def cost_text(option: selector.ProviderOption) -> str:
    if option.est_cost_usd is None:
        return "credit-based" if option.cost_basis == "credits" else "price not set"
    if option.est_cost_usd == 0:
        return "free"
    return f"{'from ' if option.price_from else ''}about ${money(option.est_cost_usd)}"


def why_not(option: selector.ProviderOption) -> str:
    if option.reason == "adapter not installed":
        return "not available yet"
    return option.reason


def unavailable_row(option: selector.ProviderOption) -> str:
    """One line for a model that cannot be chosen: what it is, what it would cost, what the free
    tier gives, and the flags that matter (a watermark, a price nobody has checked). The vendor is
    named only when the label does not already."""
    vendor = f"via {option.vendor}" if option.vendor and option.vendor.lower() not in option.label.lower() else ""
    flags = ("free tier watermarked" if option.watermark_free_tier else "",
             "" if option.verified else "price unverified")
    parts = (option.label, vendor, why_not(option), cost_text(option), option.free_tier, *flags)
    return " · ".join(part for part in parts if part)


def option_label(option: selector.ProviderOption) -> str:
    """One line for the selector: the model, what it will use (Auto), and what it costs."""
    note = option.reason if option.available and option.reason else ""
    return " · ".join(part for part in (option.label, note, cost_text(option)) if part)


# ---------------------------------------------------------------------------
# Progress and the plan
# ---------------------------------------------------------------------------

def is_active(summary: Optional[VideoSummary]) -> bool:
    return summary is not None and summary.status in ACTIVE


def step_states(summary: VideoSummary) -> list[tuple[str, str, str]]:
    """(step, title, state) for the ten steps; state is done, current, failed or waiting.
    Steps run in order and `steps_done` counts the ones that succeeded, so the next one is
    the one running (or the one that failed)."""
    rows = []
    for index, (step, title) in enumerate(STEP_TITLES.items()):
        if index < summary.steps_done:
            state = "done"
        elif index == summary.steps_done and summary.status in ACTIVE:
            state = "current"
        elif index == summary.steps_done and summary.status == "failed" and summary.current_node:
            state = "failed"
        else:
            state = "waiting"
        rows.append((step, title, state))
    return rows


def elapsed_seconds(summary: VideoSummary, now: Optional[datetime] = None) -> float:
    start = summary.started_at or summary.created_at
    end = (now or datetime.now(timezone.utc)) if summary.status in ACTIVE else (summary.updated_at or start)
    return max(0.0, (end - start).total_seconds())


def clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


def log_tail(video_id: str, lines: int = LOG_LINES) -> str:
    """The last lines of the job's log, or "" when there is none."""
    try:
        path = repository.video_output_dir(video_id) / "work" / video_jobs.LOG_NAME
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    return "\n".join(text.splitlines()[-lines:])


def expected_layout(shot: Shot, brief) -> str:
    """fill or fit for a shot, as the clip step will decide it for this frame size (it is written
    on the shot only once the clip is made)."""
    if shot.layout:
        return shot.layout
    if shot.kind != "real" or not shot.source_path or brief is None:
        return ""
    try:
        from PIL import Image

        with Image.open(shot.source_path) as picture:
            width, height = picture.size
        return layout_agent.choose(width, height, brief.width, brief.height, shot.focal, shot.camera).layout
    except Exception:  # noqa: BLE001 - a missing picture only means no layout is shown
        return ""


LAYOUT_WORDS = {"fill": "fills the frame", "fit": "whole photograph on a blurred background"}


# ---------------------------------------------------------------------------
# Page pieces
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _thumbnail(path: str, modified: float) -> bytes:
    from PIL import Image

    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((420, 420))
        buffer = io.BytesIO()
        picture.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


def _show_thumbnail(path: str) -> None:
    try:
        st.image(_thumbnail(path, Path(path).stat().st_mtime), width=150)
    except Exception:  # noqa: BLE001 - a picture that cannot be read is left out
        st.caption("(picture not found)")


def _goto(video_id: Optional[str]) -> None:
    """Names the video in the page address (None: no video, the new-video form). Also the callback of the
    buttons that open one: a callback runs even when the page that drew the button is not drawn again."""
    if video_id:
        st.query_params[VIDEO_QUERY] = video_id
    elif VIDEO_QUERY in st.query_params:
        del st.query_params[VIDEO_QUERY]


def open_video(video_id: Optional[str]) -> None:
    _goto(video_id)
    st.rerun()


def busy_notice(error: JobBusyError) -> None:
    """Another video is being made: say which, where it is, and offer to open it."""
    running = repository.list_by_status(*ACTIVE, limit=1)
    if not running:
        st.warning(str(error))
        return
    job = running[0]
    step = STEP_TITLES.get(job.current_node) or "starting"
    st.warning(f"Another video is being made: {job.property_id} ({job.video_id}), {job.status}, step: "
               f"{step.lower()}. Only one video is made at a time; wait for it to finish or cancel it.")
    st.button("Open it", key=f"vt_open_{job.video_id}", on_click=_goto, args=(job.video_id,))


# -- the new video: the prompt box (video_box.py), and this form under it when Plan is on -----------------

FORM_FIELDS = ("objective", "audience", "highlights", "cta", "creative_direction", "tone", "voice", "music_mood",
               "hero_image", "burn_captions", "grade", "allow_generated", "pronunciations")


def render_form(property_id: str, folder: Path, defaults: dict) -> dict:
    """The brief form: everything the toolbar does not cover (it has the property, the length, the shape,
    the model, offline and draft). Starts from `defaults` (form_defaults of the property's folder). Returns
    the entries by FORM_FIELDS, `highlights` and `pronunciations` as the text areas' text."""
    def key(field: str) -> str:
        return form_key(property_id, field)

    def pick(options: list, current, field: str, **kwargs):
        options = options if current in options else [*options, current]
        return st.selectbox(kwargs.pop("label"), options, index=options.index(current), key=key(field), **kwargs)

    with st.container(border=True):
        st.subheader("The video")
        objective = pick(list(VIDEO_OBJECTIVES), defaults["objective"], "objective", label="What it is for",
                         format_func=lambda o: o.replace("_", " ").capitalize())
        audience = st.text_input("Who is it for?", defaults["audience"], key=key("audience"),
                                 placeholder="e.g. young families and first-time buyers")
        highlights = st.text_area("Key highlights to feature (one per line)", defaults["highlights"],
                                  key=key("highlights"), height=100,
                                  help="Only amenities and facts the property has, and that a photograph shows, are said.")
        cta = st.text_input("Call to action", defaults["cta"], key=key("cta"),
                            placeholder="e.g. Book a site visit this weekend (blank: chosen from what it is for)")
        direction = st.text_area("Creative direction", defaults["creative_direction"], key=key("creative_direction"),
                                 height=70, placeholder="e.g. calm and premium, evening light")

    with st.container(border=True):
        st.subheader("Look and sound")
        c1, c2 = st.columns(2)
        with c1:
            tone = pick(tone_choices(), defaults["tone"], "tone", label="Tone",
                        format_func=lambda t: t.capitalize() if t else "Default for the category")
            music_mood = st.text_input("Music mood", defaults["music_mood"], key=key("music_mood"),
                                       placeholder="e.g. warm acoustic (blank: chosen from the category)")
        with c2:
            voices = voice_choices()
            voice = pick(list(voices), defaults["voice"], "voice", label="Voice", format_func=lambda v: voices.get(v, v))
            heroes = hero_choices(folder)
            hero = pick([NO_HERO, *heroes], defaults["hero_image"], "hero_image", label="Opening picture",
                        format_func=lambda h: h or "Choose the best one for me")
        c1, c2, c3 = st.columns(3)
        burn = c1.checkbox("Print the narration on screen", defaults["burn_captions"], key=key("burn_captions"))
        grade = c2.checkbox("Light colour grade", defaults["grade"], key=key("grade"))
        generated = c3.checkbox("Allow generated frames", defaults["allow_generated"], key=key("allow_generated"),
                                help="Frames made for a shot where configs/video/styles.yaml allows it. "
                                     "Never of the building itself. Off unless you turn it on.")
        pronunciations = st.text_area("Pronunciations (one per line: word=how it is said)",
                                      defaults["pronunciations"], key=key("pronunciations"), height=80,
                                      placeholder="Anamika=Ah-nah-mih-kah")
        _, bad = parse_pronunciations(pronunciations)
        if bad:
            st.warning("These pronunciation lines are not 'word=respelling' and will be left out: " + "; ".join(bad))
    return {"objective": objective, "audience": audience, "highlights": highlights, "cta": cta,
            "creative_direction": direction, "tone": tone, "voice": voice, "music_mood": music_mood,
            "hero_image": hero, "burn_captions": burn, "grade": grade, "allow_generated": generated,
            "pronunciations": pronunciations}


def _new_video() -> None:
    from src.ui import video_box              # the box imports this module for the form and the helpers

    video_box.render_new_video()


# -- one video: progress, approval, result ---------------------------------------------------------------

def render_progress(video_id: str) -> Optional[VideoSummary]:
    """One poll: reads the status from the database and draws it. Returns the summary."""
    video_jobs.sweep()                           # a job whose process has gone is failed here, not waited for forever
    summary = video_jobs.status(video_id)
    if summary is None:
        return None
    producing = summary.status == "producing"
    st.markdown(f"**{'Making the video' if producing else 'Planning the video'}** · "
                f"{clock(elapsed_seconds(summary))} elapsed")
    states = step_states(summary)
    st.progress(min(1.0, summary.steps_done / len(states)))
    marks = {"done": "✅", "current": "⏳", "failed": "❌", "waiting": "▫️"}
    for _, title, state in states:
        st.write(f"{marks[state]} {title}")
    if summary.status in ACTIVE and st.button("Cancel", key=f"vt_cancel_{video_id}"):
        video_jobs.cancel(video_id)
        st.rerun()
    return summary


@st.fragment(run_every=POLL_SECONDS)
def _watch(video_id: str) -> None:
    """Polls every POLL_SECONDS while the job runs; when it stops, the whole page is drawn again."""
    summary = render_progress(video_id)
    if not is_active(summary):
        st.rerun()


# Plan notes that are about how the plan was made, not about what is in it: the ledger and the caches, the models
# that read things, the files that were set aside. They are shown folded, under "Technical".
TECHNICAL_NOTES = re.compile(r"^Fact ledger|read by a vision model|usable photographs; set aside|refresh-ledger|"
                             r"^Blocked by the run's policy", re.I)
THIN_SCRIPT = 0.7                     # a script under this share of its word target is flagged above the script


def split_notes(record: VideoGenerationRecord) -> tuple[list[str], list[str]]:
    """(about the plan, technical): the plan's notes and defects, each once, in the order they were written."""
    notes = list(dict.fromkeys([*record.notes, *record.defects]))
    return ([n for n in notes if not TECHNICAL_NOTES.search(n)], [n for n in notes if TECHNICAL_NOTES.search(n)])


def is_thin(script, brief) -> bool:
    return bool(brief.target_words) and script.word_count < THIN_SCRIPT * brief.target_words


def _approval(video_id: str) -> None:
    record = repository.get_generation(video_id)
    if record is None or record.script is None or record.storyboard is None or record.brief is None:
        st.error("This plan cannot be shown: its script or storyboard is missing. Make the plan again.")
        return
    script, board, brief = record.script, record.storyboard, record.brief
    st.subheader("Review the plan")
    st.caption(f"{brief.name} · {brief.duration_s} seconds · {brief.aspect} · {brief.objective.replace('_', ' ')} · "
               f"{brief.tone} tone")
    about, technical = split_notes(record)
    if about:
        st.markdown("**About this plan**")
        for note in about:
            st.write(f"- {note}")
    if is_thin(script, brief):
        st.warning(f"This script is short: {script.word_count} words, {script.word_count * 100 // brief.target_words}% "
                   f"of the {brief.target_words} a {brief.duration_s} second video has room for. Add highlights or "
                   "more property details for a fuller script, or choose a shorter length.")

    st.markdown("**Script**")
    for index, line in enumerate(script.lines, start=1):
        st.write(f"{index}. {line.text}")
        st.caption(f"{line.role} · {line.words} words")
    st.caption(f"{script.word_count} words; the plan aimed for about {brief.target_words}.")

    st.markdown("**Shots**")
    for shot in board.shots:
        picture, text = st.columns([1, 3])
        with picture:
            if shot.kind == "real" and shot.source_path:
                _show_thumbnail(shot.source_path)
            else:
                st.caption("Closing card")
        with text:
            how = expected_layout(shot, brief)
            st.markdown(f"**Shot {shot.index + 1}** · {shot.seconds:.1f} s"
                        + (f" · {LAYOUT_WORDS.get(how, how)}" if how else ""))
            said = [script.lines[i].text for i in shot.line_ids if 0 <= i < len(script.lines)]
            if said:
                st.write("Says: " + " ".join(said))
            if shot.subject and shot.kind == "real":
                st.caption(f"Shows: {shot.caption or shot.subject}")
            if shot.overlay_text:
                st.write("On screen: " + " · ".join(shot.overlay_text))

    if technical:
        with st.expander(f"Technical ({len(technical)})"):
            for note in technical:
                st.write(f"- {note}")

    approve, replan, _ = st.columns([2.2, 1.2, 5.6])
    if approve.button("Approve and produce", type="primary", key=f"vt_approve_{video_id}"):
        try:
            video_jobs.start_produce(video_id)
        except JobBusyError as exc:
            busy_notice(exc)
        except JobError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    if replan.button("Re-plan", key=f"vt_replan_{video_id}",
                     help="Makes a new plan with the same choices. This plan stays as it is."):
        request = record.request or VideoRequest(property_id=record.property_id)
        try:
            new_id = video_jobs.start_plan(request.property_id, request.overrides, offline=request.offline,
                                           draft=request.draft, refresh_ledger=request.refresh_ledger,
                                           prompt=request.prompt, brief_source=request.brief_source,
                                           brief_notes=request.brief_notes, sources=request.sources)
        except JobBusyError as exc:
            busy_notice(exc)
        except JobError as exc:
            st.error(str(exc))
        else:
            open_video(new_id)


def _failure(video_id: str, summary: VideoSummary) -> None:
    if summary.status == "cancelled":
        st.warning("This video was cancelled.")
    else:
        st.error("We couldn't finish this video. " + (summary.error or "") +
                 " Make the plan again — if it keeps happening, check that the photographs open correctly.")
    tail = log_tail(video_id)
    if tail:
        with st.expander("Last lines of the job's log"):
            st.code(tail, language="text")


def _result(video_id: str, summary: VideoSummary) -> None:
    record = repository.get_generation(video_id)
    path = Path(record.video_path) if record and record.video_path else None
    if path is None or not path.is_file():
        _failure(video_id, summary)
        return
    if summary.status == "passed":
        st.success("The video is ready.")
    elif summary.status == "warned":
        st.warning("The video is ready, with a few things worth a look.")
    else:
        st.error("The video was made but did not pass the quality check.")
    st.video(str(path))
    if record.evaluation is not None:
        st.write(f"**Quality check:** {record.evaluation.gate_result.value.upper()} — "
                 f"score **{record.evaluation.quality_score:.0f}/100**")
    st.caption("Evaluation and history come next.")


SOURCE_WORDS = {"model": "Read by the model, then checked against the property's facts.",
                "rules": "Read by built-in rules (the model was not used)."}


SOURCE_LABELS = {"toolbar": "Toolbar", "form": "Form (changed by you)", "prompt": "Your prompt",
                 "saved defaults": "Saved defaults (video_input.json)", "default": "Built-in default"}
NOT_RECORDED = "Not recorded (made before sources were kept)"


def brief_rows(brief, sources: dict[str, str]) -> list[dict]:
    """The final brief, option by option, with where each value came from (see video_box.option_sources)."""
    voices = voice_choices()
    models = {i: s.label for i, s in selector.load_catalog().items()}
    shown = [("Length", "duration_s", f"{brief.duration_s} seconds"), ("Shape", "aspect", brief.aspect),
             ("Objective", "objective", brief.objective.replace("_", " ").capitalize()),
             ("Audience", "audience", brief.audience or "—"), ("Tone", "tone", brief.tone.capitalize()),
             ("Highlights", "highlights", "; ".join(brief.highlights) or "—"),
             ("Call to action", "cta", brief.cta_text or "—"), ("Music mood", "music_mood", brief.music_mood or "—"),
             ("Voice", "voice", voices.get(brief.voice or default_voice(), brief.voice)),
             ("Model", "model", "Auto" if brief.model == "auto" else models.get(brief.model, brief.model))]
    return [{"Option": name, "Value": value, "From": SOURCE_LABELS.get(sources.get(key, ""), NOT_RECORDED)}
            for name, key, value in shown]


def _prompt_panel(video_id: str) -> None:
    """The prompt this video was made from, how it was read, and what was left out or not used."""
    record = repository.get_generation(video_id)
    request = record.request if record else None
    if request is None or not request.prompt.strip():
        return
    with st.expander("Your prompt"):
        st.text(request.prompt)
        if request.brief_source:
            st.caption(SOURCE_WORDS.get(request.brief_source, request.brief_source))
        for note in request.brief_notes:
            st.write(f"- {note}")
        if record.brief is not None:
            st.markdown("**The brief this video was planned with**")
            st.table(brief_rows(record.brief, request.sources))


def _video(video_id: str) -> None:
    st.button("← New video", key="vt_new", on_click=_goto, args=(None,))
    summary = video_jobs.status(video_id)
    if summary is None:
        st.warning(f"There is no video {video_id}.")
        return
    st.caption(f"{summary.property_id} · {summary.video_id}")
    _prompt_panel(video_id)
    if summary.status in ACTIVE:
        _watch(video_id)
    elif summary.status == STATUS_PLANNED:
        _approval(video_id)
    elif summary.status in ("passed", "warned", "failed"):
        _result(video_id, summary)
    else:
        _failure(video_id, summary)


def render_video_tab() -> None:
    st.caption("A short promotional video made from a property's own photographs, with narration, captions and "
               "music. Turn Plan on to read the brief and the storyboard before anything is produced.")
    video_id = st.query_params.get(VIDEO_QUERY)
    if video_id:
        _video(video_id)
    else:
        _new_video()
