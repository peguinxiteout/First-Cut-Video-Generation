"""
The Video tab's input: a prompt box with a toolbar, and the brief form under it when Plan is on.

    [ photographs of the chosen property ]
    [ the prompt                                              ]
    [ + Property ] [ Model ] [ Shape ] [ More ]      (Plan) (send)
    Estimate: about $0.04 · 15 s · pan-and-zoom

    Plan off   send reads the prompt (rules, then one model call), and starts the full job: plan and
               produce with no stop. The toolbar's choices win over the prompt's.
    Plan on    the form shows under the box, filled from the property's video_input.json. Send reads the
               prompt, takes its value for each form field the person has not changed, keeps the person's
               value for the others, puts the toolbar's choices on top, and starts the plan job; the
               storyboard approval is the review.

The pieces that decide something (what the toolbar and the prompt each win, the estimate) are plain
functions so they can be tested without a page. Length and shape are "Auto" until chosen: Auto means the
prompt says, else the property's video_input.json says. The voice is in the form only; with Plan off the
property's own voice is used. The page's own CSS is here, not in streamlit_app.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from src.agents.video import prompt_brief
from src.core.config import load_yaml_config, settings
from src.core.exceptions import ConfigError, IngestionError, JobBusyError, JobError
from src.data.ingestion import discover_images
from src.data.video_schemas import VIDEO_ASPECTS, VIDEO_DURATIONS
from src.models.video_gen import selector
from src.ui import video_tab
from src.workflows import video_jobs

logger = logging.getLogger(__name__)

AUTO = "auto"
LENGTHS = (AUTO, *VIDEO_DURATIONS)
SHAPES = (AUTO, *VIDEO_ASPECTS)
KEY_PROMPT, KEY_PROPERTY, KEY_MODEL = "vt_prompt", "vt_property", "vt_model"
KEY_SHAPE, KEY_LENGTH, KEY_OFFLINE, KEY_DRAFT, KEY_PLAN, KEY_SEND = (
    "vt_shape", "vt_length", "vt_offline", "vt_draft", "vt_plan", "vt_send")
STRIP_PHOTOS = 12
CHARS_PER_WORD = 6                           # an English word and the space after it, for the voice's price
PLACEHOLDER = ("Describe the video you want: who it is for, what to highlight, the mood, the call to action. "
               "Camera moves and on-screen text are decided by the tool, not by the prompt.")

CSS = """
<style>
  .st-key-vt_box { border: 1px solid var(--fc-line, #E6E1F7); border-radius: 22px; background: #FFFFFF;
      padding: .7rem .9rem .5rem; box-shadow: 0 1px 2px rgba(28,23,51,.04), 0 18px 48px -22px rgba(91,51,240,.25); }
  .st-key-vt_prompt [data-testid="stTextAreaRootElement"] { border: none !important; background: transparent !important;
      box-shadow: none !important; }
  .st-key-vt_prompt textarea { background: transparent !important; font-size: 1.02rem; line-height: 1.5; }
  .st-key-vt_toolbar [data-testid="stPopover"] button { border-radius: 999px; background: var(--fc-lav, #F4F1FE);
      border: 1px solid var(--fc-line, #E6E1F7); color: var(--fc-ink, #1C1733); font-size: .88rem; padding: .25rem .8rem;
      min-height: 2.1rem; }
  .st-key-vt_toolbar [data-testid="stPopover"] button:hover { border-color: var(--fc-violet, #5B33F0);
      color: var(--fc-violet, #5B33F0); }
  .st-key-vt_send button { border-radius: 50% !important; width: 2.6rem; height: 2.6rem; min-height: 0 !important;
      padding: 0 !important; font-size: 1.2rem; justify-content: center; }
  .st-key-vt_cost p { color: var(--fc-muted, #6B6880); font-size: .85rem; }
  @media (max-width: 640px) { .st-key-vt_box { padding: .5rem .55rem .4rem; border-radius: 18px; } }
</style>
"""


@dataclass(frozen=True)
class Toolbar:
    """What the box's toolbar says. `length` and `shape` are None for Auto; `model` is a provider id or "auto"."""
    property_id: str
    prompt: str
    length: Optional[int] = None
    shape: Optional[str] = None
    model: str = AUTO
    offline: bool = False
    draft: bool = False
    plan: bool = False


@dataclass(frozen=True)
class Estimate:
    usd: float
    credits: bool                   # the chosen model is paid in credits, which are not in `usd`
    duration_s: int
    motion: str                     # the label of what moves the pictures
    offline: bool


# ---------------------------------------------------------------------------
# What wins: the toolbar, the form, the prompt, the property's file
# ---------------------------------------------------------------------------

def plan_off_overrides(prompt_values: dict, toolbar: Toolbar) -> dict:
    """The options of a Plan-off video: only what the prompt set, with the toolbar's choices on top. The
    rest is the property's video_input.json and the defaults. The voice is never in it."""
    out = dict(prompt_values)
    if toolbar.length:
        out["duration_s"] = toolbar.length
    if toolbar.shape:
        out["aspect"] = toolbar.shape
    if toolbar.model != AUTO:
        out["model"] = toolbar.model
    return out


def _as_form(field: str, value):
    """A prompt's value as the form holds it: the text areas hold text."""
    if field == "highlights":
        return "\n".join(value)
    if field == "pronunciations":
        return video_tab.pronunciations_text(value)
    return value


def merge_prompt(values: dict, defaults: dict, prompt_values: dict) -> dict:
    """The form's entries with the prompt's value in each field the person has not changed from what the form
    started with. A field the person changed keeps the person's value."""
    merged = dict(values)
    for field in video_tab.FORM_FIELDS:
        if field in prompt_values and values.get(field) == defaults.get(field):
            merged[field] = _as_form(field, prompt_values[field])
    return merged


def with_toolbar(values: dict, defaults: dict, toolbar: Toolbar, prompt_values: Optional[dict] = None) -> dict:
    """The form's entries with the length, shape and model added: the toolbar's, else the prompt's, else the
    property's own."""
    prompt_values = prompt_values or {}
    return {**values,
            "duration_s": toolbar.length or prompt_values.get("duration_s") or defaults["duration_s"],
            "aspect": toolbar.shape or prompt_values.get("aspect") or defaults["aspect"],
            "model": toolbar.model if toolbar.model != AUTO else defaults["model"]}


def plan_on_overrides(values: dict, defaults: dict, prompt_values: dict, toolbar: Toolbar) -> dict:
    """The options of a Plan-on video: all of them, as the form sends them."""
    return video_tab.build_overrides(with_toolbar(merge_prompt(values, defaults, prompt_values), defaults, toolbar,
                                                  prompt_values))


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------

def _speech_usd(duration_s: int) -> float:
    """The voice, when Google Cloud TTS is the one that speaks (it has a key); the others are free here."""
    if not settings.google_cloud_tts_api_key:
        return 0.0
    speech = load_yaml_config("video/speech.yaml")
    spoken_s = max(0, duration_s - load_yaml_config("video/pacing.yaml")["end_card_s"])
    chars = spoken_s * speech["words_per_second"] * CHARS_PER_WORD
    return chars * speech["cloud_tts"]["usd_per_million_chars"] / 1_000_000


def _music_usd(duration_s: int) -> float:
    """The music, from the first service that has a key: Beatoven, else Lyria (paid per track)."""
    music = load_yaml_config("video/music.yaml")
    if settings.beatoven_api_key:
        return float(music["beatoven"]["cost_usd"])
    if settings.gemini_api_key:
        models = music["lyria"]["models"]
        return float(next((m for m in models if duration_s <= m["max_video_s"]), models[-1])["cost_usd"])
    return 0.0


def estimate(duration_s: int, model: str, offline: bool) -> Estimate:
    """About what a video costs: the motion model, the voice and the music, from the rates in the configs."""
    if offline:
        return Estimate(0.0, False, duration_s, "pan-and-zoom", True)
    try:
        chain = selector.resolve(model, duration_s)
    except ConfigError:
        chain = selector.resolve(AUTO, duration_s)
    option = next((o for o in selector.options(duration_s) if o.id == chain.ids[0]), None)
    motion = chain.est_cost_usd or 0.0
    usd = motion + _speech_usd(duration_s) + _music_usd(duration_s)
    return Estimate(usd, bool(option and option.cost_basis == "credits" and option.est_cost_usd is None),
                    duration_s, option.label if option else "pan-and-zoom", False)


def estimate_text(est: Estimate) -> str:
    if est.offline:
        return "Estimate: free. Offline: no models are asked and nothing leaves this computer."
    if est.credits:
        amount = f"about ${video_tab.money(est.usd)} plus credits" if est.usd >= 0.01 else "credits only"
    elif est.usd <= 0:
        amount = "free"
    elif est.usd < 0.01:
        amount = "under $0.01"
    else:
        amount = f"about ${video_tab.money(est.usd)}"
    return f"Estimate: {amount} · {est.duration_s} s · {est.motion}"


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _strip_thumbnail(path: str, modified: float) -> bytes:
    import io

    from PIL import Image

    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((160, 160))
        buffer = io.BytesIO()
        picture.save(buffer, "JPEG", quality=75)
    return buffer.getvalue()


def _photo_strip(folder: Path) -> None:
    try:
        photos = [i for i in discover_images(folder) if i.role in ("", "photo")]
    except IngestionError:
        return
    shown = []
    for photo in photos[:STRIP_PHOTOS]:
        try:
            shown.append(_strip_thumbnail(photo.path, Path(photo.path).stat().st_mtime))
        except Exception:  # noqa: BLE001 - a picture that cannot be read is left out of the strip
            continue
    if shown:
        st.image(shown, width=54)
        more = len(photos) - len(shown)
        st.caption(f"{len(photos)} photographs" + (f" ({more} not shown)" if more > 0 else ""))


def _short(label: str, limit: int = 20) -> str:
    label = label.split(" (")[0]
    return label if len(label) <= limit else label[:limit - 1].rstrip() + "…"


def _guard_state(labels: dict, usable: list[str]) -> None:
    """Puts the toolbar's session values in order before the widgets are drawn: a choice that is no longer
    on offer (a property that was removed, a model that is no longer available) is dropped."""
    state = st.session_state
    if state.get(KEY_PROPERTY) is not None and state[KEY_PROPERTY] not in labels:
        del state[KEY_PROPERTY]
    for key, valid in ((KEY_LENGTH, LENGTHS), (KEY_SHAPE, SHAPES)):
        if state.get(key) not in valid:
            state[key] = AUTO
    if state.get(KEY_MODEL) not in usable:
        state[KEY_MODEL] = AUTO
    for key in (KEY_OFFLINE, KEY_DRAFT, KEY_PLAN):
        state.setdefault(key, False)
    state.setdefault(KEY_PROMPT, "")


def render_new_video() -> None:
    choices = video_tab.property_choices()
    if not choices:
        st.info("There is no property to make a video of yet. Add one with the Brochure form first; "
                "its photographs and details are what the video is made from.")
        return
    labels = dict(choices)
    state = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)

    # What the toolbar says right now (the values of the last widget run), so the model list and the
    # estimate can be drawn before the widgets that change them.
    prompt = state.get(KEY_PROMPT, "") or ""
    options = selector.options(15)
    _guard_state(labels, [o.id for o in options if o.available])
    property_id: Optional[str] = state.get(KEY_PROPERTY)
    folder = video_tab.properties_dir() / property_id if property_id else None
    defaults, notes = video_tab.form_defaults(folder) if folder else (None, [])
    length = None if state[KEY_LENGTH] == AUTO else int(state[KEY_LENGTH])
    shape = None if state[KEY_SHAPE] == AUTO else state[KEY_SHAPE]
    offline = bool(state[KEY_OFFLINE])
    duration = length or prompt_brief.length_of(prompt) or (defaults["duration_s"] if defaults else 15)
    options = selector.options(duration)
    usable = {o.id: o for o in options if o.available}
    model = state[KEY_MODEL]

    with st.container(border=True, key="vt_box"):
        if folder:
            _photo_strip(folder)
        st.text_area("Your prompt", key=KEY_PROMPT, height=150, max_chars=prompt_brief.max_prompt_chars(),
                     placeholder=PLACEHOLDER, label_visibility="collapsed")
        with st.container(key="vt_toolbar"):
            c_plus, c_model, c_shape, c_more, _, c_plan, c_send = st.columns(
                [2.0, 2.0, 1.8, 1.8, 0.5, 1.6, 0.8], vertical_alignment="center")
            with c_plus.popover("+ " + (_short(labels[property_id]) if property_id else "Property"), key="vt_plus",
                                help="Choose the property the video is made from"):
                st.caption("Property")
                with st.container(height=min(70 + 38 * len(choices), 280)):
                    st.radio("Property", list(labels), index=None, format_func=labels.get, key=KEY_PROPERTY,
                             label_visibility="collapsed")
                st.divider()
                st.button("Recipes, coming soon", key="vt_recipes", disabled=True)
            with c_model.popover("Model: " + ("Auto" if model == AUTO else _short(usable[model].label, 16)),
                                 key="vt_model_menu", help="What moves the pictures"):
                st.radio("How the pictures move", list(usable),
                         format_func=lambda i: video_tab.md(video_tab.option_label(usable[i])),
                         key=KEY_MODEL, disabled=offline)
                if offline:
                    st.caption("Offline: pan-and-zoom only.")
                unavailable = [o for o in options if not o.available]
                if unavailable:
                    st.markdown("**Not available yet**")
                    for option in unavailable:
                        st.caption(video_tab.md(video_tab.unavailable_row(option)))
                st.caption("A shot a model cannot make is made by pan-and-zoom, which is free.")
            with c_shape.popover("Shape: " + (shape or "Auto"), key="vt_shape_menu", help="The shape of the video"):
                st.radio("Shape", SHAPES, format_func=lambda s: "Auto" if s == AUTO else s, key=KEY_SHAPE)
                st.caption("Auto: the prompt says, else the property's own choice, else 9:16.")
            more = [f"{length} s" if length else "", "offline" if offline else "",
                    "draft" if state[KEY_DRAFT] else ""]
            with c_more.popover("More" + "".join(f" · {m}" for m in more if m), key="vt_more",
                                help="Length, offline, draft quality"):
                st.radio("Length", LENGTHS, format_func=lambda s: "Auto" if s == AUTO else f"{s} s",
                         key=KEY_LENGTH, horizontal=True)
                st.caption("Auto: the prompt says, else the property's own choice, else 15 s.")
                st.checkbox("Offline (no models, no cost)", key=KEY_OFFLINE,
                            help="Every step takes its built-in fallback: a voice on this machine, pan-and-zoom, no "
                                 "music from a service. Nothing leaves this computer.")
                st.checkbox("Draft quality (fast)", key=KEY_DRAFT,
                            help="Small and quick, to look at a plan before spending the time on the full size.")
            plan = c_plan.toggle("Plan", key=KEY_PLAN,
                                 help="Plan mode: review the brief and the storyboard first. The video is made only "
                                      "after you approve.")
            ready = bool(property_id and prompt.strip())
            send = c_send.button("↑", key=KEY_SEND, type="primary", disabled=not ready,
                                 help="Make the video" if ready else "Choose a property and write a prompt first")
    for note in notes:
        st.warning(note)
    effective = model if model != AUTO else (defaults["model"] if defaults else AUTO)
    with st.container(key="vt_cost"):
        st.caption(video_tab.md(estimate_text(estimate(duration, "kenburns" if offline else effective, offline))))
    if duration > 15:
        st.caption("Videos over 15 seconds have a script plan but have not been rendered and checked yet.")
    if not property_id:
        st.caption("Choose a property with + before sending.")

    toolbar = Toolbar(property_id or "", prompt, length, shape, model, offline, bool(state[KEY_DRAFT]), bool(plan))
    values: dict = {}
    if plan and folder and defaults:
        values = video_tab.render_form(property_id, folder, defaults)
        if st.button("Save as this property's defaults", key="vt_save",
                     help=f"Writes these choices to {video_tab.brief_agent.VIDEO_INPUT_FILE} in the property's folder, "
                          "keeping anything else in it."):
            try:
                path = video_tab.merge_defaults(folder, video_tab.build_overrides(with_toolbar(values, defaults, toolbar)))
            except IngestionError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved to {path.name} in the {property_id} folder.")
    elif plan:
        st.caption("Choose a property to see the form.")
    if send and folder and defaults is not None:
        _send(toolbar, folder, defaults, values)


def _send(toolbar: Toolbar, folder: Path, defaults: dict, values: dict) -> None:
    """Reads the prompt, works out the options (see the module's top) and starts the job."""
    try:
        with st.spinner("Reading your prompt…"):
            result = prompt_brief.read_prompt(toolbar.prompt, folder, offline=toolbar.offline)
    except IngestionError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - the box must not crash the page; the log has the traceback
        logger.exception("The prompt could not be read")
        st.error(f"The prompt could not be read ({type(exc).__name__}). Try again, or turn Plan on and fill the form.")
        return
    prompt_values = prompt_brief.prompt_overrides(result)
    overrides = (plan_on_overrides(values, defaults, prompt_values, toolbar) if toolbar.plan
                 else plan_off_overrides(prompt_values, toolbar))
    try:
        video_id = video_jobs.start_plan(
            folder.name, overrides, offline=toolbar.offline, draft=toolbar.draft, auto_approve=not toolbar.plan,
            prompt=toolbar.prompt.strip(), brief_source=result.source, brief_notes=result.notes)
    except JobBusyError as exc:
        video_tab.busy_notice(exc)
    except JobError as exc:
        st.error(str(exc))
    else:
        video_tab.open_video(video_id)
