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

Above the box is "Start with a Recipe" (src/agents/video/recipes.py): a recipe fills the prompt box, sets the
length and shape in the toolbar and starts the form's objective, tone, music mood and call to action. It
sits below the prompt: the toolbar, the form, the prompt, the recipe, the property's video_input.json, the defaults.

The pieces that decide something (what the toolbar and the prompt each win, the estimate) are plain
functions so they can be tested without a page. Length and shape are "Auto" until chosen: Auto means the
prompt says, else the property's video_input.json says. The voice is in the form only; with Plan off the
property's own voice is used. The page's own CSS is here, not in streamlit_app.py.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from src.agents.video import prompt_brief, recipes
from src.core.config import load_yaml_config, settings
from src.core.exceptions import ConfigError, IngestionError, JobBusyError, JobError
from src.data.ingestion import discover_images
from src.data.schemas import ImageAsset
from src.data.video_schemas import VIDEO_ASPECTS, VIDEO_DURATIONS, VideoRecipe
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
KEY_RECIPE, KEY_RECIPES_OPEN, KEY_RECIPE_CATEGORY, KEY_RECIPE_TEXT = (
    "vt_recipe", "vt_recipes_open", "vt_recipe_category", "vt_recipe_text")
ALL_RECIPES = "all"
RECIPE_COLUMNS = 3
THUMB_STYLE = "display:block;width:100%;height:108px;object-fit:cover;border-radius:10px"   # inline: the page's own image rule wins over a class
STRIP_PHOTOS = 30
CHARS_PER_WORD = 6                           # an English word and the space after it, for the voice's price
PLACEHOLDER = ("Describe the video you want: who it is for, what to highlight, the mood, the call to action. "
               "Camera moves and on-screen text are decided by the tool, not by the prompt.")

CSS = """
<style>
  .st-key-vt_box { border: 1px solid var(--fc-line, #E6E1F7); border-radius: 22px; background: #FFFFFF;
      padding: .7rem .9rem .5rem; box-shadow: 0 1px 2px rgba(28,23,51,.04), 0 18px 48px -22px rgba(91,51,240,.25); }
  .vt-strip { display: flex; gap: 6px; overflow-x: auto; padding: 2px 0 8px; }
  .vt-strip img { height: 84px; width: auto; flex: 0 0 auto; border-radius: 8px; }
  .st-key-vt_prompt [data-testid="stTextAreaRootElement"] { border: none !important; background: transparent !important;
      box-shadow: none !important; }
  .st-key-vt_prompt textarea { background: transparent !important; font-size: 1.02rem; line-height: 1.5; }
  /* the toolbar is one row of buttons as wide as their labels, evenly spaced; the fifth cell takes the slack */
  .st-key-vt_toolbar [data-testid="stHorizontalBlock"] { display: flex; flex-wrap: wrap; align-items: center;
      gap: .5rem !important; }
  .st-key-vt_toolbar [data-testid="stColumn"] { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
  .st-key-vt_toolbar [data-testid="stColumn"]:nth-child(5) { flex: 1 1 0 !important; }
  .st-key-vt_toolbar [data-testid="stPopover"] button { border-radius: 999px; background: var(--fc-lav, #F4F1FE);
      border: 1px solid var(--fc-line, #E6E1F7); color: var(--fc-ink, #1C1733); font-size: .88rem; padding: .25rem .8rem;
      min-height: 2.1rem; white-space: nowrap; }
  .st-key-vt_toolbar [data-testid="stPopover"] button:hover { border-color: var(--fc-violet, #5B33F0);
      color: var(--fc-violet, #5B33F0); }
  .st-key-vt_send button { border-radius: 50% !important; width: 2.6rem; height: 2.6rem; min-height: 0 !important;
      padding: 0 !important; font-size: 1.2rem; justify-content: center; }
  .st-key-vt_recipe_chip button { border-radius: 999px; background: var(--fc-lav, #F4F1FE); border: 1px solid var(--fc-violet, #5B33F0);
      color: var(--fc-violet, #5B33F0); font-size: .85rem; padding: .15rem .8rem; min-height: 1.9rem; }
  .vt-thumb.empty { background: linear-gradient(135deg, #F4F1FE, #E6E1F7); }
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

def recipe_defaults(defaults: dict, recipe: Optional[VideoRecipe]) -> dict:
    """The form's starting values with the recipe's laid over the property's file: the recipe sits above
    video_input.json and under everything the person or the prompt says."""
    return {**defaults, **recipes.brief_values(recipe)} if recipe else defaults


def plan_off_overrides(prompt_values: dict, toolbar: Toolbar, recipe_values: Optional[dict] = None) -> dict:
    """The options of a Plan-off video: what the recipe set, then what the prompt set, with the toolbar's choices
    on top. The rest is the property's video_input.json and the defaults. The voice is never in it."""
    out = {**(recipe_values or {}), **prompt_values}
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


SOURCE_FIELDS = ("duration_s", "aspect", "objective", "audience", "tone", "highlights", "cta", "music_mood", "voice",
                 "model")


def saved_keys(folder: Path) -> set[str]:
    """The options the property's video_input.json sets (none when there is no file or it cannot be read)."""
    try:
        data = json.loads((folder / video_tab.brief_agent.VIDEO_INPUT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(data) if isinstance(data, dict) else set()


def option_sources(toolbar: Toolbar, form_changed: set[str], prompt_keys: set[str], saved: set[str],
                   recipe: Optional[dict] = None) -> dict[str, str]:
    """Where each option of the brief came from, in the order they win: the toolbar, the form (a field the person
    changed), the prompt, the recipe, the property's saved defaults, else the built-in default. A length or shape
    in the toolbar that is the recipe's own (the recipe put it there) is the recipe's."""
    recipe = recipe or {}
    chosen = (("duration_s", toolbar.length), ("aspect", toolbar.shape),
              ("model", toolbar.model != AUTO or toolbar.offline))
    from_toolbar = {key for key, value in chosen if value and (key == "model" or value != recipe.get(key))}
    return {field: ("toolbar" if field in from_toolbar else "form" if field in form_changed
                    else "prompt" if field in prompt_keys else "recipe" if field in recipe
                    else "saved defaults" if field in saved else "default")
            for field in SOURCE_FIELDS}


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
def _strip_markup(paths: tuple[str, ...], modified: tuple[float, ...]) -> str:
    """One row of small pictures that scrolls sideways (the page's own CSS, above): a data address each, so
    nothing is fetched. Cached on the files and when they last changed."""
    from PIL import Image

    pictures = []
    for path in paths:
        try:
            with Image.open(path) as picture:
                picture = picture.convert("RGB")
                picture.thumbnail((240, 240))
                buffer = io.BytesIO()
                picture.save(buffer, "JPEG", quality=72)
        except Exception:  # noqa: BLE001 - a picture that cannot be read is left out of the strip
            continue
        pictures.append(f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}" alt="">')
    return f'<div class="vt-strip">{"".join(pictures)}</div>' if pictures else ""


def _photo_strip(folder: Path) -> None:
    try:
        photos = [i for i in discover_images(folder) if i.role in ("", "photo")]
    except IngestionError:
        return
    shown = [p.path for p in photos[:STRIP_PHOTOS] if Path(p.path).is_file()]
    markup = _strip_markup(tuple(shown), tuple(Path(p).stat().st_mtime for p in shown)) if shown else ""
    if markup:
        st.markdown(markup, unsafe_allow_html=True)
        more = len(photos) - len(shown)
        st.caption(f"{len(photos)} photographs" + (f" ({more} not shown)" if more > 0 else ""))


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
    state.setdefault(KEY_RECIPES_OPEN, True)
    if state.get(KEY_RECIPE) and recipes.get(state[KEY_RECIPE]) is None:        # a recipe that is no longer offered
        state[KEY_RECIPE] = ""


# ---------------------------------------------------------------------------
# Recipes: "Start with a Recipe", the chip in the box
# ---------------------------------------------------------------------------

RECIPE_FORM_FIELDS = ("objective", "tone", "music_mood", "cta")      # the form fields a recipe starts


def _property_name(property_id: Optional[str]) -> str:
    """The property's name for the recipe's words ("" when none is chosen or its file has no name)."""
    if not property_id:
        return ""
    try:
        data = json.loads((video_tab.properties_dir() / property_id / "property.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("name") or "").strip() if isinstance(data, dict) else ""


def _forget_form() -> None:
    """Drops the form's entries for the fields a recipe starts, so the form begins again from its defaults (the
    recipe's over the property's file, or the file's alone once the recipe is cleared)."""
    for property_id, _ in video_tab.property_choices():
        for field in RECIPE_FORM_FIELDS:
            st.session_state.pop(video_tab.form_key(property_id, field), None)


def use_recipe(recipe_id: str) -> None:
    """The recipe's words go in the prompt box, its length and shape in the toolbar, and its objective, tone, music
    mood and call to action become the form's starting values (see recipe_defaults). The Plan toggle is left
    as it is. A callback: it runs before the page is drawn, so it may set the widgets' values."""
    recipe = recipes.get(recipe_id)
    if recipe is None:
        return
    state = st.session_state
    state[KEY_PROMPT] = state[KEY_RECIPE_TEXT] = recipes.prompt_text(recipe, _property_name(state.get(KEY_PROPERTY)))
    state[KEY_LENGTH], state[KEY_SHAPE] = recipe.duration_s, recipe.aspect
    state[KEY_RECIPE], state[KEY_RECIPES_OPEN] = recipe.id, False
    _forget_form()


def clear_recipe() -> None:
    """Takes the recipe away. What it put in the toolbar and the prompt box goes too, unless the person has
    changed it since."""
    state = st.session_state
    recipe = recipes.get(state.get(KEY_RECIPE, ""))
    if recipe:
        if state.get(KEY_LENGTH) == recipe.duration_s:
            state[KEY_LENGTH] = AUTO
        if state.get(KEY_SHAPE) == recipe.aspect:
            state[KEY_SHAPE] = AUTO
        if state.get(KEY_PROMPT) == state.get(KEY_RECIPE_TEXT):
            state[KEY_PROMPT] = ""
    state[KEY_RECIPE] = state[KEY_RECIPE_TEXT] = ""
    _forget_form()


def _property_changed() -> None:
    """The recipe's words name the property: when the property changes and the words are the recipe's own, they
    are written again with the new name."""
    state = st.session_state
    recipe = recipes.get(state.get(KEY_RECIPE, ""))
    if recipe and state.get(KEY_PROMPT) == state.get(KEY_RECIPE_TEXT):
        state[KEY_PROMPT] = state[KEY_RECIPE_TEXT] = recipes.prompt_text(recipe, _property_name(state.get(KEY_PROPERTY)))


def show_recipes() -> None:
    st.session_state[KEY_RECIPES_OPEN] = True


def _toggle_recipes() -> None:
    st.session_state[KEY_RECIPES_OPEN] = not st.session_state.get(KEY_RECIPES_OPEN, True)


@st.cache_data(show_spinner=False)
def _scene_paths(paths: tuple[str, ...], modified: tuple[float, ...]) -> dict[str, str]:
    """The path of the property's photograph for each scene a card can show. Cached on the files and when they
    last changed."""
    images = [ImageAsset(image_id=f"IMG{n:03d}", filename=Path(p).name, path=p) for n, p in enumerate(paths, start=1)]
    return {scene: picture.path for scene, picture in recipes.scene_pictures(images).items()}


@st.cache_data(show_spinner=False)
def _thumbnail_uri(path: str, modified: float) -> str:
    from PIL import Image

    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((360, 240))
        buffer = io.BytesIO()
        picture.save(buffer, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _card_picture(path: Optional[str]) -> str:
    """A card's picture, or a plain placeholder when the property is not chosen or has no photograph of the scene."""
    if path:
        try:
            return f'<img class="vt-thumb" style="{THUMB_STYLE}" src="{_thumbnail_uri(path, Path(path).stat().st_mtime)}" alt="">'
        except Exception:  # noqa: BLE001 - a picture that cannot be read leaves the placeholder
            pass
    return f'<div class="vt-thumb empty" style="{THUMB_STYLE}"></div>'


def _recipe_pictures(folder: Optional[Path]) -> dict[str, str]:
    if folder is None:
        return {}
    try:
        photos = [i.path for i in discover_images(folder) if i.role in ("", "photo") and Path(i.path).is_file()]
    except IngestionError:
        return {}
    return _scene_paths(tuple(photos), tuple(Path(p).stat().st_mtime for p in photos))


def _recipe_section(folder: Optional[Path]) -> None:
    """The row above the box: category chips and a card for each recipe of the chosen category."""
    state = st.session_state
    with st.container(key="vt_recipes_section"):
        title, toggle = st.columns([5, 1.6], vertical_alignment="center")
        title.markdown("**Start with a Recipe**")
        is_open = bool(state[KEY_RECIPES_OPEN])
        toggle.button("Hide recipes" if is_open else "Show recipes", key="vt_recipes_toggle", on_click=_toggle_recipes)
        if not is_open:
            return
        categories = recipes.categories()
        st.pills("Category", [ALL_RECIPES, *categories], default=ALL_RECIPES, key=KEY_RECIPE_CATEGORY,
                 format_func=lambda c: "All" if c == ALL_RECIPES else categories[c], label_visibility="collapsed")
        chosen = state.get(KEY_RECIPE_CATEGORY) or ALL_RECIPES
        shown = [r for r in recipes.load() if chosen == ALL_RECIPES or r.category == chosen]
        pictures = _recipe_pictures(folder)
        if not shown:
            st.caption("No recipes in this category yet.")
        for start in range(0, len(shown), RECIPE_COLUMNS):
            for column, recipe in zip(st.columns(RECIPE_COLUMNS), shown[start:start + RECIPE_COLUMNS]):
                with column, st.container(border=True, key=f"vt_recipe_card_{recipe.id}"):
                    st.markdown(_card_picture(recipes.for_scene(pictures, recipe.thumbnail)), unsafe_allow_html=True)
                    st.markdown(f"**{recipe.name}**")
                    st.caption(recipe.description)
                    st.caption(f"{recipe.duration_s} s · {recipe.aspect}")
                    st.button("Use this recipe", key=f"vt_use_{recipe.id}", on_click=use_recipe, args=(recipe.id,),
                              type="primary" if state.get(KEY_RECIPE) == recipe.id else "secondary")


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
    recipe = recipes.get(state[KEY_RECIPE]) if state.get(KEY_RECIPE) else None
    if defaults is not None:
        defaults = recipe_defaults(defaults, recipe)
    length = None if state[KEY_LENGTH] == AUTO else int(state[KEY_LENGTH])
    shape = None if state[KEY_SHAPE] == AUTO else state[KEY_SHAPE]
    offline = bool(state[KEY_OFFLINE])
    duration = length or prompt_brief.length_of(prompt) or (defaults["duration_s"] if defaults else 15)
    options = selector.options(duration)
    usable = {o.id: o for o in options if o.available}
    model = state[KEY_MODEL]

    _recipe_section(folder)
    with st.container(border=True, key="vt_box"):
        if folder:
            _photo_strip(folder)
        if recipe:
            st.button(f"Recipe: {recipe.name} ✕", key="vt_recipe_chip", on_click=clear_recipe,
                      help="Clear the recipe: its length, shape and words go, and the form starts from the property's own "
                           "choices again. Anything you changed stays.")
        st.text_area("Your prompt", key=KEY_PROMPT, height=150, max_chars=prompt_brief.max_prompt_chars(),
                     placeholder=PLACEHOLDER, label_visibility="collapsed")
        with st.container(key="vt_toolbar"):
            c_plus, c_model, c_shape, c_more, _, c_plan, c_send = st.columns(
                [2.0, 2.0, 1.8, 1.8, 0.5, 1.6, 0.8], vertical_alignment="center")
            with c_plus.popover("+ " + (labels[property_id].split(" (")[0] if property_id else "Property"), key="vt_plus",
                                help="Choose the property the video is made from"):
                st.caption("Property")
                with st.container(height=min(70 + 38 * len(choices), 280)):
                    st.radio("Property", list(labels), index=None, format_func=labels.get, key=KEY_PROPERTY,
                             label_visibility="collapsed", on_change=_property_changed)
                st.divider()
                st.button("Recipes", key="vt_recipes", on_click=show_recipes,
                          help="Ready-made starts: they fill the prompt box, the toolbar and the form")
            with c_model.popover("Model: " + ("Auto" if model == AUTO else usable[model].label),
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
    state = st.session_state
    recipe = recipes.get(state.get(KEY_RECIPE, ""))
    recipe_values = recipes.brief_values(recipe) if recipe else {}
    overrides = (plan_on_overrides(values, defaults, prompt_values, toolbar) if toolbar.plan
                 else plan_off_overrides(prompt_values, toolbar, recipe_values))
    changed = {f for f in video_tab.FORM_FIELDS if toolbar.plan and values.get(f) != defaults.get(f)}
    from_prompt = set(prompt_values)
    if recipe and toolbar.prompt == state.get(KEY_RECIPE_TEXT):               # the recipe's own words, as they were
        from_prompt -= {k for k, v in prompt_values.items() if recipe_values.get(k) == v}
    sources = option_sources(toolbar, changed, from_prompt, saved_keys(folder), recipe_values)
    try:
        video_id = video_jobs.start_plan(
            folder.name, overrides, offline=toolbar.offline, draft=toolbar.draft, auto_approve=not toolbar.plan,
            prompt=toolbar.prompt.strip(), brief_source=result.source, brief_notes=result.notes, sources=sources,
            recipe_id=recipe.id if recipe else "")
    except JobBusyError as exc:
        video_tab.busy_notice(exc)
    except JobError as exc:
        st.error(str(exc))
    else:
        video_tab.open_video(video_id)
