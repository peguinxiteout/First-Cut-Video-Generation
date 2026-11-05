"""
Streamlit UI — First Cut, AI content for real estate marketing.

Two solutions:
    Brochure   one short form (the eleven things a brochure is made from,
               most important first) plus the pictures, each in its own
               slot. Category and format choose the template (A4 landscape
               or A4 portrait).
    Blog post  a topic and a few optional details; the result is a
               website-style PDF.

Every result is shown in five tabs and nothing else:
    Final PDF · Evaluation · Workflow · Human feedback

Errors are shown in plain words; the details go to the log.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import base64
import inspect
import json
import logging
from functools import lru_cache
from pathlib import Path

import streamlit as st

from src.core import win_asyncio

# Before anything starts an event loop (see the module).
win_asyncio.install()

from src.agents.brochure import catalog  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.core.ids import slugify  # noqa: E402
from src.data import blog_repository, property_form, repository, video_repository  # noqa: E402
from src.evaluation.tiers import quality_tier  # noqa: E402
from src.rendering.renderer import is_pdf_render_valid  # noqa: E402
from src.ui.video_tab import render_video_tab  # noqa: E402
from src.workflows.blog import run_blog_workflow  # noqa: E402
from src.workflows.brochure import run_brochure_workflow  # noqa: E402

logger = logging.getLogger("streamlit_app")

PRODUCT_NAME = "First Cut"
_BRAND = Path(__file__).parent / "assets" / "brand"
_ICON_PNG = _BRAND / "first-cut-icon.png"      # the painted F on a lavender tile
_LOGO_PNG = _BRAND / "first-cut-logo.png"      # mark + wordmark, transparent

st.set_page_config(page_title=f"{PRODUCT_NAME} · Real estate content",
                   layout="wide", page_icon=str(_ICON_PNG) if _ICON_PNG.exists() else None)
repository.init_db()
blog_repository.init_db()
video_repository.init_db()

# Colours, fonts and radii come from .streamlit/config.toml; this adds the
# few touches a theme cannot express.
st.markdown("""
<style>
  :root { --fc-ink: #1C1733; --fc-muted: #6B6880; --fc-violet: #5B33F0; --fc-blue: #2F74FF;
          --fc-lav: #F4F1FE; --fc-line: #E6E1F7;
          --fc-grad: linear-gradient(135deg, #6A3DF5 0%, #4A55F7 55%, #2F74FF 100%); }

  /* canvas: white, with a faint lavender wash under the header */
  .stApp { background: radial-gradient(1100px 360px at 50% -140px, #EEE8FF 0%, rgba(238,232,255,0) 72%), #FFFFFF; }
  [data-testid="stHeader"] { background: rgba(255,255,255,.72); backdrop-filter: blur(10px); }
  .block-container { padding-top: 4.5rem; max-width: 1040px; }
  .hint { color: var(--fc-muted); font-size: .85rem; margin-top: -.4rem; }
  [data-testid="stCaptionContainer"] { color: var(--fc-muted); }

  /* brand header */
  .fc-brand { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
              gap: .75rem 1.5rem; padding: .25rem 0 1.4rem; margin-bottom: 1.6rem;
              border-bottom: 1px solid var(--fc-line); }
  .fc-logo { height: 58px; width: auto; display: block; }
  .fc-tag { font-size: .95rem; color: var(--fc-muted); line-height: 1.45; max-width: 25rem; text-align: right; }
  .fc-tag b { color: var(--fc-ink); font-weight: 600; }
  [data-testid="stMarkdownContainer"] p.fc-eyebrow { font-size: .74rem !important; font-weight: 700;
      letter-spacing: .2em; color: var(--fc-violet); text-transform: uppercase; margin: 0 0 .55rem; }

  /* Brochure / Blog post: a segmented control */
  .st-key-solution [role="radiogroup"] { display: inline-flex; gap: 4px; padding: 5px;
      background: var(--fc-lav); border: 1px solid var(--fc-line); border-radius: 14px; }
  .st-key-solution [role="radiogroup"] > div { margin: 0 !important; }
  .st-key-solution [data-testid="stRadioOption"] { margin: 0 !important; padding: .55rem 1.4rem;
      border-radius: 10px; cursor: pointer; transition: background .18s ease, box-shadow .18s ease; }
  .st-key-solution [data-testid="stRadioOption"] > div > div:first-child { display: none; }
  .st-key-solution [data-testid="stRadioOption"] p { font-weight: 500; color: var(--fc-muted); }
  .st-key-solution [data-testid="stRadioOption"]:hover p { color: var(--fc-ink); }
  .st-key-solution [data-testid="stRadioOption"][data-selected="true"] { background: #FFFFFF;
      box-shadow: 0 1px 2px rgba(28,23,51,.06), 0 6px 16px -6px rgba(91,51,240,.3); }
  .st-key-solution [data-testid="stRadioOption"][data-selected="true"] p { color: var(--fc-violet); font-weight: 600; }

  /* each form is one card */
  [data-testid="stForm"] { background: #FFFFFF; border: 1px solid #ECE8F8; border-radius: 22px;
      padding: 1.9rem 2rem 1.6rem; box-shadow: 0 1px 2px rgba(28,23,51,.04), 0 18px 48px -22px rgba(91,51,240,.22); }
  [data-testid="stForm"] h3 { font-size: 1.12rem; letter-spacing: -.01em; padding-top: .6rem; }
  [data-testid="stForm"] h3::before { content: ""; display: inline-block; width: 9px; height: 9px;
      margin: 0 .6rem .12rem 0; border-radius: 50%; background: var(--fc-grad); vertical-align: middle; }

  /* buttons */
  button[kind="primary"], button[kind="primaryFormSubmit"] { background: var(--fc-grad) !important;
      border: none !important; color: #FFFFFF !important; font-weight: 600; padding: .62rem 1.6rem;
      box-shadow: 0 10px 24px -10px rgba(91,51,240,.65); transition: transform .15s ease, box-shadow .15s ease; }
  button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { transform: translateY(-1px);
      box-shadow: 0 14px 28px -10px rgba(91,51,240,.7); }
  button[kind="secondary"], button[kind="secondaryFormSubmit"] { border-color: var(--fc-line); }
  button[kind="secondary"]:hover { border-color: var(--fc-violet); color: var(--fc-violet); }

  /* uploads, expanders, tabs */
  [data-testid="stFileUploaderDropzone"] { background: #FBFAFF; border: 1.5px dashed #CFC4F7; border-radius: 14px; }
  [data-testid="stFileUploaderDropzone"]:hover { border-color: #8B6CF6; }
  [data-testid="stExpander"] details { background: #FCFBFF; border-color: var(--fc-line); border-radius: 14px; }
  [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--fc-line); }
  [data-baseweb="tab"] p { font-weight: 600; }

  .fc-footer { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--fc-line);
               display: flex; align-items: center; gap: .55rem; color: var(--fc-muted); font-size: .85rem; }
  .fc-footer img { height: 20px; width: 20px; }

  @media (max-width: 640px) {
    .fc-logo { height: 46px; }
    .fc-tag { text-align: left; font-size: .88rem; }
    [data-testid="stForm"] { padding: 1.3rem 1.1rem 1.1rem; border-radius: 18px; }
    .st-key-solution div[role="radiogroup"] > label { padding: .5rem 1rem; }
  }
</style>
""", unsafe_allow_html=True)


def _data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" if path.exists() else ""

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


# ---------------------------------------------------------------------------
# Streamlit version compatibility: "fill the container" has been spelled
# three ways across releases, and each is a TypeError on the others.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _stretch_kwargs_for(param_names: tuple[str, ...], width_default_is_str: bool) -> dict:
    if "width" in param_names and width_default_is_str:
        return {"width": "stretch"}
    if "use_container_width" in param_names:
        return {"use_container_width": True}
    if "use_column_width" in param_names:
        return {"use_column_width": True}
    return {}


def _stretch(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    width = params.get("width")
    return _stretch_kwargs_for(tuple(params), isinstance(getattr(width, "default", None), str))


# ---------------------------------------------------------------------------
# The five result tabs, shared by both solutions
# ---------------------------------------------------------------------------

AGENTS = {
    # brochure
    "property_intelligence": ("Property Intelligence", "Collects every fact the brochure may state."),
    "vision": ("Vision", "Reads each picture: what it shows, its quality, any text inside it."),
    "asset_prep": ("Asset Preparation", "Prepares pictures for print without cropping them."),
    "visual_enrichment": ("Visual Enrichment", "Adds supporting mood visuals only where they help."),
    "creative_director": ("Creative Director", "Chooses a readable colour palette for the category."),
    "art_director": ("Art Director", "Plans the pages and places every picture once, whole."),
    "copywriter": ("Copywriter", "Writes titles and short benefit-led text for each page."),
    "render_qa": ("Render & QA", "Renders the brochure, measures every page and refines it."),
    "critic": ("Critic", "Scores the finished brochure."),
    # blog
    "research": ("Research", "Gathers verifiable facts about the topic and place."),
    "planner": ("Planner", "Chooses the article type and outline."),
    "visual": ("Visual Director", "Chooses or creates the images."),
    "writer": ("Writer", "Writes the article."),
    "fact_check": ("Fact Check", "Removes claims that cannot be supported."),
    "editor": ("Editor", "Fixes formatting and readability."),
    "seo": ("SEO", "Writes the title, meta description and keywords."),
    "render": ("Renderer", "Lays the article out as a website-style PDF."),
    "revise": ("Revision", "Sends the critic's feedback back for another pass."),
}

_TIER_STYLE = {"Excellent": "success", "Very Good": "success", "Good": "info",
               "Needs Improvement": "warning"}


def _friendly_failure(kind: str) -> None:
    st.error(f"We couldn't finish this {kind}. Please try again — if it keeps happening, "
             "check that the pictures open correctly and the required fields are filled in.")


def _show_pdf(pdf_path: str | None, filename: str, kind: str) -> None:
    path = Path(pdf_path) if pdf_path else None
    if not path or not path.exists() or not is_pdf_render_valid(path):
        _friendly_failure(kind)
        return
    data = path.read_bytes()
    st.download_button("⬇️ Download PDF", data, file_name=filename, mime="application/pdf", type="primary")
    b64 = base64.b64encode(data).decode("ascii")
    st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="820" '
                'style="border:1px solid #E5E7EB;border-radius:8px;"></iframe>', unsafe_allow_html=True)


def _score_banner(score: float) -> None:
    tier = quality_tier(score)
    getattr(st, _TIER_STYLE.get(tier, "info"))(f"**{tier}** — quality score **{score:.0f}/100**")


def _brochure_evaluation(record) -> None:
    e = record.evaluation
    if e is None:
        st.info("No evaluation is available for this brochure.")
        return
    _score_banner(e.quality_score)
    cols = st.columns(4)
    cols[0].metric("Content", f"{e.content_quality:.0f}")
    cols[1].metric("Visual design", f"{e.visual_design_quality:.0f}")
    cols[2].metric("Factual consistency", f"{e.factual_consistency:.0f}")
    cols[3].metric("Layout integrity", f"{e.layout_integrity:.0f}" if e.layout_integrity is not None else "—")
    cols = st.columns(4)
    cols[0].metric("Image integrity", f"{e.image_integrity:.0f}" if e.image_integrity is not None else "—")
    cols[1].metric("Average white space", f"{e.whitespace_mean:.0%}" if e.whitespace_mean is not None else "—")
    cols[2].metric("Pages", record.page_count or "—")
    cols[3].metric("Template", record.template or "—")
    if e.checks:
        from src.evaluation.brochure_checks import LABELS

        st.markdown("**Checks**")
        for name, result in e.checks.items():
            ok = result == "pass"
            st.write(f"{'✅' if ok else '⚠️'} {LABELS.get(name, name.replace('_', ' ').capitalize())}"
                     f"{'' if ok else ' — needs attention'}")
    reasons = [(label, text) for label, text in (
        ("Content", e.content_quality_reasoning), ("Visual design", e.visual_design_quality_reasoning),
        ("Factual consistency", e.factual_consistency_reasoning)) if text]
    if reasons:
        with st.expander("Why these scores"):
            for label, text in reasons:
                st.markdown(f"**{label}** — {text}")


def _blog_evaluation(record) -> None:
    e = record.evaluation
    if e is None:
        st.info("No evaluation is available for this post.")
        return
    _score_banner(e.quality_score)
    cols = st.columns(3)
    cols[0].metric("Words", record.word_count or "—")
    cols[1].metric("Article type", (record.archetype or "—").replace("_", " ").capitalize())
    cols[2].metric("PDF pages", record.pdf_pages or "—")
    if e.dimensions:
        # The five parameters, their weights and a one-line reason each; the
        # overall score above is their weighted sum (configs/blog/evaluation.yaml).
        rows = ["| Parameter | Weight | Score | Why |", "|---|---:|---:|---|"]
        for dim in e.dimensions:
            reason = (dim.summary or (dim.findings[0] if dim.findings else "")).replace("|", "/")
            rows.append(f"| {'✅' if dim.passed else '⚠️'} {dim.label or dim.name} | {dim.weight:.0%} | "
                        f"{dim.score:.0f} | {reason} |")
        st.markdown("\n".join(rows))
        st.caption("Overall = " + " + ".join(f"{d.weight:.0%} × {d.score:.0f}" for d in e.dimensions)
                   + f" = {e.quality_score:.0f}")
        weak = [d for d in e.dimensions if not d.passed and d.suggestions]
        if weak:
            with st.expander("How to improve"):
                for d in weak:
                    st.markdown(f"**{d.label}** — {d.suggestions[0]}")
    if record.sources:
        with st.expander(f"Sources ({len(record.sources)})"):
            for url in record.sources:
                st.write(f"- {url}")


def _workflow(record) -> None:
    we = record.workflow_evaluation
    if we is None or not we.steps:
        st.info("No workflow steps were recorded.")
        return
    total = sum(s.latency_ms for s in we.steps) / 1000
    if we.all_succeeded:
        st.success(f"All {len(we.steps)} steps completed in {total:.0f} s.")
    else:
        st.warning(f"{len(we.failed_steps)} step(s) did not complete.")
    for step in we.steps:
        title = AGENTS.get(step.node, (step.node.replace("_", " ").title(), ""))[0]
        icon = "✅" if step.status == "success" else "❌"
        st.write(f"{icon} **{title}** — {step.latency_ms / 1000:.1f} s")


def _human_feedback(record, repo, key: str) -> None:
    hr = record.human_review
    label = {"pending": "🕓 Awaiting review", "approved": "✅ Approved",
             "revision_requested": "🔁 Revision requested"}.get(hr.status, hr.status)
    st.write(f"**Status:** {label}")
    if hr.comment:
        st.caption(f"Last comment: {hr.comment}")
    comment = st.text_area("Your feedback", key=f"{key}_comment", height=90)
    a, b = st.columns(2)
    if a.button("✅ Approve", key=f"{key}_approve", **_stretch(a.button)):
        repo.update_human_review(record.generation_id, status="approved", comment=comment)
        hr.status, hr.comment = "approved", comment
        st.rerun()
    if b.button("🔁 Request revision", key=f"{key}_revise", **_stretch(b.button)):
        repo.update_human_review(record.generation_id, status="revision_requested", comment=comment)
        hr.status, hr.comment = "revision_requested", comment
        st.rerun()


def _result_tabs(record, kind: str) -> None:
    pdf_tab, eval_tab, flow_tab, review_tab = st.tabs(
        ["Final PDF", "Evaluation", "Workflow", "Human feedback"])
    failed = record.status == "failed"
    with pdf_tab:
        if failed:
            _friendly_failure(kind)
        else:
            name = slugify(getattr(record, "title", "") or getattr(record, "property_id", "")
                           or getattr(record, "blog_id", "") or kind, kind)
            _show_pdf(record.pdf_path, f"{name}.pdf", kind)
    with eval_tab:
        (_brochure_evaluation if kind == "brochure" else _blog_evaluation)(record)
    with flow_tab:
        _workflow(record)
    with review_tab:
        repo = repository if kind == "brochure" else blog_repository
        _human_feedback(record, repo, f"{kind}_{record.generation_id}")


def _uploads(files) -> list[tuple[str, bytes]]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    return [(f.name, bytes(f.getbuffer())) for f in files]


def _lines(text: str) -> list[str]:
    return [line.strip(" •-\t") for line in (text or "").splitlines() if line.strip(" •-\t")]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_logo = _data_uri(_LOGO_PNG)
st.markdown(f"""
<div class="fc-brand" role="banner">
  {f'<img class="fc-logo" src="{_logo}" alt="{PRODUCT_NAME}">' if _logo else f'<h1>{PRODUCT_NAME}</h1>'}
  <div class="fc-tag"><b>AI-drafted brochures and blogs</b> for real estate teams, ready for your review.</div>
</div>
<p class="fc-eyebrow">What would you like to create?</p>
""", unsafe_allow_html=True)
if "video" in st.query_params:      # a link to a video opens the Video tab
    st.session_state.setdefault("solution", "Video")
solution = st.radio("What would you like to create?", ["Brochure", "Blog post", "Video"], horizontal=True,
                    label_visibility="collapsed", key="solution")

# =============================================================================
# Brochure
# =============================================================================

if solution == "Brochure":
    st.caption("Fields marked * are required. Everything else is optional and used only when given.")

    with st.form("brochure_form", border=False):
        st.subheader("The project")
        c1, c2 = st.columns(2)
        name = c1.text_input("1 · Project name *", key="f_name")
        city = c2.text_input("2 · City *", key="f_city")
        c1, c2 = st.columns(2)
        locality = c1.text_input("Locality", key="f_locality", placeholder="e.g. Kokapet")
        property_type = c2.selectbox("3 · Property type", property_form.PROPERTY_TYPES, key="f_type")
        c1, c2, c3 = st.columns([1.2, 1, 1.4])
        category = c1.selectbox("4 · Category *", property_form.CATEGORIES, index=None,
                                placeholder="Choose…", key="f_category")
        orientation = c2.radio("Format", ["Landscape", "Portrait"], horizontal=True,
                               key="f_orientation", help="A4 landscape or portrait")
        price = c3.text_input("5 · Starting price / range", key="f_price",
                              placeholder="e.g. ₹52 Lakh onwards")

        st.subheader("Homes")
        st.markdown('<div class="hint">6 · One row per configuration. Carpet area and price are optional.</div>',
                    unsafe_allow_html=True)
        config_rows = st.data_editor(
            [{"Configuration": "", "Carpet area": "", "Price": ""}],
            num_rows="dynamic", key="f_configs", hide_index=True, **_stretch(st.data_editor),
            column_config={
                "Configuration": st.column_config.TextColumn(help="e.g. 2 BHK"),
                "Carpet area": st.column_config.TextColumn(help="e.g. 650 – 720 sq ft"),
                "Price": st.column_config.TextColumn(help="e.g. ₹52 Lakh onwards"),
            })

        st.subheader("Why buy here")
        c1, c2 = st.columns(2)
        usps = c1.text_area("7 · Key selling points (one per line)", key="f_usps", height=130)
        amenities = c2.text_area("8 · Amenities and facilities (one per line)", key="f_amenities", height=130)
        audience = st.text_input("9 · Who is it for?", key="f_audience",
                                 placeholder="e.g. young families and first-time buyers")

        st.subheader("Developer and contact")
        c1, c2 = st.columns(2)
        developer = c1.text_input("10 · Developer name", key="f_developer")
        logo = c2.file_uploader("Developer logo", type=IMAGE_TYPES, key="f_logo")
        c1, c2, c3 = st.columns(3)
        phone = c1.text_input("11 · Phone", key="f_phone")
        email = c2.text_input("Email", key="f_email")
        website = c3.text_input("Website", key="f_website")
        cta = st.text_input("Call to action", key="f_cta", placeholder="e.g. Book a site visit this weekend")

        st.subheader("Pictures")
        photos = st.file_uploader("Project photographs * (exterior, interiors, amenities)", type=IMAGE_TYPES,
                                  accept_multiple_files=True, key="f_photos")
        c1, c2, c3 = st.columns(3)
        plans = c1.file_uploader("Floor plans", type=IMAGE_TYPES, accept_multiple_files=True, key="f_plans")
        master = c2.file_uploader("Master plan", type=IMAGE_TYPES, accept_multiple_files=True, key="f_master")
        location_map = c3.file_uploader("Location map", type=IMAGE_TYPES, key="f_map")

        with st.expander("More details (optional)"):
            c1, c2 = st.columns(2)
            tagline = c1.text_input("Tagline", key="f_tagline")
            possession = c2.text_input("Possession", key="f_possession", placeholder="e.g. December 2027")
            c1, c2 = st.columns(2)
            rera = c1.text_input("RERA registration number", key="f_rera")
            size = c2.text_input("Project size", key="f_size", placeholder="e.g. 4 towers · 620 homes · 6 acres")
            nearby = st.text_area("Nearby places — one per line: name | distance | type", key="f_nearby",
                                  height=100, placeholder="Kokapet Metro | 1.2 km | transit")
            description = st.text_area("Anything else about the project", key="f_description", height=80)

        submitted = st.form_submit_button("Generate brochure", type="primary")

    if submitted:
        rows = config_rows if isinstance(config_rows, list) else config_rows.to_dict("records")
        form = property_form.BrochureForm(
            name=name, city=city, locality=locality, property_type=property_type or "",
            category=category or "", orientation=orientation, price=price,
            configurations=[property_form.ConfigRow(configuration=str(r.get("Configuration") or ""),
                                                    carpet_area=str(r.get("Carpet area") or ""),
                                                    price=str(r.get("Price") or "")) for r in rows],
            usps=usps, amenities=amenities, target_audience=audience, developer=developer,
            phone=phone, email=email, website=website, cta=cta, tagline=tagline,
            possession=possession, rera_number=rera, project_size=size, nearby=nearby,
            description=description,
        )
        gaps = property_form.missing(form, len(photos or []))
        if gaps:
            st.warning("Please add " + ", ".join(gaps) + ".")
        else:
            uploads = {"photo": _uploads(photos), "floorplan": _uploads(plans),
                       "masterplan": _uploads(master), "map": _uploads(location_map), "logo": _uploads(logo)}
            try:
                folder = property_form.write_property(form, uploads, settings.data_dir_resolved / "properties")
                template = property_form.template_for(form)
                with st.spinner(f"Designing your {catalog.resolve(template).name} brochure…"):
                    st.session_state["brochure_record"] = run_brochure_workflow(folder, template_name=template)
            except Exception:  # noqa: BLE001 - details to the log, a plain message to the person
                logger.exception("Brochure generation failed before the workflow finished")
                st.session_state.pop("brochure_record", None)
                _friendly_failure("brochure")

    record = st.session_state.get("brochure_record")
    if record is not None:
        st.divider()
        _result_tabs(record, "brochure")

# =============================================================================
# Blog post
# =============================================================================

elif solution == "Blog post":
    st.caption("Only the topic is required. Facts you supply are the only figures the post will state "
               "as yours; anything researched is cited.")
    with st.form("blog_form", border=False):
        topic = st.text_input("Topic *", key="bl_topic",
                              placeholder="e.g. Is Kokapet a good place to buy a home in 2026?")
        c1, c2, c3 = st.columns(3)
        location = c1.text_input("Location", key="bl_location", placeholder="e.g. Kokapet, Hyderabad")
        audience = c2.selectbox("Audience", ["Homebuyers", "First-time buyers", "Families", "Investors",
                                             "NRI buyers", "Retirees"], key="bl_audience")
        tone = c3.selectbox("Tone", ["Professional", "Conversational", "Persuasive", "Friendly"], key="bl_tone")
        c1, c2 = st.columns(2)
        keyword = c1.text_input("Primary keyword", key="bl_keyword")
        words = c2.slider("Length (words)", 600, 2500, 1400, 100, key="bl_words")
        key_points = st.text_area("Key points to cover (one per line)", key="bl_points", height=90)
        facts = st.text_area("Facts you can vouch for (one per line)", key="bl_facts", height=90,
                             placeholder="Kokapet Metro station 1.2 km away")
        with st.expander("Project, brand and call to action (optional)"):
            c1, c2 = st.columns(2)
            project = c1.text_input("Project name", key="bl_project")
            brand = c2.text_input("Brand / developer", key="bl_brand")
            cta_text = st.text_input("Call to action", key="bl_cta", placeholder="Book a site visit")
            c1, c2, c3 = st.columns(3)
            cta_url = c1.text_input("Link", key="bl_cta_url", placeholder="https://…")
            cta_phone = c2.text_input("Phone", key="bl_cta_phone")
            cta_email = c3.text_input("Email", key="bl_cta_email")
            images = st.file_uploader("Images (optional)", type=IMAGE_TYPES, accept_multiple_files=True,
                                      key="bl_images")
        submitted = st.form_submit_button("Generate blog post", type="primary")

    if submitted:
        if not topic.strip():
            st.warning("Please add a topic.")
        else:
            blog_id = slugify(topic, "BLOG")
            blog_dir = settings.data_dir_resolved / "blogs" / blog_id
            try:
                (blog_dir / "images").mkdir(parents=True, exist_ok=True)
                for old in (blog_dir / "images").iterdir():
                    if old.is_file():
                        old.unlink()
                for filename, payload in _uploads(images):
                    (blog_dir / "images" / Path(filename).name).write_bytes(payload)
                brief = {
                    "blog_id": blog_id, "topic": topic.strip(), "location": location.strip(),
                    "audience": "" if audience == "Homebuyers" else audience.lower(),
                    "tone": tone.lower(), "primary_keyword": keyword.strip(),
                    "target_word_count": int(words), "key_points": _lines(key_points),
                    "local_facts": _lines(facts), "project_name": project.strip(),
                    "brand_name": brand.strip(), "cta_text": cta_text.strip(), "cta_url": cta_url.strip(),
                    "cta_phone": cta_phone.strip(), "cta_email": cta_email.strip(),
                }
                (blog_dir / "blog_input.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
                with st.spinner("Researching and writing your post…"):
                    st.session_state["blog_record"] = run_blog_workflow(blog_dir)
            except Exception:  # noqa: BLE001
                logger.exception("Blog generation failed before the workflow finished")
                st.session_state.pop("blog_record", None)
                _friendly_failure("blog post")

    record = st.session_state.get("blog_record")
    if record is not None:
        st.divider()
        _result_tabs(record, "blog post")

# =============================================================================
# Video
# =============================================================================

else:
    render_video_tab()

# =============================================================================
# Footer
# =============================================================================

_mark = _data_uri(_ICON_PNG)
st.markdown(f"""
<div class="fc-footer">{f'<img src="{_mark}" alt="">' if _mark else ''}
  <span><b>{PRODUCT_NAME}</b> · AI-drafted content for real estate teams. Every draft is reviewed by a person before use.</span>
</div>
""", unsafe_allow_html=True)
