"""Node 8: on-screen text as transparent pictures, kept in the safe zone, and the captions file."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.agents.video import brief as brief_agent
from src.agents.video import overlays
from src.data.video_schemas import BrandKit, LineTiming, ScriptLine, Shot, Storyboard, VideoBrief, VideoScript
from src.rendering import video_overlay
from src.rendering.video_overlay import OverlaySpec
from tests.video.helpers import make_video_property


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            return Path(pw.chromium.executable_path).exists()
    except Exception:  # noqa: BLE001
        return False


needs_chromium = pytest.mark.skipif(not _chromium_available(), reason="Playwright Chromium is not installed")

PALETTE = {"paper": "#F6F1EA", "ink": "#2B2118", "accent": "#8A5A36", "accent_ink": "#FFFFFF",
           "dark": "#2A1F18", "on_dark": "#FBF6EE", "on_dark_soft": "#CDBFB0"}


def _spec(kind="title", aspect="9:16", size=(540, 960), **fields) -> OverlaySpec:
    base = dict(name=f"overlay_{kind}", kind=kind, width=size[0], height=size[1], category="mid", palette=PALETTE,
                safe=video_overlay.safe_zone(aspect))
    base.update(fields)
    return OverlaySpec(**base)


HOOK = dict(heading="Aurelia Heights", sub=["From ₹85 Lakh"], chips=[])
CHIP = dict(heading="", sub=[], chips=["Swimming pool", "Tennis/pickleball court"])
END = dict(kind="end_card", heading="Aurelia", cta="Book a site visit", legal=["RERA: P52100012345"],
           contact=["+91 98765 43210", "https://northlight.example"], has_logo=True)


# -- the safe zone ---------------------------------------------------------------------------------

def test_the_safe_zone_leaves_the_platform_bands_clear():
    assert video_overlay.safe_zone("9:16") == {"top": 0.14, "bottom": 0.20, "left": 0.06, "right": 0.06}
    assert video_overlay.safe_rect(1080, 1920, video_overlay.safe_zone("9:16")) == (65, 269, 1015, 1536)
    x0, y0, x1, y1 = video_overlay.safe_rect(1920, 1080, video_overlay.safe_zone("16:9"))
    assert (x0, y0, x1, y1) == (115, 76, 1805, 950)


def test_text_outside_the_zone_is_reported_by_name():
    spec = _spec()
    inside = {"x": 40, "y": 400, "w": 200, "h": 40, "text": "Fine"}
    off_right = {"x": 400, "y": 400, "w": 200, "h": 40, "text": "Too wide"}
    in_bottom_band = {"x": 40, "y": 800, "w": 100, "h": 100, "text": "Under the controls"}
    assert video_overlay.check_rects(spec, [inside]) == []
    assert video_overlay.check_rects(spec, [off_right, in_bottom_band]) == [
        "'Too wide' is outside the safe zone", "'Under the controls' is outside the safe zone"]


# -- which overlay a shot needs ---------------------------------------------------------------------

def _script(**fields) -> VideoScript:
    base = dict(lines=[ScriptLine(text="Hello.", role="hook")], on_screen=["Aurelia Heights", "From ₹85 Lakh"],
                chips=["Swimming pool"], chip_lines=[0], cta="Book a site visit",
                end_card=["RERA: P52100012345", "+91 98765 43210"])
    base.update(fields)
    return VideoScript(**base)


def _kit(**fields) -> BrandKit:
    return BrandKit(name="Aurelia", palette=PALETTE, **fields)


def _brief() -> VideoBrief:
    return VideoBrief(property_id="P", name="Aurelia Heights", duration_s=15, width=540, height=960, quality="draft")


def test_the_hook_prints_the_name_and_the_price():
    shot = Shot(index=0, overlay_text=["Aurelia Heights", "From ₹85 Lakh"])
    spec = overlays.spec_for_shot(0, shot, _script(), _brief(), _kit(), False, video_overlay.safe_zone("9:16"))
    assert (spec.kind, spec.heading, spec.sub, spec.chips) == ("title", "Aurelia Heights", ["From ₹85 Lakh"], [])


def test_a_feature_shot_prints_only_its_chip():
    shot = Shot(index=2, overlay_text=["Swimming pool"])
    spec = overlays.spec_for_shot(2, shot, _script(), _brief(), _kit(), False, video_overlay.safe_zone("9:16"))
    assert (spec.kind, spec.heading, spec.chips) == ("title", "", ["Swimming pool"])


def test_the_closing_card_prints_the_cta_rera_and_contact_and_the_name_only_when_there_is_no_logo():
    shot = Shot(index=4, kind="card", camera="still",
                overlay_text=["Book a site visit", "RERA: P52100012345", "+91 98765 43210"])
    spec = overlays.spec_for_shot(4, shot, _script(), _brief(), _kit(logo_path="logo.jpg"), True,
                                  video_overlay.safe_zone("9:16"))
    assert spec.kind == "end_card" and spec.cta == "Book a site visit" and spec.has_logo
    assert spec.legal == ["RERA: P52100012345"] and spec.contact == ["+91 98765 43210"]
    assert overlays.spec_for_shot(4, shot, _script(), _brief(), _kit(), True,
                                  video_overlay.safe_zone("9:16")).has_logo is False


def test_a_shot_with_no_text_needs_no_overlay():
    assert overlays.spec_for_shot(1, Shot(index=1), _script(), _brief(), _kit(), False,
                                  video_overlay.safe_zone("9:16")) is None


def test_a_card_that_is_not_the_last_shot_is_a_title_not_an_end_card():
    shot = Shot(index=0, kind="card", camera="still", overlay_text=["Aurelia Heights"])
    assert overlays.spec_for_shot(0, shot, _script(), _brief(), _kit(), False,
                                  video_overlay.safe_zone("9:16")).kind == "title"


# -- the browser renderer ----------------------------------------------------------------------------

@needs_chromium
@pytest.mark.parametrize("aspect, size", [("9:16", (540, 960)), ("16:9", (960, 540)), ("1:1", (540, 540))])
def test_chromium_draws_every_overlay_transparent_and_inside_the_safe_zone(tmp_path, aspect, size):
    specs = [_spec(aspect=aspect, size=size, name="hook", **HOOK), _spec(aspect=aspect, size=size, name="chips", **CHIP),
             _spec(aspect=aspect, size=size, name="end", **{**END})]
    rendered = video_overlay.render_overlays(specs, tmp_path / "o")

    assert rendered.renderer == "browser" and rendered.notes == []
    assert rendered.problems == [], rendered.problems
    for path in rendered.paths:
        image = Image.open(path)
        assert image.mode == "RGBA" and image.size == size
        alpha = image.getchannel("A")
        assert alpha.getextrema()[0] == 0 and alpha.getextrema()[1] > 200                    # clear, and something drawn
        transparent = sum(1 for a in alpha.getdata() if a == 0) / (size[0] * size[1])
        assert transparent > 0.35


@needs_chromium
def test_chromium_uses_the_brands_own_type_and_draws_the_rupee_sign(tmp_path):
    html = video_overlay.build_html(_spec(**HOOK))
    assert "Playfair Display" in html and "Outfit" in html and "&#39;" not in html.split("</style>")[0]
    assert "₹85 Lakh" in html


@needs_chromium
def test_a_long_name_wraps_inside_the_safe_zone_instead_of_leaving_it(tmp_path):
    long = "The Extraordinarily Long Residences of Northlight Developers on S.G. Highway"
    rendered = video_overlay.render_overlays([_spec(heading=long, sub=[], chips=[])], tmp_path / "o")
    assert rendered.problems == []


@needs_chromium
def test_a_script_other_than_english_draws_in_chromium(tmp_path):
    rendered = video_overlay.render_overlays([_spec(heading="अनामिका हाइ पॉइंट")],
                                             tmp_path / "o")
    assert rendered.renderer == "browser" and rendered.paths[0].stat().st_size > 0


# -- the fallback --------------------------------------------------------------------------------------

@pytest.mark.parametrize("aspect, size", [("9:16", (540, 960)), ("16:9", (960, 540)), ("1:1", (540, 540))])
def test_the_fallback_draws_every_overlay_transparent_and_inside_the_safe_zone(tmp_path, aspect, size):
    specs = [_spec(aspect=aspect, size=size, name="hook", **HOOK), _spec(aspect=aspect, size=size, name="chips", **CHIP),
             _spec(aspect=aspect, size=size, name="end", **END)]
    rendered = video_overlay.render_overlays(specs, tmp_path / "o", browser=False)

    assert rendered.renderer == "fallback" and rendered.problems == [], rendered.problems
    for path in rendered.paths:
        image = Image.open(path)
        assert image.mode == "RGBA" and image.size == size
        extrema = image.getchannel("A").getextrema()
        assert extrema[0] == 0 and extrema[1] > 200


def test_the_fallback_wraps_a_long_name_and_keeps_it_inside_the_zone(tmp_path):
    long = "The Extraordinarily Long Residences of Northlight Developers on S.G. Highway"
    rendered = video_overlay.render_overlays([_spec(heading=long, sub=[], chips=[])], tmp_path / "o", browser=False)
    assert rendered.problems == []


def test_the_fallback_writes_rs_for_the_rupee_sign(tmp_path):
    assert video_overlay._plain("From ₹85 Lakh") == "From Rs 85 Lakh"


def test_when_chromium_cannot_run_the_overlays_are_still_drawn_and_the_note_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(video_overlay, "_WORKER", tmp_path / "no_such_worker.py")
    rendered = video_overlay.render_overlays([_spec(**HOOK)], tmp_path / "o")
    assert rendered.renderer == "fallback" and rendered.paths[0].exists()
    assert rendered.notes and rendered.notes[0].startswith("The overlays were drawn by the fallback renderer")
    assert "English only" in rendered.notes[0]


# -- timing, and the node ----------------------------------------------------------------------------------

def _timings() -> list[LineTiming]:
    return [LineTiming(line_index=0, start_s=0.4, end_s=2.2), LineTiming(line_index=1, start_s=2.6, end_s=5.1),
            LineTiming(line_index=2, start_s=5.5, end_s=7.0)]


def _board() -> Storyboard:
    return Storyboard(shots=[Shot(index=0, seconds=2.4, overlay_text=["Aurelia Heights", "From ₹85 Lakh"]),
                             Shot(index=1, seconds=3.0, overlay_text=["Swimming pool"]),
                             Shot(index=2, seconds=1.0),
                             Shot(index=3, kind="card", camera="still", seconds=2.0,
                                  overlay_text=["Book a site visit", "RERA: P52100012345"])])


def _node_script() -> VideoScript:
    return _script(lines=[ScriptLine(text="Welcome to Aurelia Heights.", role="hook"),
                          ScriptLine(text="On site: swimming pool.", role="feature"),
                          ScriptLine(text="Book a site visit.", role="cta")])


def test_each_overlay_is_timed_to_its_shot_with_the_title_a_beat_late(tmp_path):
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)

    by_shot = {i.shot_index: i for i in result.items}
    assert set(by_shot) == {0, 1, 3}                                              # the shot with no text has no overlay
    assert (by_shot[0].start_s, by_shot[0].end_s) == (0.35, 2.25)                 # 0.35 s after the cut, 0.15 s before the next
    assert (by_shot[1].start_s, by_shot[1].end_s) == (2.6, 5.25)                  # 2.4 + 0.2 ... 5.4 - 0.15
    assert (by_shot[3].start_s, by_shot[3].end_s) == (6.4, 8.4)                   # the card holds to the very end
    assert by_shot[3].kind == "end_card" and by_shot[0].kind == "title"
    assert all(Path(i.path).exists() and i.renderer == "fallback" for i in result.items)
    assert by_shot[3].texts == ["Book a site visit", "RERA: P52100012345"]


def test_no_defect_when_the_text_fits_the_zone(tmp_path):
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)
    assert result.defects == []


def test_a_zone_violation_becomes_a_defect(tmp_path, monkeypatch):
    real = video_overlay.check_rects
    monkeypatch.setattr(video_overlay, "check_rects", lambda spec, rects: ["'Aurelia Heights' is outside the safe zone"])
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)
    assert any(d.startswith("On-screen text outside the safe zone: overlay_00:") for d in result.defects)
    monkeypatch.setattr(video_overlay, "check_rects", real)


def test_a_render_that_fails_outright_is_a_defect_not_a_failed_run(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(video_overlay, "render_overlays", boom)
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)
    assert result.items == [] and result.defects == ["The on-screen text could not be drawn: disk full"]
    assert Path(result.captions_path).exists()


def test_the_overlays_come_from_the_property_through_the_brief(tmp_path):
    loaded = brief_agent.run(make_video_property(tmp_path), draft=True)
    assert video_overlay.safe_zone(loaded.brief.aspect)["bottom"] == 0.20


# -- the captions ---------------------------------------------------------------------------------------

def test_srt_times_are_hours_minutes_seconds_and_milliseconds():
    assert overlays.srt_time(0) == "00:00:00,000"
    assert overlays.srt_time(3661.5) == "01:01:01,500"
    assert overlays.srt_time(7.0004) == "00:00:07,000"
    assert overlays.srt_time(59.9996) == "00:01:00,000"


def test_the_captions_follow_the_line_timings_exactly():
    text = overlays.srt(_timings(), _node_script(), 42)
    blocks = text.strip().split("\n\n")
    assert len(blocks) == 3
    assert blocks[0] == "1\n00:00:00,400 --> 00:00:02,200\nWelcome to Aurelia Heights."
    assert blocks[1].splitlines()[1] == "00:00:02,600 --> 00:00:05,100"
    assert blocks[2].splitlines()[1] == "00:00:05,500 --> 00:00:07,000"
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_a_long_caption_is_wrapped_to_two_lines_of_a_readable_width():
    script = VideoScript(lines=[ScriptLine(text="One of the most carefully planned residential communities on the S.G. "
                                                "Highway, with a clubhouse, pools, courts and gardens.", role="hook")])
    block = overlays.srt([LineTiming(line_index=0, start_s=0, end_s=5)], script, 42).strip().splitlines()
    assert len(block) == 4                                                       # index, times, two rows
    assert len(block[2]) <= 42


def test_captions_are_written_beside_the_video_and_follow_a_silent_track_too(tmp_path):
    timings = [LineTiming(line_index=i, start_s=t.start_s, end_s=t.end_s, mode="silent")
               for i, t in enumerate(_timings())]
    result = overlays.run(_board(), _node_script(), timings, _brief(), _kit(), tmp_path, browser=False)
    assert result.captions_path == str(tmp_path / "captions.srt")
    assert (tmp_path / "captions.srt").read_text(encoding="utf-8").count("-->") == 3
