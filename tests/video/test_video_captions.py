"""Burned captions: one spoken line at a time, inside the safe zone, never over the title, a chip or the card's text."""
from __future__ import annotations

import pytest

from src.agents.video import brief as brief_agent
from src.agents.video import overlays
from src.data.video_schemas import BrandKit, LineTiming, Shot, Storyboard
from src.rendering import video_overlay
from tests.video.helpers import make_video_property
from tests.video.test_video_overlays import _board, _brief, _kit, _node_script, _timings, needs_chromium

ZONE = (269, 1536)          # the safe zone's top and bottom at 1080x1920


# -- where a caption goes -----------------------------------------------------------------------------------------

def test_a_caption_stays_at_its_default_when_nothing_is_in_the_way():
    assert overlays.place_caption(100, (0, 2), [], ZONE, 1200, 20) == 1200


def test_a_title_showing_at_the_same_moment_moves_the_caption_clear_of_it():
    title = (0.0, 3.0, (65, 1150, 600, 90))                               # y 1150 to 1240
    bottom = overlays.place_caption(100, (1, 2), [title], ZONE, 1200, 20)
    assert bottom == 1130                                                  # its lower edge 20 px above the title
    assert bottom <= 1150 - 20


def test_a_title_that_is_not_showing_then_is_ignored():
    title = (5.0, 8.0, (65, 1150, 600, 90))
    assert overlays.place_caption(100, (1, 2), [title], ZONE, 1200, 20) == 1200


def test_the_nearest_clear_place_wins_above_or_below():
    block = (0.0, 3.0, (65, 1100, 900, 300))                               # y 1100 to 1400: a card's text
    assert overlays.place_caption(100, (1, 2), [block], ZONE, 1350, 20) == 1520      # below is nearer than above (1080)
    assert overlays.place_caption(100, (1, 2), [block], ZONE, 1150, 20) == 1080      # above is nearer


def test_with_no_room_inside_the_safe_zone_there_is_no_place():
    wall = (0.0, 3.0, (65, 270, 900, 1265))                                # the whole zone
    assert overlays.place_caption(100, (1, 2), [wall], ZONE, 1200, 20) is None


def test_a_place_that_would_leave_the_safe_zone_is_not_offered():
    block = (0.0, 3.0, (65, 700, 900, 800))                                # leaves 430 above and 36 below
    bottom = overlays.place_caption(100, (1, 2), [block], ZONE, 1100, 20)
    assert bottom == 680 and bottom - 100 >= ZONE[0]


def test_several_things_showing_together_are_all_avoided():
    things = [(0.0, 3.0, (65, 1300, 600, 100)), (0.0, 3.0, (65, 1180, 600, 100))]
    bottom = overlays.place_caption(100, (1, 2), things, ZONE, 1250, 20)
    assert all(bottom <= top - 20 or bottom - 100 >= top + 100 + 20 for top in (1300, 1180))


# -- the caption windows --------------------------------------------------------------------------------------------

def test_a_caption_stays_a_moment_after_its_line_and_never_into_the_next():
    timings = [LineTiming(line_index=0, start_s=0.4, end_s=2.2), LineTiming(line_index=1, start_s=2.3, end_s=5.0),
               LineTiming(line_index=2, start_s=5.5, end_s=7.0)]
    assert overlays._caption_windows(timings) == [(0.4, 2.3), (2.3, 5.15), (5.5, 7.15)]


# -- in the overlay step ----------------------------------------------------------------------------------------------

def _run(tmp_path, **brief_fields):
    brief = _brief().model_copy(update=brief_fields)
    return overlays.run(_board(), _node_script(), _timings(), brief, _kit(), tmp_path, browser=False)


def test_each_spoken_line_is_burned_in_one_at_a_time_from_the_line_timings(tmp_path):
    result = _run(tmp_path)
    captions = [i for i in result.items if i.kind == "caption"]
    assert [i.texts for i in captions] == [["Welcome to Aurelia Heights."], ["On site: swimming pool."], ["Book a site visit."]]
    assert [(i.start_s, i.end_s) for i in captions] == [(0.4, 2.35), (2.6, 5.25), (5.5, 7.15)]
    assert all(a.end_s <= b.start_s for a, b in zip(captions, captions[1:]))               # never two at once
    assert all(i.fade_s == 0.12 and i.slide_px == 0 and i.renderer == "fallback" for i in captions)
    assert [i.shot_index for i in captions] == [0, 1, 2]                                    # the shot each begins in


def test_captions_sit_inside_the_safe_zone(tmp_path):
    result = _run(tmp_path)
    x0, y0, x1, y1 = video_overlay.safe_rect(540, 960, video_overlay.safe_zone("9:16"))
    for item in (i for i in result.items if i.kind == "caption"):
        for r in item.rects:
            assert r["x"] >= x0 - 1.5 and r["x"] + r["w"] <= x1 + 1.5 and r["y"] >= y0 - 1.5 and r["y"] + r["h"] <= y1 + 1.5
    assert result.defects == []


def test_a_caption_never_overlaps_text_that_is_on_screen_at_the_same_time(tmp_path):
    result = _run(tmp_path)
    captions = [i for i in result.items if i.kind == "caption"]
    others = [i for i in result.items if i.kind != "caption" and i.rects]
    checked = 0
    for c in captions:
        for o in others:
            if c.start_s < o.end_s and o.start_s < c.end_s:                               # on screen together
                for cr in c.rects:
                    for orr in o.rects:
                        assert cr["y"] + cr["h"] <= orr["y"] or orr["y"] + orr["h"] <= cr["y"], (c.texts, o.key)
                checked += 1
    assert checked >= 2                          # the hook's title and the closing card really do share time with a caption


def test_the_caption_over_the_hook_is_clear_of_its_title(tmp_path):
    result = _run(tmp_path)
    first = next(i for i in result.items if i.kind == "caption")
    title = [r for i in result.items if i.key in ("heading", "sub0") for r in i.rects]
    cap = first.rects[0]
    assert all(cap["y"] + cap["h"] <= t["y"] or t["y"] + t["h"] <= cap["y"] for t in title)


def test_over_the_closing_card_the_caption_clears_its_text_and_the_logo_plate(tmp_path):
    kit = BrandKit(name="Aurelia", palette=_kit().palette, logo_path=str(make_logo(tmp_path)))
    board = Storyboard(shots=[Shot(index=0, seconds=3.0), Shot(index=1, kind="card", camera="still", seconds=3.0,
                                   overlay_text=["Book a site visit", "RERA: P52100012345", "+91 98765 43210"])])
    timings = [LineTiming(line_index=2, start_s=3.2, end_s=5.2)]
    result = overlays.run(board, _node_script(), timings, _brief(), kit, tmp_path / "o", browser=False)
    caption = next(i for i in result.items if i.kind == "caption")
    text = [r for i in result.items if i.kind == "end_card" for r in i.rects]
    plate = overlays.motion.logo_plate(kit, 540, 960)
    cap = caption.rects[0]
    assert all(cap["y"] + cap["h"] <= t["y"] or t["y"] + t["h"] <= cap["y"] for t in text)
    assert cap["y"] + cap["h"] <= plate[1] or plate[3] <= cap["y"]


def make_logo(tmp_path):
    from tests.brochure import helpers as brochure

    path = tmp_path / "logo.png"
    path.write_bytes(brochure.plan_bytes((400, 400)))
    return path


def test_a_caption_with_no_free_place_is_left_off_the_picture_and_the_note_says_so(tmp_path, monkeypatch):
    real = overlays.place_caption
    monkeypatch.setattr(overlays, "place_caption", lambda *a, **k: None if a[1][0] == 2.6 else real(*a, **k))
    result = _run(tmp_path)
    captions = [i for i in result.items if i.kind == "caption"]
    assert [i.texts for i in captions] == [["Welcome to Aurelia Heights."], ["Book a site visit."]]
    assert any(n.startswith("Caption 2 is not drawn on the picture") and "captions.srt has it" in n for n in result.notes)
    assert (tmp_path / "captions.srt").read_text(encoding="utf-8").count("-->") == 3             # the file still has all three


def test_a_caption_that_had_to_move_is_drawn_again_at_its_new_place(tmp_path, monkeypatch):
    calls = []
    real = video_overlay.render_overlays
    monkeypatch.setattr(video_overlay, "render_overlays", lambda specs, *a, **k: calls.append(specs) or real(specs, *a, **k))
    kit = BrandKit(name="Aurelia", palette=_kit().palette, logo_path=str(make_logo(tmp_path)))
    config = overlays.load_yaml_config
    monkeypatch.setattr(overlays, "load_yaml_config", lambda name: {**config(name), "captions": {
        **config(name)["captions"], "bottom": 0.55}} if name == "video/pacing.yaml" else config(name))   # the card's text is there
    overlays.run(_board(), _node_script(), _timings(), _brief(), kit, tmp_path / "o", browser=False)
    caption_calls = [c for c in calls if c and c[0].kind == "caption"]
    assert len(caption_calls) == 2 and len(caption_calls[0]) == 3 and 0 < len(caption_calls[1]) < 3
    assert all(s.bottom_y != caption_calls[0][0].bottom_y for s in caption_calls[1])


def test_the_srt_is_still_written_beside_the_video_with_captions_burned(tmp_path):
    result = _run(tmp_path)
    assert result.captions_path == str(tmp_path / "captions.srt")
    assert (tmp_path / "captions.srt").read_text(encoding="utf-8").count("-->") == 3


def test_captions_can_be_turned_off_and_the_srt_is_kept(tmp_path):
    result = _run(tmp_path, burn_captions=False)
    assert not [i for i in result.items if i.kind == "caption"]
    assert [i for i in result.items if i.kind != "caption"]                                      # the title and card are unaffected
    assert not list((tmp_path / "overlays").glob("caption_*.png"))
    assert (tmp_path / "captions.srt").read_text(encoding="utf-8").count("-->") == 3


def test_a_silent_track_is_captioned_too(tmp_path):
    silent = [t.model_copy(update={"mode": "silent"}) for t in _timings()]
    result = overlays.run(_board(), _node_script(), silent, _brief(), _kit(), tmp_path, browser=False)
    assert len([i for i in result.items if i.kind == "caption"]) == 3


def test_a_render_that_fails_for_the_captions_alone_leaves_the_rest_and_says_so(tmp_path, monkeypatch):
    real = video_overlay.render_overlays

    def flaky(specs, *a, **k):
        if specs and specs[0].kind == "caption":
            raise RuntimeError("disk full")
        return real(specs, *a, **k)

    monkeypatch.setattr(video_overlay, "render_overlays", flaky)
    result = _run(tmp_path)
    assert [i for i in result.items if i.kind != "caption"] and not [i for i in result.items if i.kind == "caption"]
    assert any("captions could not be drawn on the picture" in n and "captions.srt has them" in n for n in result.notes)
    assert result.defects == []


# -- the option ----------------------------------------------------------------------------------------------------------------

def test_burn_captions_is_on_unless_the_properties_video_input_json_says_otherwise(tmp_path):
    assert brief_agent.run(make_video_property(tmp_path / "a"), draft=True).brief.burn_captions is True
    off = brief_agent.run(make_video_property(tmp_path / "b", video_input={"burn_captions": False}), draft=True)
    assert off.brief.burn_captions is False


def test_the_captions_settings_are_in_pacing_yaml():
    from src.core.config import load_yaml_config

    captions = load_yaml_config("video/pacing.yaml")["captions"]
    assert set(captions) >= {"fade_s", "font_px", "bottom", "gap_px", "width_share"}
    assert 0.5 < captions["bottom"] < 0.8


@needs_chromium
def test_in_the_browser_the_caption_is_measured_and_stays_in_the_zone(tmp_path):
    spec = overlays.OverlaySpec(name="caption_00", kind="caption", width=540, height=960, category="mid", palette={},
                                safe=video_overlay.safe_zone("9:16"), text="Two mini-theatres and virtual gaming area, all on site.",
                                bottom_y=640, font_px=50, width_share=0.88)
    rendered = video_overlay.render_overlays([spec], tmp_path, browser=True)
    (rect,) = rendered.rects[0]
    assert rendered.renderer == "browser" and rendered.problems == []
    assert rect["y"] + rect["h"] == pytest.approx(640, abs=1.5)                              # its lower edge is where it was put
    assert rect["h"] > 40                                                                    # two rows
