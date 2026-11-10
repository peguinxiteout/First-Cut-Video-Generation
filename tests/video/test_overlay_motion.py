"""Titles, chips and card lines come in one by one: each is its own picture, faded and slid in."""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from src.agents.video import overlays
from src.rendering import video_compose, video_overlay
from src.rendering.video_compose import OverlayPart
from tests.video.test_transitions import pacing  # noqa: F401 - the fixture
from tests.video.test_video_compose import H, W, _clip, _compose, _frame
from tests.video.test_video_overlays import (
    CHIP, END, HOOK, _board, _brief, _kit, _node_script, _spec, _timings, needs_chromium,
)
from src.rendering.video_compose import ClipPart


# -- which pieces a picture has -----------------------------------------------------------------------------

def test_a_title_is_a_scrim_the_name_the_price_and_the_chips_in_that_order():
    spec = _spec(heading="Aurelia Heights", sub=["From ₹85 Lakh"], chips=["Pool", "Court"])
    assert video_overlay.parts(spec) == ["scrim", "heading", "sub0", "chip0", "chip1"]
    assert video_overlay.parts(_spec(**CHIP)) == ["scrim", "chip0", "chip1"]


def test_the_closing_card_is_its_wordmark_the_call_to_action_then_the_small_print_and_contact():
    assert video_overlay.parts(_spec(**{**END, "has_logo": False})) == [
        "wordmark", "cta", "legal0", "contact0", "contact1"]
    assert video_overlay.parts(_spec(**END)) == ["cta", "legal0", "contact0", "contact1"]      # the logo is the plate, not text


def test_a_caption_is_one_piece():
    assert video_overlay.parts(_spec(kind="caption", text="Hello")) == ["caption"]


# -- a piece is drawn where the whole page would put it -------------------------------------------------------------

def _alpha(path):
    return Image.open(path).getchannel("A").point(lambda v: 255 if v else 0)


def test_each_piece_is_drawn_alone_and_together_they_make_the_whole_page(tmp_path):
    spec = _spec(heading="Aurelia Heights", sub=["From ₹85 Lakh"], chips=["Pool"])
    full = video_overlay.render_overlays([spec], tmp_path / "full", browser=False)
    keys = video_overlay.parts(spec)
    pieces = video_overlay.render_overlays([_spec(heading=spec.heading, sub=spec.sub, chips=spec.chips, show=k,
                                                  name=f"p_{k}") for k in keys], tmp_path / "pieces", browser=False)
    union = Image.new("L", (spec.width, spec.height), 0)
    for path, key in zip(pieces.paths, keys):
        mask = _alpha(path)
        assert mask.getbbox() is not None, key
        union.paste(255, mask=mask)
    assert union.tobytes() == _alpha(full.paths[0]).tobytes()


def test_a_pieces_text_lands_exactly_where_it_does_on_the_whole_page(tmp_path):
    spec = _spec(heading="Aurelia Heights", sub=["From ₹85 Lakh"], chips=["Pool"])
    full = video_overlay.render_overlays([spec], tmp_path / "full", browser=False).rects[0]
    pieces = video_overlay.render_overlays(
        [_spec(heading=spec.heading, sub=spec.sub, chips=spec.chips, show=k, name=f"p_{k}") for k in ("heading", "chip0")],
        tmp_path / "pieces", browser=False).rects
    assert pieces[0] == [r for r in full if r["text"] == "Aurelia Heights"]
    assert pieces[1] == [r for r in full if r["text"] == "Pool"]


def test_a_piece_with_no_text_measures_nothing(tmp_path):
    rects = video_overlay.render_overlays([_spec(heading="Aurelia", show="scrim")], tmp_path, browser=False).rects
    assert rects == [[]]


@pytest.mark.slow
@needs_chromium
def test_in_the_browser_too_a_piece_sits_where_the_page_puts_it_and_hidden_text_is_not_measured(tmp_path):
    spec = _spec(heading="Aurelia Heights", sub=["From ₹85 Lakh"], chips=["Swimming pool"])
    full = video_overlay.render_overlays([spec], tmp_path / "full", browser=True)
    pieces = video_overlay.render_overlays(
        [_spec(heading=spec.heading, sub=spec.sub, chips=spec.chips, show=k, name=f"p_{k}") for k in ("heading", "sub0", "chip0")],
        tmp_path / "pieces", browser=True)
    assert full.renderer == pieces.renderer == "browser"
    assert [len(r) for r in pieces.rects] == [1, 1, 1]
    for found, text in zip(pieces.rects, ("Aurelia Heights", "From ₹85 Lakh", "Swimming pool")):
        match = next(r for r in full.rects[0] if r["text"] == text)
        assert all(abs(found[0][k] - match[k]) <= 1.0 for k in ("x", "y", "w", "h"))


# -- when each piece comes in ---------------------------------------------------------------------------------------

def test_pieces_come_in_one_stagger_apart_with_the_scrim_beside_the_first():
    keys = ["scrim", "heading", "sub0", "chip0"]
    assert overlays._offsets(keys, 0.15, 2.0) == [0.0, 0.0, 0.15, 0.30]


def test_the_last_piece_is_never_later_than_half_the_window():
    keys = ["cta", "legal0", "contact0", "contact1"]
    assert overlays._offsets(keys, 0.3, 1.5) == pytest.approx([0.0, 0.25, 0.5, 0.75])      # 0.3 would have been 0.9
    assert overlays._offsets(["heading"], 0.15, 2.0) == [0.0]


def test_the_hooks_name_then_its_price_come_in_in_sequence_and_leave_together(tmp_path):
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)
    hook = {i.key: i for i in result.items if i.shot_index == 0 and i.kind == "title"}
    assert [(k, hook[k].start_s) for k in ("scrim", "heading", "sub0")] == [("scrim", 0.35), ("heading", 0.35), ("sub0", 0.5)]
    assert {i.end_s for i in hook.values()} == {2.25}
    assert hook["scrim"].slide_px == 0 and hook["heading"].slide_px == hook["sub0"].slide_px == 20     # 40 px at 1080, 20 at 540
    assert hook["heading"].fade_s == 0.3


def test_the_end_cards_lines_appear_in_sequence(tmp_path):
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)
    card = [i for i in result.items if i.kind == "end_card"]
    assert [(i.key, i.start_s) for i in card] == [("wordmark", 6.4), ("cta", 6.7), ("legal0", 7.0)]
    assert {i.end_s for i in card} == {8.4} and all(i.slide_px == 20 for i in card)


def test_every_piece_carries_the_words_it_prints(tmp_path):
    result = overlays.run(_board(), _node_script(), _timings(), _brief(), _kit(), tmp_path, browser=False)
    said = {i.key: i.texts for i in result.items if i.kind != "caption" and i.shot_index == 0}
    assert said == {"scrim": [], "heading": ["Aurelia Heights"], "sub0": ["From ₹85 Lakh"]}
    assert all(i.rects for i in result.items if i.kind != "caption" and i.key != "scrim")


# -- in the picture ------------------------------------------------------------------------------------------------------

def _box_overlay(path, box=(40, 300, 230, 380)):
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=(255, 255, 255, 255))
    image.save(path)
    return path


def _top_of_box(frame) -> int | None:
    """The first row, in the column through the box, that has gone toward white on the red ground."""
    return next((y for y in range(H) if frame.getpixel((135, y))[1] > 40), None)


def test_a_piece_slides_up_into_place_while_it_fades_in(tmp_path, pacing):  # noqa: F811
    pacing(transition_s=0)
    clips = [ClipPart(_clip(tmp_path / "a.mp4", 3, "red"), 30)]
    part = OverlayPart(_box_overlay(tmp_path / "o.png"), 0.5, 2.5, fade_s=0.3, slide_px=40)
    result = _compose(tmp_path, clips=clips, overlays=[part], voices=[])

    early = _frame(result.video_path, 0.6, tmp_path, "early")       # 0.1 s in: low, and faint
    settled = _frame(result.video_path, 1.5, tmp_path, "settled")
    before = _frame(result.video_path, 0.3, tmp_path, "before")
    assert _top_of_box(before) is None
    assert _top_of_box(settled) == 300 and settled.getpixel((135, 340))[1] > 240            # in place, fully opaque
    assert _top_of_box(early) > 300 + 10                                                     # still sliding up
    assert 40 < early.getpixel((135, 370))[1] < 240                                          # part way through its fade


def test_a_piece_with_no_slide_stays_where_it_is(tmp_path, pacing):  # noqa: F811
    pacing(transition_s=0)
    clips = [ClipPart(_clip(tmp_path / "a.mp4", 3, "red"), 30)]
    part = OverlayPart(_box_overlay(tmp_path / "o.png"), 0.5, 2.5, fade_s=0.3, slide_px=0)
    result = _compose(tmp_path, clips=clips, overlays=[part], voices=[])
    assert _top_of_box(_frame(result.video_path, 0.7, tmp_path, "early")) == 300


def test_pieces_with_different_start_times_appear_one_after_another(tmp_path, pacing):  # noqa: F811
    pacing(transition_s=0)
    clips = [ClipPart(_clip(tmp_path / "a.mp4", 3, "red"), 30)]
    first = OverlayPart(_box_overlay(tmp_path / "o1.png", (40, 100, 230, 160)), 0.4, 2.5, slide_px=20)
    second = OverlayPart(_box_overlay(tmp_path / "o2.png", (40, 300, 230, 360)), 1.2, 2.5, slide_px=20)
    result = _compose(tmp_path, clips=clips, overlays=[first, second], voices=[])
    at_one = _frame(result.video_path, 1.0, tmp_path, "one")
    at_two = _frame(result.video_path, 1.9, tmp_path, "two")
    assert at_one.getpixel((135, 130))[1] > 200 and at_one.getpixel((135, 330))[1] < 20      # only the first so far
    assert at_two.getpixel((135, 130))[1] > 200 and at_two.getpixel((135, 330))[1] > 200     # both
    assert video_compose.OverlayPart(_box_overlay(tmp_path / "x.png"), 0, 1).fade_s == 0.3     # the default
