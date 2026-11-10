"""Node 10, the quality gate: each measured check, the verdict, and what the workflow does with it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.agents.video import gate
from src.core.config import load_yaml_config, settings
from src.data.video_schemas import (
    ClipResult, GateCheck, LineTiming, MusicInfo, OverlayItem, Shot, Storyboard, VideoBrief, VideoGate,
    VideoMeasurements, VideoScript, ScriptLine,
)
from src.rendering import ffmpeg
from src.workflows import video as workflow
from tests.video.helpers import Plan, make_video_property

CFG = load_yaml_config("video/evaluation.yaml")


def _brief(**fields) -> VideoBrief:
    return VideoBrief(property_id="P", name="Aurelia Heights", duration_s=15, quality="draft", **fields)


BRIEF = _brief(width=540, height=960, fps=12)                        # a draft, as the brief step resolves it


def _m(**fields) -> VideoMeasurements:
    base = dict(duration_s=15.0, width=540, height=960, fps=12.0, vcodec="h264", acodec="aac", has_audio=True,
                size_bytes=500_000, pix_fmt="yuv420p", loudness_lufs=-16.0, true_peak_dbfs=-4.0)
    base.update(fields)
    return VideoMeasurements(**base)


def _board(*seconds: float, tags=None, overlays=None) -> Storyboard:
    """Photograph shots of the given lengths and a two second closing card."""
    shots = [Shot(index=i, seconds=s, tags=list((tags or {}).get(i, [])), overlay_text=list((overlays or {}).get(i, [])))
             for i, s in enumerate(seconds)]
    shots.append(Shot(index=len(shots), kind="card", camera="still", seconds=2.0))
    return Storyboard(shots=shots)


def _timing(i, start, end, provider="mock", voice="v", mode="speech") -> LineTiming:
    return LineTiming(line_index=i, start_s=start, end_s=end, provider=provider, voice=voice, mode=mode)


def _status(check: GateCheck) -> str:
    return check.status


# -- format ---------------------------------------------------------------------------------------

def test_format_passes_for_h264_aac_at_the_asked_size_and_rate():
    check = gate.check_format(_m(), BRIEF, CFG["format"])
    assert check.status == "pass" and "540x960" in check.detail


@pytest.mark.parametrize("fields, words", [
    ({"vcodec": "vp8"}, "video codec is vp8"), ({"pix_fmt": "yuv444p"}, "pixel format is yuv444p"),
    ({"width": 1080}, "size is 1080x960"), ({"fps": 25.0}, "frame rate is 25"),
    ({"has_audio": False, "acodec": ""}, "no audio track"), ({"acodec": "mp3"}, "audio codec is mp3"),
    ({"size_bytes": 900}, "only 900 bytes"),
])
def test_format_fails_for_each_thing_that_is_wrong_and_says_which(fields, words):
    check = gate.check_format(_m(**fields), BRIEF, CFG["format"])
    assert check.status == "fail" and words in check.detail


# -- duration -------------------------------------------------------------------------------------

@pytest.mark.parametrize("seconds, status", [(15.0, "pass"), (15.2, "pass"), (15.4, "warn"), (14.6, "warn"),
                                              (15.5, "warn"), (15.6, "fail"), (14.0, "fail")])
def test_duration_is_checked_against_the_length_asked_for(seconds, status):
    assert gate.check_duration(_m(duration_s=seconds), BRIEF, CFG["duration"]).status == status


# -- loudness -------------------------------------------------------------------------------------

@pytest.mark.parametrize("lufs, status", [(-16.0, "pass"), (-15.1, "pass"), (-17.0, "pass"), (-14.5, "warn"),
                                           (-18.5, "warn"), (-19.5, "fail"), (-12.0, "fail")])
def test_loudness_is_minus_sixteen_give_or_take_one(lufs, status):
    assert gate.check_loudness(_m(loudness_lufs=lufs), True, CFG["loudness"]).status == status


@pytest.mark.parametrize("peak, status", [(-4.0, "pass"), (-1.0, "pass"), (-0.5, "warn"), (0.3, "fail")])
def test_the_true_peak_has_a_ceiling(peak, status):
    assert gate.check_loudness(_m(true_peak_dbfs=peak), True, CFG["loudness"]).status == status


def test_a_silent_track_fails_when_narration_or_music_was_made_and_is_skipped_when_none_was():
    assert gate.check_loudness(_m(loudness_lufs=-70.0), True, CFG["loudness"]).status == "fail"
    assert gate.check_loudness(_m(loudness_lufs=None), True, CFG["loudness"]).status == "fail"
    assert gate.check_loudness(_m(loudness_lufs=-70.0), False, CFG["loudness"]).status == "skipped"


# -- black frames and frozen stretches --------------------------------------------------------------------

def test_black_frames_pass_warn_and_fail_by_how_long_the_picture_is_black():
    board = _board(4, 4, 3, 2)
    assert gate.check_black(_m(), board, CFG["black"]).status == "pass"
    small = gate.check_black(_m(black_segments=1, black_s=0.2, black_spans=[[5.0, 5.2]]), board, CFG["black"])
    assert small.status == "warn" and small.shots == [2] and "5.0-5.2 s" in small.detail
    big = gate.check_black(_m(black_segments=1, black_s=1.0, black_spans=[[0.0, 1.0]]), board, CFG["black"])
    assert big.status == "fail" and big.shots == [1]


def test_the_still_closing_card_is_not_a_frozen_picture():
    board = _board(4, 4, 3)                                          # the card runs from 11 s to 13 s
    assert gate.check_frozen(_m(frozen_spans=[[11.0, 13.0]]), board, CFG["frozen"]).status == "pass"
    assert gate.check_frozen(_m(), board, CFG["frozen"]).status == "pass"


def test_a_picture_that_stands_still_before_the_card_warns_and_then_fails_with_its_length():
    board = _board(4, 4, 3)
    warn = gate.check_frozen(_m(frozen_spans=[[2.0, 3.4]]), board, CFG["frozen"])
    assert warn.status == "warn" and warn.shots == [1] and "1.4 s" in warn.detail
    assert gate.check_frozen(_m(frozen_spans=[[4.5, 7.0]]), board, CFG["frozen"]).status == "fail"
    assert gate.check_frozen(_m(frozen_spans=[[2.0, 2.6]]), board, CFG["frozen"]).status == "pass"      # under a second
    run_into_the_card = gate.check_frozen(_m(frozen_spans=[[9.5, 13.0]]), board, CFG["frozen"])
    assert run_into_the_card.status == "pass" or "1.5 s" in run_into_the_card.detail                   # only what is before the card counts


# -- narration ------------------------------------------------------------------------------------

def test_narration_with_short_pauses_passes():
    timings = [_timing(0, 0.3, 1.5), _timing(1, 1.8, 4.0), _timing(2, 4.3, 6.0)]
    check = gate.check_narration(timings, CFG["narration"])
    assert check.status == "pass" and "0.30 s" in check.detail


def test_a_long_pause_between_lines_warns_and_a_very_long_one_fails():
    warn = gate.check_narration([_timing(0, 0.3, 1.5), _timing(1, 3.0, 4.0)], CFG["narration"])
    assert warn.status == "warn" and "1.5 s of silence after line 1" in warn.detail
    assert gate.check_narration([_timing(0, 0.3, 1.5), _timing(1, 4.5, 5.0)], CFG["narration"]).status == "fail"


def test_silence_before_the_first_word_warns_and_no_spoken_line_warns():
    assert gate.check_narration([_timing(0, 2.0, 3.0)], CFG["narration"]).status == "warn"
    none = gate.check_narration([_timing(0, 0.3, 1.5, mode="silent")], CFG["narration"])
    assert none.status == "warn" and "no narration" in none.detail


# -- text inside the safe zones -----------------------------------------------------------------------

def _item(shot, *rects, texts=("x",), **fields) -> OverlayItem:
    return OverlayItem(shot_index=shot, texts=list(texts), rects=list(rects), **fields)


def _rect(x, y, w=100, h=40, text="Aurelia") -> dict:
    return {"x": x, "y": y, "w": w, "h": h, "text": text}


def test_text_inside_the_safe_zone_passes():
    check = gate.check_safe_zones([_item(0, _rect(100, 400))], BRIEF, CFG["safe_zones"])
    assert check.status == "pass" and "1 piece" in check.detail


@pytest.mark.parametrize("rect", [_rect(100, 20), _rect(100, 900), _rect(5, 400), _rect(480, 400)])
def test_text_over_any_edge_of_the_safe_zone_fails(rect):
    check = gate.check_safe_zones([_item(2, rect)], BRIEF, CFG["safe_zones"])
    assert check.status == "fail" and check.shots == [3] and "Aurelia" in check.detail


def test_text_a_pixel_or_two_over_the_edge_is_within_tolerance():
    top = round(BRIEF.height * 0.14)
    assert gate.check_safe_zones([_item(0, _rect(100, top - 3))], BRIEF, CFG["safe_zones"]).status == "pass"
    assert gate.check_safe_zones([_item(0, _rect(100, top - 12))], BRIEF, CFG["safe_zones"]).status == "fail"


def test_text_whose_place_was_not_measured_is_not_called_a_pass():
    assert gate.check_safe_zones([_item(0)], BRIEF, CFG["safe_zones"]).status == "warn"
    assert gate.check_safe_zones([], BRIEF, CFG["safe_zones"]).status == "skipped"
    mixed = gate.check_safe_zones([_item(0, _rect(100, 400)), _item(1)], BRIEF, CFG["safe_zones"])
    assert mixed.status == "pass" and "not measured" in mixed.detail


# -- the figures ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plan(tmp_path_factory) -> Plan:
    return Plan(make_video_property(tmp_path_factory.mktemp("gate") / "p"))


def _inputs(plan: Plan, **fields) -> gate.GateInputs:
    base = dict(video_path=Path("none.mp4"), thumbnail_path=None, brief=plan.brief, script=plan.script,
                storyboard=_board(4, 4, 3), guard=gate.build_guard(plan.prop, plan.knowledge, plan.brief))
    base.update(fields)
    return gate.GateInputs(**base)


def test_figures_that_are_facts_of_the_property_pass(plan):
    check = gate.check_numbers(_inputs(plan, overlay_items=[_item(0, texts=[line.text for line in plan.script.lines])]))
    assert check.status == "pass" and "printed or spoken" in check.detail


def test_an_invented_figure_on_screen_fails_and_names_it(plan):
    item = _item(1, texts=["Only 37 homes left"])
    check = gate.check_numbers(_inputs(plan, overlay_items=[item]))
    assert check.status == "fail" and "'37'" in check.detail and "Only 37 homes left" in check.detail


def test_an_invented_figure_in_a_spoken_line_fails_too(plan):
    script = plan.script.model_copy(update={"lines": [*plan.script.lines[:-1],
                                                      ScriptLine(text="Prices from 99 crore.", role="cta")]})
    assert gate.check_numbers(_inputs(plan, script=script)).status == "fail"


def test_without_a_ledger_the_figures_are_not_checked(plan):
    assert gate.check_numbers(_inputs(plan, guard=None)).status == "skipped"


# -- the thumbnail ------------------------------------------------------------------------------------

def _picture(path: Path, colour=None, size=(270, 480), noise=False) -> Path:
    image = Image.new("RGB", size, colour or (200, 30, 30))
    if noise:
        import random
        rng = random.Random(1)
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size[0] * size[1])])
    image.save(path, "JPEG", quality=90)
    return path


def test_a_real_picture_is_a_good_thumbnail(tmp_path):
    check = gate.check_thumbnail(_picture(tmp_path / "t.jpg", noise=True), BRIEF, CFG["thumbnail"])
    assert check.status == "pass" and check.values["stddev"] > 30


def test_a_blank_thumbnail_fails_and_a_nearly_blank_one_warns(tmp_path):
    blank = gate.check_thumbnail(_picture(tmp_path / "blank.jpg"), BRIEF, CFG["thumbnail"])
    assert blank.status == "fail" and "almost blank" in blank.detail
    image = Image.new("RGB", (270, 480), (120, 120, 120))
    image.paste((140, 140, 140), (0, 0, 135, 480))
    image.save(tmp_path / "flat.jpg", quality=95)
    flat = gate.check_thumbnail(tmp_path / "flat.jpg", BRIEF, CFG["thumbnail"])
    assert flat.status in ("warn", "fail")


def test_a_missing_or_unreadable_thumbnail_fails(tmp_path):
    assert gate.check_thumbnail(None, BRIEF, CFG["thumbnail"]).status == "fail"
    assert gate.check_thumbnail(tmp_path / "nope.jpg", BRIEF, CFG["thumbnail"]).status == "fail"
    (tmp_path / "bad.jpg").write_bytes(b"not a picture" * 500)
    assert "cannot be read" in gate.check_thumbnail(tmp_path / "bad.jpg", BRIEF, CFG["thumbnail"]).detail


def test_a_thumbnail_of_the_wrong_shape_warns(tmp_path):
    check = gate.check_thumbnail(_picture(tmp_path / "wide.jpg", size=(480, 270), noise=True), BRIEF, CFG["thumbnail"])
    assert check.status == "warn" and "shape" in check.detail


# -- upscale ------------------------------------------------------------------------------------------

def _clip(shot, layout="fill", peak=1.0) -> ClipResult:
    return ClipResult(shot_index=shot, layout=layout, peak_upscale=peak, mode="kenburns")


@pytest.mark.parametrize("peak, status", [(0.9, "pass"), (1.3, "pass"), (1.5, "warn"), (2.4, "fail")])
def test_the_most_a_photograph_is_enlarged_is_limited(peak, status):
    check = gate.check_upscale([_clip(0, "fill", 0.8), _clip(1, "fill", peak), _clip(2, "fit", 0.5)], CFG["upscale"])
    assert check.status == status
    assert "x%.2f" % peak in check.detail and "2: fill" in check.detail                      # every shot is listed
    assert check.shots == ([] if status == "pass" else [2])


def test_a_video_with_no_laid_out_photograph_skips_the_upscale_check():
    assert gate.check_upscale([ClipResult(shot_index=0, mode="card")], CFG["upscale"]).status == "skipped"


# -- one voice ------------------------------------------------------------------------------------------

def test_one_voice_across_all_lines_passes():
    check = gate.check_one_voice([_timing(0, 0, 1, "cloud_tts", "Sulafat"), _timing(1, 2, 3, "cloud_tts", "Sulafat")])
    assert check.status == "pass" and "cloud_tts Sulafat" in check.detail


def test_two_voices_fail_and_so_does_a_mix_of_spoken_and_silent_lines():
    two = gate.check_one_voice([_timing(0, 0, 1, "cloud_tts", "Sulafat"), _timing(1, 2, 3, "offline", "female")])
    assert two.status == "fail" and "cloud_tts Sulafat" in two.detail and "offline female" in two.detail
    mixed = gate.check_one_voice([_timing(0, 0, 1), _timing(1, 2, 3, mode="silent")])
    assert mixed.status == "fail" and "silent" in mixed.detail


def test_a_wholly_silent_video_is_one_voice_and_no_lines_is_skipped():
    assert gate.check_one_voice([_timing(0, 0, 1, mode="silent"), _timing(1, 2, 3, mode="silent")]).status == "pass"
    assert gate.check_one_voice([]).status == "skipped"


# -- the hook -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seconds, status", [(2.6, "pass"), (2.5, "pass"), (2.46, "pass"), (2.3, "fail"), (1.8, "fail")])
def test_the_opening_shot_lasts_at_least_two_and_a_half_seconds(seconds, status):
    check = gate.check_hook(_board(seconds, 4, 4), load_yaml_config("video/pacing.yaml")["min_hook_s"])
    assert check.status == status and check.shots == ([1] if status == "fail" else [])


# -- chips -----------------------------------------------------------------------------------------------------

def _chip_script(*chips) -> VideoScript:
    return VideoScript(lines=[ScriptLine(text="Hi.", role="hook")], chips=list(chips), chip_lines=[0] * len(chips))


def test_a_chip_over_a_photograph_tagged_with_what_it_names_passes():
    board = _board(3, 3, 3, tags={1: ["rooftop", "bbq", "dining"]}, overlays={1: ["BBQ"]})
    check = gate.check_chips(board, _chip_script("BBQ"))
    assert check.status == "pass" and "1 chip" in check.detail


def test_a_chip_over_a_photograph_that_does_not_show_it_fails_with_the_shot():
    board = _board(3, 3, 3, tags={1: ["lawn", "gym"]}, overlays={1: ["BBQ"]})
    check = gate.check_chips(board, _chip_script("BBQ"))
    assert check.status == "fail" and check.shots == [2] and "lawn, gym" in check.detail


def test_a_chip_over_a_photograph_with_no_tags_warns_and_no_chips_passes():
    board = _board(3, 3, 3, overlays={1: ["BBQ"]})
    assert gate.check_chips(board, _chip_script("BBQ")).status == "warn"
    assert gate.check_chips(_board(3, 3, 3), _chip_script("BBQ")).status == "pass"


def test_the_title_over_the_hook_is_not_a_chip():
    board = _board(3, 3, 3, tags={0: ["exterior"]}, overlays={0: ["Aurelia Heights"]})
    assert gate.check_chips(board, _chip_script("BBQ")).status == "pass"


# -- narration against picture ------------------------------------------------------------------------------------

def _spoken_script(*texts: str) -> VideoScript:
    return VideoScript(lines=[ScriptLine(text=t, role="hook" if i == 0 else "feature") for i, t in enumerate(texts)])


def _board_with(*shots: tuple[list[int], list[str]]) -> Storyboard:
    """Photograph shots of (the lines they carry, the tags of their photograph) and the closing card."""
    real = [Shot(index=i, seconds=3.0, line_ids=lines, tags=tags, image_id=f"IMG{i}") for i, (lines, tags) in enumerate(shots)]
    real.append(Shot(index=len(real), kind="card", camera="still", seconds=2.0))
    return Storyboard(shots=real)


def test_every_amenity_a_line_names_is_a_tag_of_the_photograph_under_it():
    script = _spoken_script("Welcome.", "On site: gym and swimming pool.")
    board = _board_with(([0], ["exterior"]), ([1], ["gym", "pool"]))
    check = gate.check_narration_picture(script, board, CFG["narration_picture"])
    assert check.status == "pass" and "2 amenity name(s)" in check.detail


def test_an_amenity_named_and_not_shown_fails_with_the_line_and_the_shot():
    script = _spoken_script("Welcome.", "On site: gym and swimming pool.")
    board = _board_with(([0], ["exterior"]), ([1], ["gym", "lawn"]))
    check = gate.check_narration_picture(script, board, CFG["narration_picture"])
    assert check.status == "fail" and check.shots == [2] and "line 2 names pool" in check.detail


def test_a_line_over_two_shots_may_show_its_amenities_in_either():
    script = _spoken_script("Welcome.", "On site: gym and swimming pool.")
    board = _board_with(([0], ["exterior"]), ([1], ["gym"]), ([1], ["pool"]))
    assert gate.check_narration_picture(script, board, CFG["narration_picture"]).status == "pass"


def test_a_line_naming_an_amenity_that_plays_over_no_photograph_fails():
    script = _spoken_script("Welcome.", "Come and see the pool.")
    board = _board_with(([0], ["exterior"]))                              # nothing carries line 1
    check = gate.check_narration_picture(script, board, CFG["narration_picture"])
    assert check.status == "fail" and "line 2 names pool" in check.detail


def test_lines_that_name_no_amenity_are_not_checked():
    script = _spoken_script("Welcome to Aurelia Heights.", "Two minutes from the metro.")
    board = _board_with(([0], ["exterior"]), ([1], ["lawn"]))
    check = gate.check_narration_picture(script, board, CFG["narration_picture"])
    assert check.status == "pass" and "no spoken line names an amenity" in check.detail


def test_pictures_with_no_tags_cannot_confirm_the_narration():
    script = _spoken_script("Welcome.", "On site: swimming pool.")
    assert gate.check_narration_picture(script, _board_with(([0], []), ([1], [])), CFG["narration_picture"]).status == "skipped"
    mixed = gate.check_narration_picture(script, _board_with(([0], ["exterior"]), ([1], [])), CFG["narration_picture"])
    assert mixed.status == "warn" and "line 2" in mixed.detail


def test_the_number_of_missing_amenities_that_fails_comes_from_the_config():
    script = _spoken_script("Welcome.", "On site: gym and swimming pool.")
    board = _board_with(([0], ["exterior"]), ([1], ["gym"]))
    assert gate.check_narration_picture(script, board, {"fail_missing": 2}).status == "warn"
    assert gate.check_narration_picture(script, board, {"fail_missing": 1}).status == "fail"


def test_narration_against_picture_is_a_hard_check_like_the_chips():
    assert "narration_picture" in CFG["hard"] and "chips" in CFG["hard"]
    assert gate.verdict(_checks(narration_picture="fail"), CFG).gate_result == VideoGate.FAIL


# -- the same photograph twice in a row ------------------------------------------------------------------------------

def _row(*ids: str) -> Storyboard:
    shots = [Shot(index=i, seconds=3.0, image_id=image_id) for i, image_id in enumerate(ids)]
    shots.append(Shot(index=len(shots), kind="card", camera="still", seconds=2.0))
    return Storyboard(shots=shots)


def test_different_photographs_in_every_shot_pass():
    check = gate.check_repeated_photo(_row("A", "B", "C"), CFG["repeated_photo"])
    assert check.status == "pass"


def test_the_same_photograph_in_two_shots_in_a_row_warns_and_names_the_shots():
    check = gate.check_repeated_photo(_row("A", "B", "B", "C"), CFG["repeated_photo"])
    assert check.status == "warn" and check.shots == [2, 3] and "shots 2 and 3" in check.detail
    assert check.values["repeats"] == 1


def test_a_photograph_that_comes_back_later_is_not_a_repeat_in_a_row():
    assert gate.check_repeated_photo(_row("A", "B", "A"), CFG["repeated_photo"]).status == "pass"


def test_the_number_of_repeats_that_warns_comes_from_the_config():
    board = _row("A", "A", "B", "B")
    assert gate.check_repeated_photo(board, {"warn_at": 2}).status == "warn"
    assert gate.check_repeated_photo(_row("A", "A", "B"), {"warn_at": 2}).status == "pass"


def test_a_repeat_only_warns_and_is_not_a_hard_check():
    assert "repeated_photo" not in CFG["hard"]
    assert gate.verdict(_checks(repeated_photo="warn"), CFG).gate_result == VideoGate.WARN


# -- the verdict ------------------------------------------------------------------------------------------------

def _checks(**statuses) -> list[GateCheck]:
    names = list(CFG["weights"])
    return [GateCheck(name=n, status=statuses.get(n, "pass"), detail=f"{n} detail") for n in names]


def test_all_passes_score_a_hundred_and_pass():
    result = gate.verdict(_checks(), CFG)
    assert (result.quality_score, result.gate_result, result.defects) == (100.0, VideoGate.PASS, [])
    assert result.judge == "measured" and result.failed_shots == []


def test_the_weights_sum_to_one_and_every_check_has_one():
    assert sum(CFG["weights"].values()) == pytest.approx(1.0)
    assert set(CFG["hard"]) <= set(CFG["weights"])


def test_a_warning_halves_its_weight_and_the_gate_warns():
    result = gate.verdict(_checks(loudness="warn"), CFG)
    assert result.quality_score == pytest.approx(100 - 0.5 * CFG["weights"]["loudness"] * 100)
    assert result.gate_result == VideoGate.WARN and result.defects == ["[WARN] loudness: loudness detail"]


def test_a_hard_check_failing_fails_the_gate_even_when_the_score_is_high():
    result = gate.verdict(_checks(numbers="fail"), CFG)
    assert result.quality_score == pytest.approx(88.0) and result.gate_result == VideoGate.FAIL
    assert result.defects == ["[FAIL] numbers: numbers detail"]


def test_a_soft_check_failing_only_warns_while_the_score_holds():
    assert gate.verdict(_checks(frozen="fail"), CFG).gate_result == VideoGate.WARN
    assert gate.verdict(_checks(loudness="fail"), CFG).gate_result == VideoGate.WARN


def test_many_soft_failures_drag_the_score_under_the_floor_and_fail_the_gate():
    result = gate.verdict(_checks(loudness="fail", frozen="fail", narration_gaps="fail", upscale="fail",
                                  black_frames="warn", one_voice="warn", narration_evidence="fail", hold_length="fail"), CFG)
    assert result.quality_score < CFG["fail_below"] and result.gate_result == VideoGate.FAIL


def test_a_check_that_was_not_measured_is_left_out_of_the_score_and_listed():
    items = _checks(loudness="skipped", upscale="skipped")
    for item in items:
        if item.status == "skipped":
            item.detail = "nothing to measure"
    result = gate.verdict(items, CFG)
    assert result.quality_score == 100.0 and result.gate_result == VideoGate.PASS
    assert result.feedback == ["loudness was not measured: nothing to measure", "upscale was not measured: nothing to measure"]


def test_the_shots_of_every_problem_are_collected_for_the_record():
    items = _checks(upscale="warn", chips="fail")
    items[[i.name for i in items].index("upscale")].shots = [3]
    items[[i.name for i in items].index("chips")].shots = [2, 3]
    assert gate.verdict(items, CFG).failed_shots == [2, 3]


# -- measuring a real file ----------------------------------------------------------------------------------------

def _movie(tmp_path) -> Path:
    """4 s moving picture, 1 s black, 3 s of one red frame, 2 s moving: a 10 s MP4 with a tone."""
    def part(name, source, seconds):
        path = tmp_path / f"{name}.mp4"
        ffmpeg.run(["-f", "lavfi", "-i", f"{source}:s=270x480:r=10:d={seconds}" if "=" in source
                    else f"{source}=s=270x480:r=10:d={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
        return path

    parts = [part("a", "testsrc2", 4), part("b", "color=c=black", 1), part("c", "color=c=red", 3),
             part("d", "testsrc2", 2)]
    listing = tmp_path / "list.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    silent = tmp_path / "silent.mp4"
    ffmpeg.run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(silent)])
    movie = tmp_path / "movie.mp4"
    ffmpeg.run(["-i", str(silent), "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=10",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(movie)])
    return movie


def test_measure_reads_the_streams_loudness_black_and_frozen_stretches_off_the_file(tmp_path):
    m = gate.measure(_movie(tmp_path), CFG)
    assert (m.width, m.height, m.vcodec, m.acodec, m.pix_fmt) == (270, 480, "h264", "aac", "yuv420p")
    assert m.duration_s == pytest.approx(10.0, abs=0.2) and m.has_audio and m.loudness_lufs is not None
    assert m.black_segments == 1 and m.black_s == pytest.approx(1.0, abs=0.2)
    assert m.black_spans[0][0] == pytest.approx(4.0, abs=0.2)
    assert m.frozen_s == pytest.approx(4.0, abs=0.4)                    # the black second and the red three seconds


def test_the_gate_catches_black_and_frozen_picture_in_a_real_file(tmp_path, plan):
    video = _movie(tmp_path)
    board = _board(4, 1, 3)                                              # shots 1-3 and a card, as the file runs 0-8 s, then 2 s
    inputs = gate.GateInputs(video_path=video, thumbnail_path=None, brief=_brief(), script=plan.script, storyboard=board)
    result, m = gate.evaluate(inputs, cfg=CFG)
    by = {i.name: i for i in result.items}
    assert by["black_frames"].status == "fail" and by["black_frames"].shots == [2]      # one second of black: over 0.3 s
    assert by["frozen"].status == "fail" and "does not change" in by["frozen"].detail
    assert by["format"].status == "fail"                                                 # 270x480, not the brief's 540x960
    assert result.gate_result == VideoGate.FAIL


def test_a_file_that_cannot_be_read_fails_the_format_check_and_the_gate(tmp_path, plan):
    (tmp_path / "bad.mp4").write_bytes(b"nothing")
    result, _ = gate.evaluate(_inputs(plan, video_path=tmp_path / "bad.mp4"), cfg=CFG)
    assert result.gate_result == VideoGate.FAIL
    assert result.items[0].name == "format" and "could not be read" in result.items[0].detail


def test_a_check_that_raises_is_skipped_with_the_reason_and_the_rest_still_run(plan, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no pixels")

    monkeypatch.setattr(gate, "check_thumbnail", boom)
    result, _ = gate.evaluate(_inputs(plan), measurements=_m(), cfg=CFG)
    by = {i.name: i for i in result.items}
    assert by["thumbnail"].status == "skipped" and "RuntimeError" in by["thumbnail"].detail
    assert {"format", "duration", "hook", "chips", "numbers"} <= set(by)


def test_evaluate_runs_all_fifteen_checks_and_the_music_counts_as_an_audio_programme(plan):
    inputs = _inputs(plan, music_info=MusicInfo(provider="lyria"),
                     line_timings=[_timing(0, 0.3, 1.5)], thumbnail_path=None)
    result, _ = gate.evaluate(inputs, measurements=_m(), cfg=CFG)
    assert [i.name for i in result.items] == list(CFG["weights"])
    assert {i.name: i.weight for i in result.items} == CFG["weights"]


# -- the workflow --------------------------------------------------------------------------------------------------

@pytest.fixture
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"), raising=False)
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "v.db"), raising=False)


def _render(tmp_path):
    return workflow.run_video_workflow(make_video_property(tmp_path / "properties"), offline=True, draft=True)


def test_the_status_is_the_gates_verdict():
    from src.data.video_schemas import VideoEvaluationResult, VideoGenerationRecord

    record = VideoGenerationRecord(video_id="v", property_id="p")
    for gate_result, status in ((VideoGate.PASS, "passed"), (VideoGate.WARN, "warned"), (VideoGate.FAIL, "failed")):
        record.evaluation = VideoEvaluationResult(gate_result=gate_result)
        assert workflow._status(record) == status


@pytest.mark.slow
def test_a_render_is_gated_and_qa_json_lists_every_check_and_shot(tmp_path, _isolated):
    record = _render(tmp_path)
    qa = json.loads((Path(record.output_dir) / "qa.json").read_text(encoding="utf-8"))
    assert qa["video_id"] == record.video_id and qa["gate_result"] == record.evaluation.gate_result.value
    assert [c["name"] for c in qa["checks"]] == list(CFG["weights"]) and qa["score"] == record.evaluation.quality_score
    assert len(qa["shots"]) == len(record.storyboard.shots) and qa["measurements"]["width"] == 540
    assert record.measurements.pix_fmt == "yuv420p" and record.evaluation.items and record.agent_reports["gate"]
    assert record.status == {"pass": "passed", "warn": "warned", "fail": "failed"}[qa["gate_result"]]


@pytest.mark.slow
def test_a_failed_gate_keeps_the_mp4_and_says_why(tmp_path, _isolated, monkeypatch):
    real = gate.evaluate

    def failing(inputs, *args, **kwargs):
        result, measured = real(inputs, *args, **kwargs)
        result.items.append(GateCheck(name="hook", status="fail", detail="the opening shot lasts 1.0 s"))
        return gate.verdict([i for i in result.items if i.name != "hook" or i.status == "fail"], CFG), measured

    monkeypatch.setattr(gate, "evaluate", failing)
    record = _render(tmp_path)
    assert record.status == "failed" and Path(record.video_path).exists() and Path(record.thumbnail_path).exists()
    assert "the opening shot lasts 1.0 s" in record.error and record.evaluation.gate_result == VideoGate.FAIL
    assert record.workflow_evaluation.steps[-1].node == "gate" and record.workflow_evaluation.steps[-1].status == "success"


@pytest.mark.slow
def test_a_gate_that_crashes_only_warns_and_the_video_is_kept(tmp_path, _isolated, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(gate, "evaluate", boom)
    record = _render(tmp_path)
    assert record.status == "warned" and record.error is None and Path(record.video_path).exists()
    assert "the gate could not run (RuntimeError: detector exploded)" in record.evaluation.defects[0]
    assert (Path(record.output_dir) / "qa.json").exists()


# -- narration evidence: tags AND captions -------------------------------------------------------------------------

def _evidence_board(*shots: tuple[list[int], list[str], str, list[str]]) -> Storyboard:
    """Photograph shots of (the lines they carry, the tags of their photograph, its caption, the text on it)."""
    real = [Shot(index=i, seconds=3.0, line_ids=lines, tags=tags, caption=caption, overlay_text=text,
                 image_id=f"IMG{i}") for i, (lines, tags, caption, text) in enumerate(shots)]
    real.append(Shot(index=len(real), kind="card", camera="still", seconds=2.0))
    return Storyboard(shots=real)


def test_an_amenity_over_a_photograph_whose_tags_and_caption_both_say_it_passes():
    script = _spoken_script("Welcome.", "On site: gym.")
    board = _evidence_board(([0], ["exterior"], "A tower at dusk", []), ([1], ["gym"], "A gym with treadmills", []))
    check = gate.check_narration_evidence(script, board)
    assert check.status == "pass" and "1 amenity mention(s)" in check.detail


def test_an_amenity_over_a_photograph_with_the_tag_and_no_caption_support_warns():
    script = _spoken_script("Welcome.", "On site: gym.")
    board = _evidence_board(([0], ["exterior"], "A tower at dusk", []), ([1], ["gym"], "A room with grey walls", []))
    check = gate.check_narration_evidence(script, board)
    assert check.status == "warn" and "line 2 names gym" in check.detail and check.values["weak"] == 1


def test_one_shot_among_those_under_a_line_is_enough():
    script = _spoken_script("Welcome.", "On site: gym.")
    board = _evidence_board(([0], ["exterior"], "A tower", []), ([1], ["gym"], "Grey walls", []),
                            ([1], ["gym"], "Rows of treadmills", []))
    assert gate.check_narration_evidence(script, board).status == "pass"


def test_a_chip_is_held_to_the_same_rule():
    script = _chip_script("Gym")
    over_caption = _evidence_board(([0], ["gym"], "A gym with weights", ["Gym"]))
    assert gate.check_narration_evidence(script, over_caption).status == "pass"
    tag_only = _evidence_board(([0], ["gym"], "Two towers at dusk", ["Gym"]))
    check = gate.check_narration_evidence(script, tag_only)
    assert check.status == "warn" and "the chip 'Gym' (shot 1)" in check.detail


def test_without_tags_there_is_nothing_to_audit_and_no_amenity_passes():
    script = _spoken_script("Welcome.", "On site: gym.")
    assert gate.check_narration_evidence(script, _evidence_board(([0], [], "", []), ([1], [], "", []))).status == "skipped"
    plain = _evidence_board(([0], ["exterior"], "A tower", []))
    assert gate.check_narration_evidence(_spoken_script("Welcome."), plain).status == "pass"


def test_narration_evidence_is_soft_and_weighted():
    assert "narration_evidence" in CFG["weights"] and "narration_evidence" not in CFG["hard"]
    assert gate.verdict(_checks(narration_evidence="warn"), CFG).gate_result == VideoGate.WARN
    assert gate.verdict(_checks(narration_evidence="fail"), CFG).gate_result == VideoGate.WARN


# -- hold length -----------------------------------------------------------------------------------------------------

HOLD = load_yaml_config("video/pacing.yaml")["hold"]


def test_the_hold_limits_are_in_pacing_yaml():
    assert (HOLD["target_s"], HOLD["cap_s"], HOLD["warn_s"], HOLD["fail_s"]) == (3.5, 4.5, 4.5, 6.0)


@pytest.mark.parametrize("longest,status", [(3.1, "pass"), (4.5, "pass"), (4.6, "warn"), (6.0, "warn"), (6.2, "fail")])
def test_the_longest_photograph_hold_passes_up_to_the_cap_warns_up_to_six_and_fails_above(longest, status):
    board = _board(3.0, longest, 3.0)
    check = gate.check_hold_length(board, HOLD)
    assert check.status == status and check.values["hold_s"] == pytest.approx(longest)
    assert (check.shots == [2]) == (status != "pass") and f"held {longest:.1f} s (shot 2)" in check.detail


def test_the_closing_card_is_not_a_photograph_hold():
    board = Storyboard(shots=[Shot(index=0, seconds=3.0), Shot(index=1, kind="card", camera="still", seconds=8.0)])
    assert gate.check_hold_length(board, HOLD).status == "pass"


def test_a_video_with_no_photograph_shot_skips_the_hold_check():
    board = Storyboard(shots=[Shot(index=0, kind="card", camera="still", seconds=5.0)])
    assert gate.check_hold_length(board, HOLD).status == "skipped"


def test_hold_length_is_soft_weighted_and_lowers_the_score_only():
    assert CFG["weights"]["hold_length"] == pytest.approx(0.04) and "hold_length" not in CFG["hard"]
    result = gate.verdict(_checks(hold_length="fail"), CFG)
    assert result.gate_result == VideoGate.WARN and result.quality_score == pytest.approx(96.0)
    assert gate.verdict(_checks(hold_length="warn"), CFG).quality_score == pytest.approx(98.0)
