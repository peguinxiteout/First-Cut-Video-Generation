from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.video_schemas import (
    ClipResult, LineTiming, Shot, Storyboard, VideoBrief, VideoEvaluationResult, VideoGenerationRecord,
    VideoGate, VideoInput, VideoRequest, VideoScript,
)


def test_an_empty_input_is_a_valid_15_second_vertical_video():
    video_input = VideoInput()
    assert (video_input.duration_s, video_input.aspect, video_input.model) == (15, "9:16", "auto")
    assert video_input.allow_generated is False


@pytest.mark.parametrize("seconds", [10, 15, 30, 45, 60])
def test_the_five_durations_are_accepted(seconds):
    assert VideoInput(duration_s=seconds).duration_s == seconds


@pytest.mark.parametrize("seconds", [0, 7, 20, 90])
def test_any_other_duration_is_refused_with_the_list(seconds):
    with pytest.raises(ValidationError, match=r"\[10, 15, 30, 45, 60\]"):
        VideoInput(duration_s=seconds)


def test_aspect_and_objective_are_normalised_then_checked():
    video_input = VideoInput(aspect="16x9", objective="Site Visit", model=" Kenburns ")
    assert (video_input.aspect, video_input.objective, video_input.model) == ("16:9", "site_visit", "kenburns")
    with pytest.raises(ValidationError):
        VideoInput(aspect="4:3")
    with pytest.raises(ValidationError):
        VideoInput(objective="viral")


def test_unknown_keys_in_video_input_json_are_ignored():
    assert VideoInput.model_validate({"duration_s": 10, "note": "hand-edited"}).duration_s == 10


def test_the_brief_is_the_input_plus_the_resolved_parts():
    brief = VideoBrief(property_id="P1", name="Anamika High Point", duration_s=10, quality="draft")
    assert brief.duration_s == 10 and brief.width == 1080 and brief.quality == "draft"
    with pytest.raises(ValidationError):
        VideoBrief(property_id="P1", name="X", quality="ultra")


def test_a_shot_needs_a_known_kind_and_camera_move():
    assert Shot(index=0).camera == "push_in"
    with pytest.raises(ValidationError):
        Shot(index=0, camera="whip_pan")
    with pytest.raises(ValidationError):
        Shot(index=0, kind="text_to_video")


def test_storyboard_totals_and_generated_shots():
    board = Storyboard(shots=[Shot(index=0, seconds=3.5), Shot(index=1, kind="generated", seconds=4.0),
                              Shot(index=2, kind="card", seconds=2.0)])
    assert board.total_seconds == 9.5
    assert board.generated_shots == [1]


def test_a_line_timing_knows_its_length():
    assert LineTiming(line_index=0, start_s=1.0, end_s=3.25).duration_s == 2.25


def test_an_unevaluated_result_does_not_pass():
    assert VideoEvaluationResult().gate_result == VideoGate.FAIL


def test_the_record_round_trips_through_json():
    record = VideoGenerationRecord(
        video_id="vid_abc123def0", property_id="P1", status="completed", video_path="/x/video.mp4",
        script=VideoScript(lines=[], mode="template"), storyboard=Storyboard(shots=[Shot(index=0)]),
        clips=[ClipResult(shot_index=0, provider="kenburns", cost_usd=0.0)], providers_used={"kenburns": 1})
    back = VideoGenerationRecord.model_validate_json(record.model_dump_json())
    assert back == record


def test_a_request_from_the_form_has_no_prompt_and_stops_for_approval():
    request = VideoRequest(property_id="P1")
    assert request.auto_approve is False
    assert (request.prompt, request.brief_source, request.brief_notes) == ("", "", [])


def test_a_request_saved_before_the_prompt_fields_existed_still_loads():
    old = '{"property_id": "P1", "overrides": {"duration_s": 10}, "offline": true, "draft": false, "refresh_ledger": false}'
    request = VideoRequest.model_validate_json(old)
    assert request.offline is True and request.overrides == {"duration_s": 10}
    assert request.auto_approve is False and request.prompt == "" and request.brief_notes == []


def test_the_prompt_and_what_was_made_of_it_round_trip_on_the_record():
    request = VideoRequest(property_id="P1", auto_approve=True, prompt="A calm 30 second reel", brief_source="rules",
                           brief_notes=["Length 30 s from the prompt."])
    record = VideoGenerationRecord(video_id="vid_abc123def0", property_id="P1", status="planning", request=request)
    back = VideoGenerationRecord.model_validate_json(record.model_dump_json())
    assert back.request == request and back.request.prompt == "A calm 30 second reel"
