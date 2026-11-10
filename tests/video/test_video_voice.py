"""Node 6: the voice, and the picture cut to the speech."""
from __future__ import annotations

import wave

import pytest

from src.agents.video import voice
from src.data.video_schemas import ScriptLine, Shot, Storyboard, VideoBrief, VideoScript
from src.models.music import MusicService
from src.models.speech import SpeechService
from src.models.speech.mock import MockSpeechProvider
from src.models.speech.silent import SilentSpeechProvider

CONFIG = {"end_card_s": 2.0, "voice": {"lead_s": 0.3, "gap_s": 0.25, "tail_s": 0.3, "min_lead_s": 0.1,
                                        "min_gap_s": 0.1, "min_end_card_s": 1.5, "pre_roll_s": 0.15,
                                        "fit_tolerance_s": 0.5}}


def _board(shot_lines=((0,), (1,), (2,), ())) -> Storyboard:
    """Three photograph shots and a closing card carrying the last line."""
    shots = [Shot(index=i, seconds=3.0, line_ids=list(ids)) for i, ids in enumerate(shot_lines[:-1])]
    shots.append(Shot(index=len(shots), kind="card", camera="still", seconds=2.0, line_ids=[len(shot_lines) - 1]))
    return Storyboard(shots=shots)


def _timing(durations, duration_s, board=None):
    return voice.retime(durations, board or _board(), duration_s, CONFIG)


def _gaps(durations, timing):
    return [timing.starts[i + 1] - (timing.starts[i] + durations[i]) for i in range(len(durations) - 1)]


# -- the timeline ------------------------------------------------------------------------------

def test_a_short_narration_is_spread_out_and_the_video_keeps_its_length():
    durations = [1.5, 2.5, 2.5, 2.0]
    t = _timing(durations, 15)

    assert t.total_s == pytest.approx(15.0, abs=0.01)
    assert sum(t.shot_seconds) == pytest.approx(t.total_s, abs=0.01)
    assert t.defects == []
    assert all(b > a for a, b in zip(t.starts, t.starts[1:]))
    assert all(g > 0.25 for g in _gaps(durations, t))                          # the spare time went into the pauses
    assert t.starts[-1] + durations[-1] <= t.total_s - 0.2                      # the last word ends before the video does
    assert t.notes and "pauses are" in t.notes[0]


def test_lines_never_overlap_and_speech_is_never_shortened():
    durations = [1.5, 2.5, 2.5, 2.0]
    for total in (15, 11, 9.5):
        t = _timing(durations, total)
        for i in range(3):
            assert t.starts[i + 1] >= t.starts[i] + durations[i] + 0.1 - 1e-6


def test_a_narration_a_little_too_long_is_fitted_by_shortening_the_gaps():
    durations = [2.0, 2.5, 2.5, 1.0]
    t = _timing(durations, 9.5)

    assert t.total_s == pytest.approx(9.5, abs=0.01) and t.defects == []
    assert all(0.1 <= g < 0.25 for g in _gaps(durations, t))
    assert t.shot_seconds[-1] >= 2.0 - 1e-6                                     # the card kept its length


def test_when_gaps_are_not_enough_the_closing_card_shrinks_but_never_below_one_and_a_half_seconds():
    durations = [2.0, 2.5, 2.5, 1.0]
    t = _timing(durations, 9.0)

    assert t.total_s == pytest.approx(9.0, abs=0.01) and t.defects == []
    assert 1.5 <= t.shot_seconds[-1] < 2.0
    assert all(g == pytest.approx(0.1, abs=1e-6) for g in _gaps(durations, t))     # gaps are already at their floor
    assert any("closing card is" in n for n in t.notes)


def test_a_narration_that_cannot_fit_runs_over_and_says_so_without_speeding_up_the_speech():
    durations = [2.0, 2.5, 2.5, 1.0]
    t = _timing(durations, 6.0)

    assert t.total_s > 6.5
    assert len(t.defects) == 1
    assert "runs" in t.defects[0] and "over" in t.defects[0] and "not sped up" in t.defects[0]
    assert t.shot_seconds[-1] >= 1.5
    assert sum(t.shot_seconds) == pytest.approx(t.total_s, abs=0.01)
    for i in range(3):                                                           # every line still has its whole length
        assert t.starts[i + 1] - t.starts[i] >= durations[i]


def test_an_overrun_within_the_tolerance_is_a_note_not_a_defect():
    t = _timing([2.0, 2.5, 2.5, 1.0], 8.7)
    assert t.defects == [] and any("runs 0.1 s over" in n for n in t.notes)


def test_a_shot_begins_just_before_the_line_it_carries():
    durations = [1.5, 2.5, 2.5, 2.0]
    t = _timing(durations, 15)
    starts = [0.0]
    for seconds in t.shot_seconds[:-1]:
        starts.append(starts[-1] + seconds)
    for shot_start, line_start in zip(starts[1:], t.starts[1:3]):
        assert shot_start == pytest.approx(line_start - 0.15, abs=0.01)
    card_start = sum(t.shot_seconds[:-1])
    assert card_start == pytest.approx(t.starts[-1] - 0.15, abs=0.01)           # the card opens as the call to action starts


def test_shots_that_only_continue_a_line_share_its_time_equally():
    board = _board(((0,), (1,), (1,), ()))
    t = _timing([1.5, 4.0, 2.0], 15, board)
    assert t.shot_seconds[1] == pytest.approx(t.shot_seconds[2], abs=0.001)
    assert sum(t.shot_seconds) == pytest.approx(t.total_s, abs=0.01)


def test_a_shot_that_carries_two_lines_covers_both():
    board = _board(((0,), (1, 2), ()))
    durations = [1.5, 2.0, 2.0, 2.0]
    t = _timing(durations, 15, board)
    second_shot_start = t.shot_seconds[0]
    assert second_shot_start == pytest.approx(t.starts[1] - 0.15, abs=0.01)
    assert t.shot_seconds[1] >= durations[1] + durations[2]


def test_no_lines_leaves_the_plan_as_it_was():
    board = _board()
    t = voice.retime([], board, 15, CONFIG)
    assert t.shot_seconds == [s.seconds for s in board.shots] and t.total_s == 15


# -- the node --------------------------------------------------------------------------------

def _script() -> VideoScript:
    return VideoScript(lines=[ScriptLine(text="Welcome to Aurelia Heights.", role="hook"),
                              ScriptLine(text="On site: swimming pool and gymnasium.", role="feature"),
                              ScriptLine(text="Book a site visit.", role="cta")])


def _brief(**fields) -> VideoBrief:
    return VideoBrief(property_id="P", name="Aurelia Heights", duration_s=15, **fields)


def _service(tmp_path, provider) -> SpeechService:
    return SpeechService([provider], tmp_path / "speech_cache")


def test_every_line_is_spoken_timed_and_the_shots_follow(tmp_path):
    board = _board(((0,), (1,), ()))
    board.shots.insert(1, Shot(index=1, seconds=3.0, line_ids=[1]))        # keep indexes tidy: 0, 1, 2(card)
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0]), Shot(index=1, seconds=4.0, line_ids=[1]),
                              Shot(index=2, kind="card", camera="still", seconds=2.0, line_ids=[2])])
    result = voice.run(_script(), board, _brief(), tmp_path, _service(tmp_path, MockSpeechProvider()))

    assert len(result.line_timings) == len(result.voice_tracks) == 3
    for timing in result.line_timings:
        with wave.open(timing.path, "rb") as handle:
            assert handle.getnframes() / handle.getframerate() == pytest.approx(timing.end_s - timing.start_s, abs=0.01)
        assert timing.mode == "speech" and timing.provider == "mock"
    starts = [t.start_s for t in result.line_timings]
    assert starts == sorted(starts)
    assert result.total_s == pytest.approx(15.0, abs=0.5)
    assert sum(s.seconds for s in result.storyboard.shots) == pytest.approx(result.total_s, abs=0.01)
    assert result.storyboard.shots[0].seconds != 4.0                           # re-timed, and the input is not touched
    assert board.shots[0].seconds == 4.0
    assert result.voice_name == "mock/mock" and result.defects == []


def test_with_no_voice_the_track_is_silent_and_the_result_says_so(tmp_path):
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0, 1]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[2])])
    result = voice.run(_script(), board, _brief(), tmp_path, _service(tmp_path, SilentSpeechProvider()))

    assert result.voice_name == "silent"
    assert all(t.mode == "silent" for t in result.line_timings)
    assert any("No voice was available" in n for n in result.notes)
    assert result.total_s == pytest.approx(15.0, abs=0.5)                       # silent lines are timed by their words too


def test_a_voice_that_cannot_speak_is_noted_with_the_reason(tmp_path):
    from src.models.speech.base import SpeechError

    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0, 1]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[2])])
    service = SpeechService([MockSpeechProvider(fail=SpeechError("quota spent"))], tmp_path / "c")
    result = voice.run(_script(), board, _brief(), tmp_path, service)
    assert any("mock: quota spent" in n for n in result.notes)


def test_a_narration_too_long_for_the_video_is_a_defect_on_the_result(tmp_path):
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0, 1]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[2])])
    slow = MockSpeechProvider(words_per_second=0.5)
    result = voice.run(_script(), board, _brief(), tmp_path, _service(tmp_path, slow))
    assert result.total_s > 15.5 and len(result.defects) == 1


def test_the_voice_the_brief_asks_for_is_passed_to_the_service(tmp_path):
    mock = MockSpeechProvider()
    service = SpeechService([mock], tmp_path / "c", voices={"default": {"mock": "A"}, "deep": {"mock": "B"}})
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0, 1]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[2])])
    voice.run(_script(), board, _brief(voice="deep"), tmp_path, service)
    assert {v for _, v in mock.calls} == {"B"}


def test_with_no_music_provider_there_is_no_music_and_no_beats(tmp_path):
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0, 1]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[2])])
    result = voice.run(_script(), board, _brief(), tmp_path, _service(tmp_path, MockSpeechProvider()),
                       MusicService([], tmp_path / "music_cache"))
    assert result.music_track is None and result.beat_times == []
    assert result.music_info.provider == "none"


# -- no shot is left shorter than its floor -----------------------------------------------------------------------

def test_a_short_line_shared_by_two_shots_does_not_leave_either_under_a_second():
    board = _board(((0,), (1,), (1,), ()))
    t = _timing([3.0, 0.8, 8.0], 15, board)
    photo = t.shot_seconds[:-1]
    assert min(photo) >= 1.0 - 1e-6
    assert sum(t.shot_seconds) == pytest.approx(t.total_s, abs=0.01)
    assert any("were under 1 s and were lengthened to it" in n for n in t.notes)


def test_shots_over_the_floor_are_left_as_they_are():
    t = _timing([1.5, 2.5, 2.5, 2.0], 15)
    assert not any("under 1 s" in n for n in t.notes)


def test_the_floor_takes_its_time_from_the_longer_shots_in_proportion_to_what_they_have_above_it():
    seconds = [0.4, 3.0, 5.0]
    note = voice._lift_to_floor(seconds, 1.0)
    assert seconds == pytest.approx([1.0, 2.8, 4.6]) and sum(seconds) == pytest.approx(8.4)
    assert note.startswith("1 shot(s) were under 1 s and were lengthened to it")


def test_when_the_others_have_nothing_to_give_the_short_shots_stay_short_and_it_is_said():
    seconds = [0.5, 0.9, 1.0]
    note = voice._lift_to_floor(seconds, 1.0)
    assert seconds == [0.5, 0.9, 1.0] and note == "2 shot(s) are under 1 s and the others have no time to give."
    assert voice._lift_to_floor([2.0, 3.0], 1.0) == ""


# -- the closing card keeps to its cap; the spare time goes to the photograph shots ---------------------------------

def test_a_short_narration_leaves_the_closing_card_within_three_seconds_and_lengthens_the_shots():
    board = _board(((0,), (1,), (2,), ()))
    t = _timing([1.5, 1.5, 1.5, 1.5], 30, board)
    card = t.shot_seconds[-1]
    assert card <= 3.0 + 1e-6 and card >= 1.5
    assert sum(t.shot_seconds) == pytest.approx(t.total_s, abs=0.01) and t.total_s == pytest.approx(30, abs=0.01)
    assert sum(t.shot_seconds[:-1]) > 26                                                    # the time is in the shots


def test_the_card_takes_its_share_of_the_spare_time_only_up_to_the_cap():
    board = _board(((0,), (1,), (2,), ()))
    small = _timing([2.0, 2.0, 2.0, 2.0], 12, board)                                          # a little spare time
    assert small.shot_seconds[-1] <= 3.0 + 1e-6
    lax = voice.retime([1.5] * 4, board, 30, {**CONFIG, "voice": {**CONFIG["voice"], "max_end_card_s": 5.0}})
    assert 3.0 < lax.shot_seconds[-1] <= 5.0 + 1e-6                                          # the cap is the config's


def test_a_card_that_already_fills_its_cap_takes_none_of_the_spare_time():
    board = _board(((0,), (1,), (2,), ()))
    t = _timing([1.0, 1.0, 1.0, 2.9], 30, board)                    # the last line alone makes the card 0.15 + 2.9 + 0.3 s
    assert t.shot_seconds[-1] == pytest.approx(3.35, abs=0.01)
