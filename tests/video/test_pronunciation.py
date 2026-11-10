"""Pronunciation: respelled words reach the voice only; captions, titles and the card keep the real spelling."""
from __future__ import annotations

import pytest

from src.agents.video import brief as brief_agent
from src.agents.video import overlays, voice
from src.core.config import load_yaml_config
from src.data.video_schemas import ScriptLine, Shot, Storyboard, VideoBrief, VideoScript
from src.models.music import MusicService
from src.models.speech import SpeechService
from src.models.speech.mock import MockSpeechProvider
from src.models.speech.pronunciation import merge, respell
from tests.video.helpers import make_video_property


# -- the substitution -------------------------------------------------------------------------------------

def test_a_mapped_word_is_replaced_in_any_case():
    mapping = {"Anamika": "Anaamika"}
    assert respell("Meet Anamika High Point. ANAMIKA, anamika!", mapping) == "Meet Anaamika High Point. Anaamika, Anaamika!"


def test_only_whole_words_are_replaced():
    assert respell("Anamikas and PreAnamika and Anamika", {"Anamika": "X"}) == "Anamikas and PreAnamika and X"


def test_a_name_with_dots_and_spaces_is_one_entry_and_the_longest_match_wins():
    mapping = {"S.G. Highway": "S G Highway", "Highway": "Hiway"}
    assert respell("On S.G. Highway, off the Highway.", mapping) == "On S G Highway, off the Hiway."
    assert respell("s.g. highway", mapping) == "S G Highway"


def test_a_respelling_is_never_respelled():
    assert respell("Ahmedabad", {"Ahmedabad": "Ahmedaabaad", "Ahmedaabaad": "loop"}) == "Ahmedaabaad"


def test_no_map_or_no_match_leaves_the_text_alone():
    assert respell("Nothing to change.", {}) == "Nothing to change."
    assert respell("Nothing to change.", {"Anamika": "x"}) == "Nothing to change."


def test_maps_merge_with_the_later_winning_and_keys_compared_without_case():
    assert merge({"Anamika": "one", "Ahmedabad": "two"}, {"anamika": "three"}, None) == {"anamika": "three", "Ahmedabad": "two"}
    assert merge({"": "x", "a": ""}) == {}


# -- the shipped map -----------------------------------------------------------------------------------------

def test_speech_yaml_respells_the_names_the_demo_property_needs():
    mapping = load_yaml_config("video/speech.yaml")["pronunciations"]
    keys = {k.lower() for k in mapping}
    assert {"anamika", "ahmedabad", "s.g. highway"} <= keys
    assert all(isinstance(v, str) and v.strip() for v in mapping.values())


# -- in the voice step --------------------------------------------------------------------------------------

def _run(tmp_path, brief_fields=None, text="Meet Anamika High Point, in Ahmedabad, off S.G. Highway."):
    script = VideoScript(lines=[ScriptLine(text=text, role="hook"), ScriptLine(text="Book a visit.", role="cta")],
                         on_screen=["Anamika High Point"], cta="Book a visit")
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[1])])
    brief = VideoBrief(property_id="P", name="Anamika High Point", duration_s=10, quality="draft", **(brief_fields or {}))
    mock = MockSpeechProvider()
    service = SpeechService([mock], tmp_path / "speech")
    result = voice.run(script, board, brief, tmp_path / "out", service, MusicService([], tmp_path / "m"))
    return script, brief, mock, result


def test_the_voice_is_given_the_respelled_text_and_the_script_keeps_the_real_one(tmp_path, monkeypatch):
    real = load_yaml_config
    monkeypatch.setattr(voice, "load_yaml_config", lambda name: {**real(name), "pronunciations": {
        "Anamika": "Anaamika", "Ahmedabad": "Ahmedaabaad", "S.G. Highway": "S G Highway"}}
        if name == "video/speech.yaml" else real(name))
    script, _, mock, result = _run(tmp_path)

    spoken = "Meet Anaamika High Point, in Ahmedaabaad, off S G Highway."
    assert mock.calls[0][0] == spoken
    assert result.line_timings[0].spoken == spoken
    assert script.lines[0].text == "Meet Anamika High Point, in Ahmedabad, off S.G. Highway."
    assert result.line_timings[1].spoken == "Book a visit."                    # nothing to respell


def test_captions_and_overlays_keep_the_real_spelling(tmp_path, monkeypatch):
    real = load_yaml_config
    monkeypatch.setattr(voice, "load_yaml_config", lambda name: {**real(name), "pronunciations": {"Anamika": "Anaamika"}}
                        if name == "video/speech.yaml" else real(name))
    script, brief, _, result = _run(tmp_path)

    captions = overlays.srt(result.line_timings, script, 80)
    assert "Anamika High Point" in captions and "Anaamika" not in captions
    assert script.on_screen == ["Anamika High Point"]


def test_a_property_s_own_pronunciations_win_over_the_shared_map(tmp_path, monkeypatch):
    real = load_yaml_config
    monkeypatch.setattr(voice, "load_yaml_config", lambda name: {**real(name), "pronunciations": {"Anamika": "shared"}}
                        if name == "video/speech.yaml" else real(name))
    _, _, mock, _ = _run(tmp_path, {"pronunciations": {"anamika": "own", "Constera": "Konstera"}},
                         text="Anamika by Constera.")
    assert mock.calls[0][0] == "own by Konstera."


def test_the_pronunciations_come_from_the_properties_video_input_json(tmp_path):
    folder = make_video_property(tmp_path / "p", video_input={"pronunciations": {" Aurelia ": " Aurelya ", "": "x", "Blank": ""}})
    loaded = brief_agent.run(folder, draft=True)
    assert loaded.brief.pronunciations == {"Aurelia": "Aurelya"}
    assert brief_agent.run(make_video_property(tmp_path / "q"), draft=True).brief.pronunciations == {}


def test_the_cache_keys_on_what_is_spoken_so_a_new_respelling_is_a_new_take(tmp_path, monkeypatch):
    _, _, mock, _ = _run(tmp_path, {"pronunciations": {"Anamika": "Anaamika"}}, text="Anamika.")
    _, _, again, _ = _run(tmp_path / "again", {"pronunciations": {"Anamika": "Ahnamika"}}, text="Anamika.")
    assert mock.calls[0][0] == "Anaamika." and again.calls[0][0] == "Ahnamika."
