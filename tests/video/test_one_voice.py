"""One voice per video: if the first provider fails on any line, every line is voiced again by the next."""
from __future__ import annotations

import pytest

from src.agents.video import voice
from src.data.video_schemas import ScriptLine, Shot, Storyboard, VideoBrief, VideoScript
from src.models.music import MusicService
from src.models.speech import SpeechService
from src.models.speech.base import SpeechAuthError, SpeechError, SpeechQuotaError, SpeechTransportError
from src.models.speech.mock import MockSpeechProvider
from src.models.speech.silent import SilentSpeechProvider


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


class Named(MockSpeechProvider):
    """A mock voice that goes by another name and fails on the attempt numbers it is given.
    `calls` holds the lines it spoke; `attempts` counts every try."""

    def __init__(self, name, fail_calls=(), error=None, available=True):
        super().__init__()
        self.name, self.model_id = name, f"{name}-model"
        self.fail_calls, self.error, self._available = set(fail_calls), error or SpeechQuotaError("429"), available
        self.attempts = 0

    def is_available(self):
        return self._available

    def unavailable_reason(self):
        return f"{self.name} has no key"

    def synthesize(self, text, voice, output_path):
        self.attempts += 1
        if self.attempts in self.fail_calls:
            raise self.error
        return super().synthesize(text, voice, output_path)


def _service(tmp_path, *providers) -> SpeechService:
    return SpeechService(list(providers), tmp_path / "cache")


def _speak(tmp_path, service, n=4):
    texts = [f"Line number {i} of the script." for i in range(n)]
    return service.synthesize_all(texts, "", [tmp_path / "out" / f"l{i}.wav" for i in range(n)])


def test_when_the_first_provider_speaks_every_line_nothing_is_redone(tmp_path):
    first, second = Named("gemini"), Named("offline")
    audios = _speak(tmp_path, _service(tmp_path, first, second))
    assert {a.provider for a in audios} == {"gemini"} and len(first.calls) == 4 and second.calls == []
    assert all(not a.notes for a in audios)


def test_a_failure_on_a_later_line_voices_every_line_again_with_the_next_provider(tmp_path):
    first, second = Named("gemini", fail_calls={3}), Named("offline")
    audios = _speak(tmp_path, _service(tmp_path, first, second))

    assert {a.provider for a in audios} == {"offline"}                          # one voice, not two
    assert sorted(c[0] for c in second.calls) == [f"Line number {i} of the script." for i in range(4)]    # all four
    assert [a.path.name for a in audios] == [f"l{i}.wav" for i in range(4)] and all(a.path.is_file() for a in audios)


def test_the_reason_is_recorded_on_every_line(tmp_path):
    first, second = Named("gemini", fail_calls={3}, error=SpeechQuotaError("Gemini speech allowance reached (429).")), Named("offline")
    audios = _speak(tmp_path, _service(tmp_path, first, second))
    wanted = ("gemini could not speak line 3 (Gemini speech allowance reached (429).): every line was voiced again "
              "by offline, so the video has one voice.")
    assert all(wanted in a.notes for a in audios)


def test_a_transport_error_on_one_line_counts_too(tmp_path):
    first, second = Named("gemini", fail_calls={2}, error=SpeechTransportError("timed out")), Named("offline")
    audios = _speak(tmp_path, _service(tmp_path, first, second))
    assert {a.provider for a in audios} == {"offline"} and "could not speak line 2 (timed out)" in audios[0].notes[-1]


def test_a_failure_on_the_first_line_is_one_voice_already_and_nothing_is_redone(tmp_path):
    first, second = Named("gemini", fail_calls={1}), Named("offline")
    audios = _speak(tmp_path, _service(tmp_path, first, second))
    assert {a.provider for a in audios} == {"offline"} and len(second.calls) == 4
    assert not any("voiced again" in n for a in audios for n in a.notes)           # the plain per-line note says why instead
    assert any(n.startswith("gemini:") for n in audios[0].notes)


def test_a_provider_that_is_not_available_is_one_voice_too(tmp_path):
    audios = _speak(tmp_path, _service(tmp_path, Named("gemini", available=False), Named("offline")))
    assert {a.provider for a in audios} == {"offline"} and audios[0].notes == ["gemini: gemini has no key"]


def test_it_falls_down_the_chain_as_far_as_it_has_to(tmp_path):
    first, second, third = Named("gemini", fail_calls={4}), Named("offline", fail_calls={2}), Named("backup")
    audios = _speak(tmp_path, _service(tmp_path, first, second, third))
    assert {a.provider for a in audios} == {"backup"}
    note = audios[0].notes[-1]
    assert "gemini could not speak line 4" in note and "voiced again by backup" in note
    assert any(n.startswith("offline:") for n in audios[0].notes)               # and why offline did not speak is on the line


def test_the_silent_track_is_the_last_resort_and_is_one_voice_too(tmp_path):
    first = Named("gemini", fail_calls={2})
    audios = _speak(tmp_path, _service(tmp_path, first, SilentSpeechProvider(2.1)))
    assert {(a.provider, a.mode) for a in audios} == {("silent", "silent")}


def test_a_provider_dropped_for_the_run_is_not_asked_again_in_the_redo(tmp_path):
    first, second = Named("gemini", fail_calls={3}, error=SpeechAuthError("rejected")), Named("offline")
    service = _service(tmp_path, first, second)
    _speak(tmp_path, service)
    assert first.attempts == 3                                                      # lines 1-3 only: nothing after the refusal
    assert "gemini" in service.dropped


def test_with_every_provider_failing_the_silent_track_that_every_service_ends_with_speaks_all(tmp_path):
    audios = _speak(tmp_path, _service(tmp_path, Named("gemini", fail_calls={1, 2, 3, 4})))
    assert {(a.provider, a.mode) for a in audios} == {("silent", "silent")}


def test_a_later_run_reuses_the_lines_the_redo_made_and_the_ones_gemini_made_are_not_mixed_in(tmp_path):
    first = _speak(tmp_path, _service(tmp_path, Named("gemini", fail_calls={4}), Named("offline")))
    offline = Named("offline")
    # gemini's first three lines are in the cache; the fourth is not, and fails again
    second = _speak(tmp_path, _service(tmp_path, Named("gemini", fail_calls={1}), offline))
    assert {a.provider for a in first} == {a.provider for a in second} == {"offline"}
    assert offline.attempts == 0 and all(a.from_cache for a in second)           # all four came from the redo's cache


def test_the_failure_list_is_on_the_audio_a_fallback_provider_returns(tmp_path):
    audio = _service(tmp_path, Named("gemini", fail_calls={1}), Named("offline")).synthesize("Hello there.", "", tmp_path / "x.wav")
    assert audio.provider == "offline" and audio.failed == [("gemini", "429")]


# -- in the voice step ----------------------------------------------------------------------------------------------------

def test_the_voice_step_voices_the_whole_script_in_one_voice_and_says_why(tmp_path):
    script = VideoScript(lines=[ScriptLine(text=f"Line {i} here.", role="feature") for i in range(4)])
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0, 1, 2]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[3])])
    brief = VideoBrief(property_id="P", name="Aurelia", duration_s=15, quality="draft")
    service = _service(tmp_path, Named("gemini", fail_calls={3}), Named("offline"))
    result = voice.run(script, board, brief, tmp_path / "out", service, MusicService([], tmp_path / "m"))

    assert {t.provider for t in result.line_timings} == {"offline"} and result.voice_name.startswith("offline/")
    assert any("every line was voiced again by offline, so the video has one voice" in n for n in result.notes)
    assert "silent" not in result.voice_name
