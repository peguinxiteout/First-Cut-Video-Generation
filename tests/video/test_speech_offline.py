"""The voices that need no key: a voice on this machine, a silent track, and
the stand-in used by tests."""
from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from types import SimpleNamespace

import pytest

from src.models.speech.base import SpeechError, pcm_to_wav, wav_duration
from src.models.speech.mock import MockSpeechProvider
from src.models.speech.offline import OfflineSpeechProvider
from src.models.speech.silent import SilentSpeechProvider
from src.rendering import ffmpeg
from tests.video.fakes import silence

LINE = 'Say "hello" to $HOME `now`; it\'s 3 BHK.'


def _which(*available):
    return lambda name: f"C:/bin/{name}" if name in available else None


class Runner:
    """A subprocess.run that records its call and, like the real engines,
    writes a WAV where it was told to (or fails as scripted)."""

    def __init__(self, returncode: int = 0, write: bool = True, error: Exception | None = None):
        self.returncode, self.write, self.error = returncode, write, error
        self.calls: list[tuple[list, dict]] = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        if self.error:
            raise self.error
        if self.write and self.returncode == 0:
            out = (kwargs.get("env") or {}).get("XT_OUT") or args[args.index("-w") + 1]
            with open(out, "wb") as handle:
                handle.write(pcm_to_wav(silence(1.0, 16000), 16000))
        return SimpleNamespace(returncode=self.returncode, stdout="", stderr="engine said no")


# -- Windows SAPI -------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell.exe"),
                    reason="Windows SAPI is only on Windows")
def test_windows_sapi_really_speaks_a_line(tmp_path):
    provider = OfflineSpeechProvider()
    assert provider.backend() == "sapi"
    audio = provider.synthesize("Welcome home.", "female", tmp_path / "line.wav")
    assert (audio.provider, audio.model_id, audio.mode) == ("offline", "sapi", "speech")
    assert 0.3 < audio.duration_s < 5
    assert wav_duration(tmp_path / "line.wav") == audio.duration_s
    loud = ffmpeg.measure_loudness(tmp_path / "line.wav")
    assert loud is not None and loud.integrated_lufs > -60          # audible, not silence


def test_sapi_gets_the_text_through_the_environment_never_the_command_line(tmp_path):
    runner = Runner()
    provider = OfflineSpeechProvider(runner, _which("powershell.exe"), platform="win32")

    audio = provider.synthesize(LINE, "male", tmp_path / "line.wav")

    args, kwargs = runner.calls[0]
    assert args[0] == "C:/bin/powershell.exe" and "-NoProfile" in args
    assert LINE not in " ".join(args)
    assert kwargs["env"]["XT_TEXT"] == LINE and kwargs["env"]["XT_VOICE"] == "male"
    assert kwargs["env"]["XT_OUT"] == str(tmp_path / "line.wav")
    assert audio.duration_s == 1.0 and audio.model_id == "sapi" and audio.voice == "male"


# -- espeak-ng ---------------------------------------------------------------------------

def test_espeak_ng_is_used_when_there_is_no_sapi(tmp_path):
    runner = Runner()
    provider = OfflineSpeechProvider(runner, _which("espeak-ng"), platform="linux")
    assert provider.backend() == "espeak-ng"

    audio = provider.synthesize(LINE, "female", tmp_path / "line.wav")

    args, kwargs = runner.calls[0]
    assert args == ["C:/bin/espeak-ng", "-v", "en+f3", "-s", "165", "-w", str(tmp_path / "line.wav"), "--stdin"]
    assert kwargs["input"] == LINE
    assert audio.model_id == "espeak-ng"


def test_sapi_is_preferred_over_espeak_on_windows(tmp_path):
    provider = OfflineSpeechProvider(Runner(), _which("powershell.exe", "espeak-ng"), platform="win32")
    assert provider.backend() == "sapi"


def test_powershell_is_never_used_off_windows():
    provider = OfflineSpeechProvider(Runner(), _which("powershell.exe"), platform="linux")
    assert provider.is_available() is False


# -- when it cannot ----------------------------------------------------------------------

def test_with_no_engine_it_is_unavailable_and_says_what_is_missing(tmp_path):
    provider = OfflineSpeechProvider(Runner(), _which(), platform="linux")
    assert provider.is_available() is False
    assert "Windows SAPI or espeak-ng" in provider.unavailable_reason()
    with pytest.raises(SpeechError, match="no speech engine"):
        provider.synthesize("Hello.", None, tmp_path / "a.wav")


def test_an_engine_that_fails_is_a_speech_error_with_its_message(tmp_path):
    provider = OfflineSpeechProvider(Runner(returncode=1), _which("espeak-ng"), platform="linux")
    with pytest.raises(SpeechError, match="engine said no"):
        provider.synthesize("Hello.", None, tmp_path / "a.wav")


def test_an_engine_that_exits_cleanly_but_writes_nothing_is_an_error(tmp_path):
    provider = OfflineSpeechProvider(Runner(write=False), _which("espeak-ng"), platform="linux")
    with pytest.raises(SpeechError, match="no readable audio"):
        provider.synthesize("Hello.", None, tmp_path / "a.wav")


def test_a_hung_engine_is_a_speech_error(tmp_path):
    runner = Runner(error=subprocess.TimeoutExpired("espeak-ng", 90))
    provider = OfflineSpeechProvider(runner, _which("espeak-ng"), platform="linux")
    with pytest.raises(SpeechError, match="did not finish"):
        provider.synthesize("Hello.", None, tmp_path / "a.wav")


# -- the silent track ---------------------------------------------------------------------

def test_a_silent_track_is_as_long_as_the_words_take_to_say(tmp_path):
    provider = SilentSpeechProvider(words_per_second=2.1)
    line = " ".join(["word"] * 21)

    audio = provider.synthesize(line, None, tmp_path / "s.wav")

    assert audio.mode == "silent" and audio.provider == "silent"
    assert audio.duration_s == pytest.approx(10.0, abs=0.001)
    assert wav_duration(tmp_path / "s.wav") == pytest.approx(10.0, abs=0.001)


@pytest.mark.parametrize("words, seconds", [(8, 3.81), (32, 15.238), (64, 30.476)])
def test_the_silent_length_scales_with_the_word_count(tmp_path, words, seconds):
    audio = SilentSpeechProvider(2.1).synthesize(" ".join(["w"] * words), None, tmp_path / "s.wav")
    assert audio.duration_s == pytest.approx(seconds, abs=0.01)


def test_a_very_short_line_still_gets_a_minimum_and_it_really_is_silent(tmp_path):
    audio = SilentSpeechProvider().synthesize("Hi", None, tmp_path / "s.wav")
    assert audio.duration_s == 0.6
    with wave.open(str(tmp_path / "s.wav"), "rb") as handle:
        assert set(handle.readframes(handle.getnframes())) == {0}
    assert SilentSpeechProvider().synthesize("", None, tmp_path / "e.wav").duration_s == 0.6


def test_a_slower_pace_makes_a_longer_track(tmp_path):
    line = " ".join(["w"] * 20)
    fast = SilentSpeechProvider(2.5).synthesize(line, None, tmp_path / "f.wav").duration_s
    slow = SilentSpeechProvider(1.5).synthesize(line, None, tmp_path / "s.wav").duration_s
    assert slow > fast


# -- the stand-in --------------------------------------------------------------------------

def test_the_mock_voice_is_audible_and_follows_the_word_count(tmp_path):
    mock = MockSpeechProvider(words_per_second=2.0)
    audio = mock.synthesize(" ".join(["w"] * 10), "v", tmp_path / "m.wav")
    assert audio.duration_s == 5.0 and audio.mode == "speech"
    assert mock.calls == [(" ".join(["w"] * 10), "v")]
    loud = ffmpeg.measure_loudness(tmp_path / "m.wav")
    assert loud is not None and loud.integrated_lufs > -40


def test_the_mock_voice_can_be_told_to_fail(tmp_path):
    with pytest.raises(SpeechError, match="boom"):
        MockSpeechProvider(fail=SpeechError("boom")).synthesize("Hello.", None, tmp_path / "m.wav")
