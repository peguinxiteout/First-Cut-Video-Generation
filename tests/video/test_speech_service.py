"""The speech chain: who speaks when someone cannot, and what a repeat costs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.exceptions import ConfigError
from src.models import speech
from src.models.speech import SpeechService, get_speech_service
from src.models.speech.base import SpeechAuthError, SpeechError, SpeechQuotaError
from src.models.speech.gemini import GeminiSpeechProvider
from src.models.speech.mock import MockSpeechProvider
from src.models.speech.offline import OfflineSpeechProvider
from src.models.speech.silent import SilentSpeechProvider
from tests.video.fakes import FakePost, FakeResponse, wav_response
from tests.video.test_speech_offline import Runner, _which


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


def _gemini(post, key="k") -> GeminiSpeechProvider:
    return GeminiSpeechProvider(api_key=key, model="m", default_voice="v", url="https://example.invalid/x",
                                post=post)


def _working_offline() -> OfflineSpeechProvider:
    return OfflineSpeechProvider(Runner(), _which("espeak-ng"), platform="linux")


def _no_engine() -> OfflineSpeechProvider:
    return OfflineSpeechProvider(Runner(), _which(), platform="linux")


def _service(tmp_path, *providers, **kwargs) -> SpeechService:
    return SpeechService(list(providers), tmp_path / "cache", **kwargs)


LINE = "Welcome to Anamika High Point."


class TestFallback:
    def test_when_gemini_fails_a_local_voice_speaks_and_the_reason_is_recorded(self, tmp_path):
        service = _service(tmp_path, _gemini(FakePost(FakeResponse(401, text="no"))), _working_offline())

        audio = service.synthesize(LINE, "", tmp_path / "line.wav")

        assert (audio.provider, audio.mode) == ("offline", "speech")
        assert len(audio.notes) == 1 and audio.notes[0].startswith("gemini: Google rejected the key")
        assert (tmp_path / "line.wav").exists()

    def test_with_no_voice_anywhere_the_track_is_silent_and_both_reasons_are_recorded(self, tmp_path):
        service = _service(tmp_path, _gemini(FakePost(wav_response()), key=""), _no_engine())

        audio = service.synthesize(" ".join(["w"] * 21), "", tmp_path / "line.wav")

        assert (audio.provider, audio.mode) == ("silent", "silent")
        assert audio.duration_s == pytest.approx(10.0, abs=0.01)
        assert audio.notes == ["gemini: GEMINI_API_KEY is not set",
                               "offline: no speech engine on this machine (needs Windows SAPI or espeak-ng)"]

    def test_a_transient_failure_costs_that_line_only(self, tmp_path):
        post = FakePost(FakeResponse(503), FakeResponse(503), FakeResponse(503), wav_response(0.5))
        gemini = _gemini(post)
        gemini._max_retries = 3
        service = _service(tmp_path, gemini, _working_offline())

        first = service.synthesize("First line.", "", tmp_path / "1.wav")
        second = service.synthesize("Second line.", "", tmp_path / "2.wav")

        assert first.provider == "offline" and "busy" in first.notes[0]
        assert second.provider == "gemini" and second.notes == []
        assert service.dropped == {}

    def test_a_quota_or_auth_error_drops_the_provider_for_the_rest_of_the_run(self, tmp_path):
        post = FakePost(FakeResponse(429))
        service = _service(tmp_path, _gemini(post), _working_offline())

        service.synthesize("First line.", "", tmp_path / "1.wav")
        second = service.synthesize("Second line.", "", tmp_path / "2.wav")

        assert len(post.calls) == 1
        assert second.provider == "offline" and "skipped for this run" in second.notes[0]
        assert set(service.dropped) == {"gemini"}

    def test_the_failure_of_the_last_real_provider_still_ends_in_silence(self, tmp_path):
        service = _service(tmp_path, MockSpeechProvider(fail=SpeechQuotaError("spent")))
        audio = service.synthesize(LINE, "", tmp_path / "line.wav")
        assert audio.mode == "silent" and audio.notes == ["mock: spent"]

    def test_a_silent_track_is_always_added_at_the_end(self, tmp_path):
        service = _service(tmp_path, MockSpeechProvider())
        assert isinstance(service.providers[-1], SilentSpeechProvider)
        assert len([p for p in _service(tmp_path, SilentSpeechProvider()).providers
                    if isinstance(p, SilentSpeechProvider)]) == 1

    def test_an_output_folder_that_cannot_exist_is_an_error_not_a_silent_swallow(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(OSError):
            _service(tmp_path).synthesize(LINE, "", blocker / "line.wav")


class TestCache:
    def test_the_same_line_twice_speaks_once(self, tmp_path):
        mock = MockSpeechProvider()
        service = _service(tmp_path, mock)

        first = service.synthesize(LINE, "", tmp_path / "a.wav")
        again = service.synthesize(LINE, "", tmp_path / "b.wav")

        assert (first.from_cache, again.from_cache) == (False, True)
        assert len(mock.calls) == 1
        assert again.duration_s == first.duration_s and (tmp_path / "b.wav").exists()

    def test_a_different_text_or_voice_is_a_miss(self, tmp_path):
        mock = MockSpeechProvider()
        service = _service(tmp_path, mock, voices={"default": {"mock": "A"}, "other": {"mock": "B"}})

        service.synthesize(LINE, "", tmp_path / "a.wav")
        service.synthesize(LINE + " Visit today.", "", tmp_path / "b.wav")
        service.synthesize(LINE, "other", tmp_path / "c.wav")
        service.synthesize(LINE, "other", tmp_path / "d.wav")             # this one is a hit

        assert [voice for _, voice in mock.calls] == ["A", "A", "B"]

    def test_a_cached_line_needs_no_key(self, tmp_path):
        post = FakePost(wav_response(0.5))
        _service(tmp_path, _gemini(post)).synthesize(LINE, "", tmp_path / "a.wav")
        offline_run = _service(tmp_path, _gemini(post, key=""), _no_engine())

        audio = offline_run.synthesize(LINE, "", tmp_path / "b.wav")

        assert audio.from_cache and audio.provider == "gemini" and len(post.calls) == 1

    def test_the_silent_track_is_not_cached(self, tmp_path):
        service = _service(tmp_path)
        service.synthesize(LINE, "", tmp_path / "a.wav")
        assert not (tmp_path / "cache").exists() or list((tmp_path / "cache").glob("*.wav")) == []


class TestVoices:
    def test_each_provider_is_asked_for_its_own_voice(self, tmp_path):
        mock = MockSpeechProvider()
        voices = {"default": {"mock": "mock-voice", "gemini": "g-voice"}}
        _service(tmp_path, mock, voices=voices).synthesize(LINE, "", tmp_path / "a.wav")
        assert mock.calls[0][1] == "mock-voice"

    def test_a_null_voice_leaves_the_provider_on_its_own_default(self, tmp_path):
        mock = MockSpeechProvider()
        _service(tmp_path, mock, voices={"default": {"mock": None}}).synthesize(LINE, "", tmp_path / "a.wav")
        assert mock.calls[0][1] is None

    def test_an_unknown_voice_falls_back_to_the_default_and_says_so(self, tmp_path):
        mock = MockSpeechProvider()
        service = _service(tmp_path, mock, voices={"default": {"mock": "A"}})
        audio = service.synthesize(LINE, "baritone", tmp_path / "a.wav")
        assert mock.calls[0][1] == "A"
        assert audio.notes == ["unknown voice 'baritone'; using 'default'"]


class TestBuilding:
    def _settings(self, tmp_path, provider="auto", key=""):
        return SimpleNamespace(speech_provider=provider, gemini_api_key=key, gemini_tts_model="gemini-test-tts",
                               gemini_tts_voice="voice-a", data_dir_resolved=tmp_path / "data")

    def test_auto_is_gemini_then_offline_then_silent_from_the_yaml(self, tmp_path):
        service = get_speech_service(self._settings(tmp_path))
        assert [p.name for p in service.providers] == ["gemini", "offline", "silent"]
        gemini = service.providers[0]
        assert gemini.is_available() is False and gemini.model_id == "gemini-test-tts"
        assert service.cache_dir == tmp_path / "data" / "speech_cache"
        assert service.default_voice == "default" and "default" in service.voices

    def test_a_key_makes_gemini_available(self, tmp_path):
        assert get_speech_service(self._settings(tmp_path, key="k")).providers[0].is_available()

    def test_a_named_provider_is_followed_by_silence(self, tmp_path):
        service = get_speech_service(self._settings(tmp_path, provider="offline"))
        assert [p.name for p in service.providers] == ["offline", "silent"]
        assert [p.name for p in get_speech_service(self._settings(tmp_path, provider="silent")).providers] == ["silent"]

    def test_the_cache_folder_follows_the_current_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.config.settings.data_dir", str(tmp_path / "elsewhere"), raising=False)
        assert get_speech_service().cache_dir == (tmp_path / "elsewhere").resolve() / "speech_cache"

    def test_an_unknown_provider_name_is_a_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="polly"):
            get_speech_service(self._settings(tmp_path, provider="polly"))

    def test_with_no_key_the_default_chain_still_makes_a_track(self, tmp_path, monkeypatch):
        monkeypatch.setattr(speech, "OfflineSpeechProvider", _no_engine)
        service = get_speech_service(self._settings(tmp_path))
        audio = service.synthesize(LINE, "", tmp_path / "a.wav")
        assert audio.mode == "silent" and len(audio.notes) == 2


def test_speech_errors_are_platform_errors_and_only_transport_is_retryable():
    from src.core.exceptions import ModelProviderError, PlatformError
    from src.models.speech.base import SpeechTransportError

    assert issubclass(SpeechError, PlatformError)
    assert issubclass(SpeechTransportError, ModelProviderError)
    assert not issubclass(SpeechAuthError, ModelProviderError)
    assert not issubclass(SpeechQuotaError, ModelProviderError)
