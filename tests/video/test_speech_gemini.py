"""Gemini speech through a fake transport: the request, the audio decode, and
what each status code means."""
from __future__ import annotations

import wave

import pytest

from src.models.speech.base import (
    SpeechAuthError, SpeechError, SpeechQuotaError, SpeechTransportError, wav_duration,
)
from src.models.speech.gemini import GeminiSpeechProvider
from tests.video.fakes import FakePost, FakeResponse, audio_response, silence, wav_response

URL = "https://example.invalid/v1beta/interactions"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


def _provider(post, **kwargs) -> GeminiSpeechProvider:
    fields = dict(api_key="test-key", model="gemini-test-tts", default_voice="voice-a", url=URL,
                  style="brisk", post=post)
    fields.update(kwargs)
    return GeminiSpeechProvider(**fields)


def test_the_request_has_the_shape_the_interactions_api_wants(tmp_path):
    post = FakePost(wav_response())
    _provider(post).synthesize("Welcome to Anamika High Point.", None, tmp_path / "a.wav")

    url, kwargs = post.calls[0]
    assert url == URL
    assert kwargs["headers"] == {"x-goog-api-key": "test-key", "Content-Type": "application/json"}
    assert kwargs["timeout"] == (10.0, 180.0)
    body = kwargs["json"]
    assert body["model"] == "gemini-test-tts"
    content = body["input"][0]["content"][0]
    assert body["input"][0]["type"] == "user_input"
    assert content["text"] == "Welcome to Anamika High Point."
    assert content["annotations"] == [{"type": "speech_metadata", "style": "brisk"}]
    assert body["response_format"] == {"type": "audio", "mime_type": "audio/wav", "sample_rate": 24000}
    assert body["generation_config"] == {"speech_config": [{"voice": "voice-a"}]}


def test_a_voice_asked_for_overrides_the_default_and_no_style_means_no_annotation(tmp_path):
    post = FakePost(wav_response())
    _provider(post, style="").synthesize("Hello.", "voice-b", tmp_path / "a.wav")
    body = post.calls[0][1]["json"]
    assert body["generation_config"]["speech_config"] == [{"voice": "voice-b"}]
    assert "annotations" not in body["input"][0]["content"][0]


def test_a_wav_answer_is_written_as_it_came(tmp_path):
    audio = _provider(FakePost(wav_response(0.5))).synthesize("Hello.", None, tmp_path / "a.wav")
    assert (audio.provider, audio.mode, audio.from_cache) == ("gemini", "speech", False)
    assert audio.model_id == "gemini-test-tts" and audio.voice == "voice-a"
    assert audio.duration_s == 0.5 == wav_duration(tmp_path / "a.wav")


def test_raw_pcm_is_wrapped_as_wav_at_the_rate_the_answer_names(tmp_path):
    pcm = silence(1.0, 16000)
    audio = _provider(FakePost(audio_response(pcm, "audio/L16;rate=16000"))).synthesize(
        "Hello.", None, tmp_path / "a.wav")
    with wave.open(str(tmp_path / "a.wav"), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (16000, 1, 2)
    assert audio.duration_s == 1.0


def test_raw_pcm_with_no_rate_uses_the_configured_one(tmp_path):
    audio = _provider(FakePost(audio_response(silence(1.0, 24000), "audio/pcm")), sample_rate=24000).synthesize(
        "Hello.", None, tmp_path / "a.wav")
    assert audio.duration_s == 1.0


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_an_auth_error_and_is_not_retried(tmp_path, status):
    post = FakePost(FakeResponse(status, text="denied"))
    with pytest.raises(SpeechAuthError, match="GEMINI_API_KEY"):
        _provider(post).synthesize("Hello.", None, tmp_path / "a.wav")
    assert len(post.calls) == 1


def test_google_reports_a_bad_key_as_a_400_and_it_is_still_auth(tmp_path):
    post = FakePost(FakeResponse(400, text="API key not valid. Please pass a valid API key."))
    with pytest.raises(SpeechAuthError):
        _provider(post).synthesize("Hello.", None, tmp_path / "a.wav")


def test_429_is_a_quota_error_and_is_not_retried(tmp_path):
    post = FakePost(FakeResponse(429, text="RESOURCE_EXHAUSTED"))
    with pytest.raises(SpeechQuotaError):
        _provider(post).synthesize("Hello.", None, tmp_path / "a.wav")
    assert len(post.calls) == 1


def test_a_busy_service_is_retried_until_it_answers(tmp_path):
    post = FakePost(FakeResponse(503), FakeResponse(500), wav_response(0.5))
    audio = _provider(post).synthesize("Hello.", None, tmp_path / "a.wav")
    assert audio.duration_s == 0.5 and len(post.calls) == 3


def test_a_service_that_stays_busy_gives_up_with_a_transport_error(tmp_path):
    post = FakePost(FakeResponse(503))
    with pytest.raises(SpeechTransportError, match="busy"):
        _provider(post, max_retries=3).synthesize("Hello.", None, tmp_path / "a.wav")
    assert len(post.calls) == 3


def test_a_dropped_connection_is_a_retried_transport_error(tmp_path):
    post = FakePost(ConnectionError("reset by peer"), wav_response(0.5))
    audio = _provider(post).synthesize("Hello.", None, tmp_path / "a.wav")
    assert audio.duration_s == 0.5 and len(post.calls) == 2


def test_a_timeout_that_never_clears_is_a_transport_error(tmp_path):
    post = FakePost(TimeoutError("read timed out"))
    with pytest.raises(SpeechTransportError, match="did not answer in time"):
        _provider(post, max_retries=2).synthesize("Hello.", None, tmp_path / "a.wav")
    assert len(post.calls) == 2


def test_an_answer_with_no_audio_is_an_error_not_a_silent_file(tmp_path):
    post = FakePost(FakeResponse(200, {"steps": [{"type": "model_output", "content": [{"type": "text"}]}]}))
    with pytest.raises(SpeechError, match="without any audio") as caught:
        _provider(post).synthesize("Hello.", None, tmp_path / "a.wav")
    assert not isinstance(caught.value, SpeechTransportError)
    assert len(post.calls) == 1 and not (tmp_path / "a.wav").exists()


def test_other_refusals_are_plain_errors_with_the_status(tmp_path):
    with pytest.raises(SpeechError, match="HTTP 400"):
        _provider(FakePost(FakeResponse(400, text="bad voice"))).synthesize("Hello.", None, tmp_path / "a.wav")


def test_without_a_key_it_is_unavailable_and_never_calls_out(tmp_path):
    post = FakePost(wav_response())
    provider = _provider(post, api_key="")
    assert provider.is_available() is False
    assert provider.unavailable_reason() == "GEMINI_API_KEY is not set"
    with pytest.raises(SpeechAuthError):
        provider.synthesize("Hello.", None, tmp_path / "a.wav")
    assert post.calls == []
