"""Google Cloud TTS (Chirp 3: HD) through a fake transport: the request, the audio decode, what each status
code means, the custom pronunciations, and the chain it leads."""
from __future__ import annotations

import base64

import pytest

from src.models.speech import SpeechService, get_speech_service
from src.models.speech.base import (
    SpeechAuthError, SpeechError, SpeechQuotaError, SpeechTransportError, pcm_to_wav, wav_duration,
)
from src.models.speech.cloud_tts import CloudTTSProvider
from src.models.speech.gemini import GeminiSpeechProvider
from src.models.speech.mock import MockSpeechProvider
from tests.video.fakes import FakePost, FakeResponse, silence, wav_response

URL = "https://example.invalid/v1/text:synthesize"
PHONEMES = {"Anamika": "ənɑːmɪkɑː", "Velachery": "ˈʋeːləˌtʃeːɾi"}
RESPELLINGS = {"Anamika": "Ah-nah-mee-kah", "Velachery": "Vellachery"}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


def _content(seconds: float = 0.5) -> FakeResponse:
    wav = pcm_to_wav(silence(seconds, 24000), 24000)
    return FakeResponse(200, {"audioContent": base64.b64encode(wav).decode("ascii")})


def _provider(post, **kwargs) -> CloudTTSProvider:
    fields = dict(api_key="test-key", url=URL, phonemes=PHONEMES, respellings=RESPELLINGS,
                  usd_per_million_chars=30.0, post=post)
    fields.update(kwargs)
    return CloudTTSProvider(**fields)


# -- the request ---------------------------------------------------------------------------------

def test_the_request_has_the_shape_the_rest_reference_gives(tmp_path):
    post = FakePost(_content())
    _provider(post).synthesize("Welcome home.", None, tmp_path / "a.wav")

    url, kwargs = post.calls[0]
    assert url == URL
    assert kwargs["headers"] == {"x-goog-api-key": "test-key", "Content-Type": "application/json"}
    assert kwargs["timeout"] == (10.0, 60.0)
    assert kwargs["json"] == {
        "input": {"text": "Welcome home."},
        "voice": {"languageCode": "en-IN", "name": "en-IN-Chirp3-HD-Sulafat"},
        "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.0},
    }


def test_a_full_voice_name_is_used_as_given_and_a_persona_is_completed(tmp_path):
    post = FakePost(_content())
    provider = _provider(post)
    provider.synthesize("Hi.", "en-IN-Chirp3-HD-Achird", tmp_path / "a.wav")
    provider.synthesize("Hi.", "Puck", tmp_path / "b.wav")
    assert [c[1]["json"]["voice"]["name"] for c in post.calls] == ["en-IN-Chirp3-HD-Achird", "en-IN-Chirp3-HD-Puck"]


def test_the_audio_is_written_and_the_cost_is_the_characters_at_the_list_price(tmp_path):
    audio = _provider(FakePost(_content(0.5))).synthesize("x" * 100, None, tmp_path / "a.wav")
    assert (audio.provider, audio.mode, audio.model_id) == ("cloud_tts", "speech", "en-IN-Chirp3-HD")
    assert audio.voice == "en-IN-Chirp3-HD-Sulafat"
    assert audio.duration_s == 0.5 == wav_duration(tmp_path / "a.wav")
    assert audio.cost_usd == pytest.approx(0.003)


def test_raw_pcm_is_wrapped_as_wav(tmp_path):
    raw = FakeResponse(200, {"audioContent": base64.b64encode(silence(1.0, 24000)).decode("ascii")})
    assert _provider(FakePost(raw)).synthesize("Hi.", None, tmp_path / "a.wav").duration_s == 1.0


def test_a_line_over_the_size_limit_is_refused_before_any_call(tmp_path):
    post = FakePost(_content())
    with pytest.raises(SpeechError, match="5000 bytes"):
        _provider(post).synthesize("é" * 2600, None, tmp_path / "a.wav")
    assert post.calls == []


# -- what each answer means --------------------------------------------------------------------------

@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_an_auth_error_and_is_not_retried(tmp_path, status):
    post = FakePost(FakeResponse(status, text="denied"))
    with pytest.raises(SpeechAuthError, match="GOOGLE_CLOUD_TTS_API_KEY"):
        _provider(post).synthesize("Hi.", None, tmp_path / "a.wav")
    assert len(post.calls) == 1


def test_a_bad_request_naming_the_key_is_an_auth_error(tmp_path):
    with pytest.raises(SpeechAuthError):
        _provider(FakePost(FakeResponse(400, text="API key not valid. Please pass a valid API key."))).synthesize(
            "Hi.", None, tmp_path / "a.wav")


def test_429_is_a_quota_error_and_is_not_retried(tmp_path):
    post = FakePost(FakeResponse(429, text="RESOURCE_EXHAUSTED"))
    with pytest.raises(SpeechQuotaError):
        _provider(post).synthesize("Hi.", None, tmp_path / "a.wav")
    assert len(post.calls) == 1


def test_a_5xx_is_retried_and_then_reported_as_transport(tmp_path):
    post = FakePost(FakeResponse(503, text="busy"))
    with pytest.raises(SpeechTransportError):
        _provider(post, max_retries=3).synthesize("Hi.", None, tmp_path / "a.wav")
    assert len(post.calls) == 3


def test_a_dropped_connection_is_transport(tmp_path):
    with pytest.raises(SpeechTransportError, match="ConnectionError"):
        _provider(FakePost(ConnectionError("reset"))).synthesize("Hi.", None, tmp_path / "a.wav")


def test_an_answer_without_audio_or_json_is_a_speech_error(tmp_path):
    with pytest.raises(SpeechError, match="without any audio"):
        _provider(FakePost(FakeResponse(200, {}))).synthesize("Hi.", None, tmp_path / "a.wav")
    with pytest.raises(SpeechError, match="not JSON"):
        _provider(FakePost(FakeResponse(200))).synthesize("Hi.", None, tmp_path / "a.wav")


def test_without_a_key_the_provider_is_unavailable_and_says_which_variable():
    provider = _provider(FakePost(), api_key="")
    assert not provider.is_available()
    assert provider.unavailable_reason() == "GOOGLE_CLOUD_TTS_API_KEY is not set"


# -- custom pronunciations -------------------------------------------------------------------------

def test_a_name_with_an_ipa_spelling_is_sent_as_a_custom_pronunciation(tmp_path):
    post = FakePost(_content())
    _provider(post).synthesize("Meet Anamika, in Velachery. Anamika awaits.", None, tmp_path / "a.wav")
    source = post.calls[0][1]["json"]["input"]
    assert source["text"] == "Meet Anamika, in Velachery. Anamika awaits."
    assert source["customPronunciations"] == {"pronunciations": [
        {"phrase": "Anamika", "phoneticEncoding": "PHONETIC_ENCODING_IPA", "pronunciation": "ənɑːmɪkɑː"},
        {"phrase": "Velachery", "phoneticEncoding": "PHONETIC_ENCODING_IPA", "pronunciation": "ˈʋeːləˌtʃeːɾi"}]}


def test_only_the_names_in_the_line_are_sent_and_whole_words_only(tmp_path):
    post = FakePost(_content())
    provider = _provider(post)
    provider.synthesize("Anamikas is not the name.", None, tmp_path / "a.wav")
    provider.synthesize("Velachery is.", None, tmp_path / "b.wav")
    assert "customPronunciations" not in post.calls[0][1]["json"]["input"]
    assert [p["phrase"] for p in post.calls[1][1]["json"]["input"]["customPronunciations"]["pronunciations"]] == ["Velachery"]


def test_a_refusal_of_custom_pronunciations_respells_in_plain_text_and_says_so(tmp_path):
    refusal = FakeResponse(400, text="Invalid value at 'input.custom_pronunciations': pronunciation not supported for en-IN")
    post = FakePost(refusal, _content(), _content())
    provider = _provider(post)

    first = provider.synthesize("Meet Anamika.", None, tmp_path / "a.wav")
    assert post.calls[1][1]["json"]["input"] == {"text": "Meet Ah-nah-mee-kah."}
    assert first.notes and "were refused" in first.notes[0]

    assert provider.phoneme_words() == frozenset()                       # the service respells from now on
    provider.synthesize("Meet Ah-nah-mee-kah.", None, tmp_path / "b.wav")
    assert "customPronunciations" not in post.calls[2][1]["json"]["input"]


def test_the_refusal_note_carries_googles_message_and_not_the_raw_body(tmp_path):
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message":
            "The following custom pronunciation phrases are invalid: Anamika."}}
    post = FakePost(FakeResponse(400, body, text=str(body)), _content(), _content())
    audio = _provider(post).synthesize("Meet Anamika.", None, tmp_path / "a.wav")
    assert audio.notes == ["custom pronunciations were refused (The following custom pronunciation phrases "
                           "are invalid: Anamika.); the names were respelled in plain text"]


def test_the_shipped_config_sends_no_phonemes_until_en_in_is_known_to_take_them():
    from src.core.config import load_yaml_config

    assert not load_yaml_config("video/speech.yaml")["cloud_tts"]["phonemes"]


def test_another_bad_request_is_not_taken_for_a_pronunciation_refusal(tmp_path):
    post = FakePost(FakeResponse(400, text="Voice en-IN-Nope does not exist"))
    with pytest.raises(SpeechError, match="refused the request"):
        _provider(post).synthesize("Meet Anamika.", "en-IN-Nope", tmp_path / "a.wav")
    assert len(post.calls) == 1


# -- through the service --------------------------------------------------------------------------------

def _service(tmp_path, *providers) -> SpeechService:
    return SpeechService(list(providers), tmp_path / "cache")


def test_the_service_gives_this_voice_the_name_as_written_and_others_the_respelling(tmp_path):
    cloud_post, gemini_post = FakePost(_content()), FakePost(wav_response())
    cloud = _provider(cloud_post)
    gemini = GeminiSpeechProvider(api_key="k", model="m", default_voice="v", url="https://example.invalid/g",
                                  post=gemini_post)

    one = _service(tmp_path / "one", cloud).synthesize_all(
        ["Meet Anamika."], "", [tmp_path / "one.wav"], respellings=RESPELLINGS)[0]
    assert cloud_post.calls[0][1]["json"]["input"]["text"] == "Meet Anamika."
    assert one.spoken == "Meet Anamika."

    broken = _provider(FakePost(FakeResponse(401, text="no")))
    two = _service(tmp_path / "two", broken, gemini).synthesize_all(
        ["Meet Anamika."], "", [tmp_path / "two.wav"], respellings=RESPELLINGS)[0]
    assert two.provider == "gemini" and two.spoken == "Meet Ah-nah-mee-kah."


def test_a_property_s_own_respelling_beats_the_phonetic_spelling(tmp_path):
    post = FakePost(_content())
    audio = _service(tmp_path, _provider(post)).synthesize_all(
        ["Meet Anamika."], "", [tmp_path / "a.wav"], respellings={**RESPELLINGS, "Anamika": "Anna-mika"},
        local=["anamika"])[0]
    assert post.calls[0][1]["json"]["input"] == {"text": "Meet Anna-mika."}
    assert audio.spoken == "Meet Anna-mika."


def test_a_repeated_line_is_served_from_the_cache_with_no_second_call_and_no_cost(tmp_path):
    post = FakePost(_content())
    service = _service(tmp_path, _provider(post))
    first = service.synthesize("Hello there.", "", tmp_path / "a.wav")
    second = service.synthesize("Hello there.", "", tmp_path / "b.wav")
    assert len(post.calls) == 1
    assert (first.from_cache, second.from_cache, second.cost_usd) == (False, True, 0.0)


def test_a_new_phonetic_spelling_is_a_new_take(tmp_path):
    post = FakePost(_content())
    _service(tmp_path, _provider(post)).synthesize("Meet Anamika.", "", tmp_path / "a.wav")
    _service(tmp_path, _provider(post, phonemes={"Anamika": "ɑːnɑːmɪkɑː"})).synthesize("Meet Anamika.", "", tmp_path / "b.wav")
    assert len(post.calls) == 2


def test_the_one_voice_rule_still_holds_when_cloud_tts_fails_on_one_line(tmp_path):
    cloud = _provider(FakePost(_content(), FakeResponse(429, text="RESOURCE_EXHAUSTED")))
    mock = MockSpeechProvider()
    audios = _service(tmp_path, cloud, mock).synthesize_all(
        ["One two three.", "Four five six."], "", [tmp_path / "a.wav", tmp_path / "b.wav"])
    assert {a.provider for a in audios} == {"mock"}
    assert "cloud_tts could not speak line 2" in audios[0].notes[-1]


# -- the configured chain ------------------------------------------------------------------------------

def test_the_configured_chain_leads_with_cloud_tts_and_names_both_voices(tmp_path, monkeypatch):
    from src.core.config import load_yaml_config

    monkeypatch.setattr("src.core.config.settings.speech_provider", "auto", raising=False)
    monkeypatch.setattr("src.core.config.settings.google_cloud_tts_api_key", "k", raising=False)
    service = get_speech_service()
    assert [p.name for p in service.providers] == ["cloud_tts", "gemini", "offline", "silent"]
    assert service.providers[0].is_available()
    voices = load_yaml_config("video/speech.yaml")["voices"]
    assert voices["default"]["cloud_tts"] == "en-IN-Chirp3-HD-Sulafat"
    assert voices["male"]["cloud_tts"] == "en-IN-Chirp3-HD-Achird"


def test_with_no_key_the_chain_notes_it_and_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.speech_provider", "auto", raising=False)
    audio = _service(tmp_path, _provider(FakePost(), api_key=""), MockSpeechProvider()).synthesize(
        "Hello.", "", tmp_path / "a.wav")
    assert audio.provider == "mock" and audio.notes == ["cloud_tts: GOOGLE_CLOUD_TTS_API_KEY is not set"]
