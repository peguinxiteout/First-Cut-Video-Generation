"""Background music: the chain (Lyria, a licensed library, none), the prompt, the cache and the cut to length."""
from __future__ import annotations

import base64
import wave

import pytest

from src.agents.video import voice
from src.data.video_schemas import ScriptLine, Shot, Storyboard, VideoBrief, VideoScript
from src.models.music import MusicService, build_prompt, get_music_service
from src.models.music.base import MusicAuthError, MusicError, MusicQuotaError, MusicRequest, MusicTransportError
from src.models.music.library import LibraryMusicProvider, mood_slug
from src.models.music.lyria import LyriaProvider
from src.models.speech import SpeechService
from src.models.speech.mock import MockSpeechProvider
from tests.video.fakes import FakeMusic, FakePost, FakeResponse, tone_wav

URL = "https://example.invalid/v1beta/interactions"
MODELS = [{"id": "clip-model", "max_video_s": 30, "cost_usd": 0.04}, {"id": "long-model", "max_video_s": 120, "cost_usd": 0.08}]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


def _request(duration_s=15.0, prompt="Instrumental only. Calm.") -> MusicRequest:
    return MusicRequest(prompt=prompt, mood="calm", duration_s=duration_s)


def _audio_response(path) -> FakeResponse:
    block = {"type": "audio", "data": base64.b64encode(path.read_bytes()).decode("ascii")}
    return FakeResponse(200, {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "lyrics"}, block]}]})


def _lyria(post, key="k") -> LyriaProvider:
    return LyriaProvider(api_key=key, url=URL, models=MODELS, api_revision="2026-05-20", post=post)


def _level(path, start_s, end_s) -> float:
    """Peak amplitude (0-1) of a mono or stereo 16-bit WAV between two times."""
    with wave.open(str(path), "rb") as handle:
        rate, channels = handle.getframerate(), handle.getnchannels()
        handle.setpos(round(start_s * rate))
        data = handle.readframes(round((end_s - start_s) * rate))
    samples = [int.from_bytes(data[i:i + 2], "little", signed=True) for i in range(0, len(data) - 1, 2 * channels)]
    return max(abs(s) for s in samples) / 32768


# -- the prompt ---------------------------------------------------------------------------------------

def test_the_prompt_is_built_from_the_brief_and_is_always_instrumental():
    template = "Instrumental only, no vocals and no lyrics. Music: $mood. A $tone feel, $objective. $direction End softly."
    text = build_prompt(template, {"site_visit": "to invite people to visit"}, "calm cinematic", "refined", "site_visit",
                        "Golden hour, slow pans.")
    assert text == ("Instrumental only, no vocals and no lyrics. Music: calm cinematic. A refined feel, to invite people "
                    "to visit. Golden hour, slow pans. End softly.")


def test_the_configured_prompt_names_every_part_of_the_brief_and_says_instrumental():
    service = get_music_service()
    text = build_prompt(service.template, service.objectives, "warm acoustic", "warm", "launch", "Sunrise over the towers.")
    assert text.lower().startswith("instrumental only, no vocals")
    for part in ("warm acoustic", "warm", "Sunrise over the towers.", service.objectives["launch"]):
        assert part in text


def test_a_missing_creative_direction_leaves_no_stray_gap():
    text = build_prompt("Mood: $mood. $direction Done.", {}, "calm", "warm", "awareness", "")
    assert text == "Mood: calm. Done."


# -- Lyria ----------------------------------------------------------------------------------------------

def test_lyria_asks_the_interactions_api_with_the_documented_request(tmp_path):
    post = FakePost(_audio_response(tone_wav(tmp_path / "t.wav", 2.0)))
    track = _lyria(post).generate(_request(), tmp_path / "out.wav")

    url, kwargs = post.calls[0]
    assert url == URL
    assert kwargs["headers"] == {"x-goog-api-key": "k", "Content-Type": "application/json", "Api-Revision": "2026-05-20"}
    assert kwargs["json"] == {"model": "clip-model", "input": "Instrumental only. Calm.",
                              "response_format": {"type": "audio"}}
    assert (track.provider, track.model_id, track.cost_usd) == ("lyria", "clip-model", 0.04)
    assert track.duration_s == pytest.approx(2.0, abs=0.1) and track.path.is_file()


def test_lyria_picks_the_model_by_the_length_of_the_video(tmp_path):
    post = FakePost(_audio_response(tone_wav(tmp_path / "t.wav", 1.0)))
    provider = _lyria(post)
    assert provider.generate(_request(15), tmp_path / "a.wav").model_id == "clip-model"
    assert provider.generate(_request(30), tmp_path / "b.wav").model_id == "clip-model"
    long = provider.generate(_request(45), tmp_path / "c.wav")
    assert (long.model_id, long.cost_usd) == ("long-model", 0.08)
    assert provider.cache_key_parts(_request(15))["model"] == "clip-model"
    assert provider.cache_key_parts(_request(45))["model"] == "long-model"


@pytest.mark.parametrize("response, error", [
    (FakeResponse(401, text="bad key"), MusicAuthError),
    (FakeResponse(400, text="API key not valid. Please pass a valid API key."), MusicAuthError),
    (FakeResponse(429, text="RESOURCE_EXHAUSTED"), MusicQuotaError),
    (FakeResponse(403, text="Lyria is not available on the free tier; enable billing"), MusicQuotaError),
    (FakeResponse(400, text="model requires a paid plan"), MusicQuotaError),
    (FakeResponse(400, text="prompt blocked"), MusicError),
    (FakeResponse(200, {"steps": []}), MusicError),
    (FakeResponse(200, None), MusicError),
])
def test_lyria_errors_are_named_for_what_to_do_next(tmp_path, response, error):
    with pytest.raises(error) as caught:
        _lyria(FakePost(response)).generate(_request(), tmp_path / "x.wav")
    assert type(caught.value) is error


def test_lyria_retries_a_busy_service_but_not_a_refusal(tmp_path):
    busy = FakePost(FakeResponse(503), FakeResponse(503), _audio_response(tone_wav(tmp_path / "t.wav", 1.0)))
    assert _lyria(busy).generate(_request(), tmp_path / "x.wav").path.is_file()
    assert len(busy.calls) == 3
    refused = FakePost(FakeResponse(429))
    with pytest.raises(MusicQuotaError):
        _lyria(refused).generate(_request(), tmp_path / "y.wav")
    assert len(refused.calls) == 1
    always_busy = FakePost(FakeResponse(503))
    with pytest.raises(MusicTransportError):
        _lyria(always_busy).generate(_request(), tmp_path / "z.wav")


def test_lyria_without_a_key_says_so_and_makes_no_call(tmp_path):
    post = FakePost(FakeResponse(200))
    provider = _lyria(post, key="")
    assert not provider.is_available() and provider.unavailable_reason() == "GEMINI_API_KEY is not set"
    with pytest.raises(MusicAuthError):
        provider.generate(_request(), tmp_path / "x.wav")
    assert post.calls == []


# -- the library ----------------------------------------------------------------------------------------

def _library(tmp_path, tracks):
    """tracks: (mood folder, track folder, seconds, licence text or None)."""
    root = tmp_path / "music"
    for mood, name, seconds, licence in tracks:
        folder = root / mood / name
        tone_wav(folder / "track.wav", seconds)
        if licence is not None:
            (folder / "licence.txt").write_text(licence, encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    return LibraryMusicProvider(root)


def test_a_mood_matches_its_folder_however_it_is_spelled():
    assert {mood_slug(m) for m in ("warm acoustic", "Warm-Acoustic", "warm_acoustic", " WARM  acoustic ")} == {"warm-acoustic"}


def test_the_library_gives_a_licensed_track_for_the_mood_and_copies_its_licence(tmp_path):
    provider = _library(tmp_path, [("warm-acoustic", "morning", 20, "CC0 - Sam Smith, https://example.invalid/morning\nmore")])
    track = provider.generate(MusicRequest("p", "warm acoustic", 15), tmp_path / "out.wav")
    assert (track.provider, track.title, track.cost_usd) == ("library", "morning", 0)
    assert track.licence == "CC0 - Sam Smith, https://example.invalid/morning"
    assert track.duration_s == pytest.approx(20, abs=0.1) and track.notes == []


def test_a_track_without_a_licence_is_not_used_and_the_reason_is_given(tmp_path):
    provider = _library(tmp_path, [("calm", "nolicence", 20, None), ("calm", "blank", 20, "  \n")])
    with pytest.raises(MusicError) as caught:
        provider.generate(MusicRequest("p", "calm", 15), tmp_path / "out.wav")
    assert "no track for the mood 'calm'" in str(caught.value) and "nolicence has no licence.txt" in str(caught.value)


def test_the_first_track_long_enough_is_used_else_the_longest_with_a_note(tmp_path):
    provider = _library(tmp_path, [("calm", "a_short", 5, "x"), ("calm", "b_long", 20, "y"), ("calm", "c_longer", 30, "z")])
    assert provider.generate(MusicRequest("p", "calm", 15), tmp_path / "one.wav").title == "b_long"
    short = provider.generate(MusicRequest("p", "calm", 45), tmp_path / "two.wav")
    assert short.title == "c_longer" and "shorter than the video" in short.notes[0]


def test_another_moods_tracks_are_not_used(tmp_path):
    provider = _library(tmp_path, [("upbeat", "t", 20, "x")])
    with pytest.raises(MusicError):
        provider.generate(MusicRequest("p", "calm", 15), tmp_path / "out.wav")


def test_a_missing_library_folder_is_unavailable(tmp_path):
    provider = LibraryMusicProvider(tmp_path / "nope")
    assert not provider.is_available() and "no music library" in provider.unavailable_reason()


# -- the service ------------------------------------------------------------------------------------------

def _service(tmp_path, *providers, **kwargs) -> MusicService:
    return MusicService(list(providers), tmp_path / "cache", template="Instrumental only. $mood. $direction", **kwargs)


def _make(service, tmp_path, total_s=10.0, name="music.wav", direction="Slow."):
    return service.make("calm cinematic", "refined", "awareness", direction, total_s, tmp_path / "audio" / name)


def test_the_first_provider_that_makes_a_track_wins_and_the_record_says_how(tmp_path):
    fake = FakeMusic("lyria")
    result = _make(_service(tmp_path, fake, FakeMusic("library")), tmp_path)
    info = result.info
    assert (info.provider, info.model_id, info.cost_usd, info.from_cache) == ("lyria", "fake-model", 0.04, False)
    assert info.prompt == "Instrumental only. calm cinematic. Slow." and info.notes == []
    assert result.path == tmp_path / "audio" / "music.wav" and result.path.is_file()
    assert fake.requests[0].mood == "calm cinematic" and fake.requests[0].objective == "awareness"


def test_a_lyria_refusal_falls_to_the_library_and_is_recorded(tmp_path):
    lyria = FakeMusic("lyria", error=MusicQuotaError("Lyria is not available on this key's plan (HTTP 429)"))
    result = _make(_service(tmp_path, lyria, FakeMusic("library", cost_usd=0)), tmp_path)
    assert result.info.provider == "library" and result.info.cost_usd == 0
    assert result.info.notes == ["lyria: Lyria is not available on this key's plan (HTTP 429)"]


def test_with_no_provider_able_there_is_no_music_and_each_reason_is_kept(tmp_path):
    lyria = FakeMusic("lyria", available=False)
    library = FakeMusic("library", error=MusicError("there is no track for the mood"))
    result = _make(_service(tmp_path, lyria, library), tmp_path)
    assert result.path is None and result.info.provider == "none"
    assert result.info.notes == ["lyria: lyria has no key", "library: there is no track for the mood"]
    assert result.info.prompt.startswith("Instrumental only")


def test_a_provider_that_is_refused_is_not_asked_again_in_the_same_run(tmp_path):
    lyria = FakeMusic("lyria", error=MusicAuthError("rejected"))
    service = _service(tmp_path, lyria, FakeMusic("library"))
    _make(service, tmp_path, name="a.wav")
    second = _make(service, tmp_path, name="b.wav", direction="Different.")
    assert len(lyria.requests) == 1
    assert "lyria: skipped for this run (rejected)" in second.info.notes


def test_the_same_prompt_is_served_from_the_cache_and_costs_nothing(tmp_path):
    fake = FakeMusic("lyria")
    service = _service(tmp_path, fake)
    first = _make(service, tmp_path, name="a.wav")
    again = _make(_service(tmp_path, fake), tmp_path, name="b.wav")
    assert len(fake.requests) == 1
    assert (first.info.from_cache, first.info.cost_usd) == (False, 0.04)
    assert (again.info.from_cache, again.info.cost_usd, again.info.provider) == (True, 0.0, "lyria")
    assert again.info.prompt == first.info.prompt and again.info.model_id == "fake-model"


def test_a_different_prompt_is_not_a_cache_hit(tmp_path):
    fake = FakeMusic("lyria")
    service = _service(tmp_path, fake)
    _make(service, tmp_path, name="a.wav", direction="Slow.")
    _make(service, tmp_path, name="b.wav", direction="Fast.")
    assert len(fake.requests) == 2


def test_a_library_track_is_not_cached(tmp_path):
    library = FakeMusic("library", cost_usd=0, cacheable=False)
    service = _service(tmp_path, library)
    _make(service, tmp_path, name="a.wav")
    _make(service, tmp_path, name="b.wav")
    assert len(library.requests) == 2 and not (tmp_path / "cache").exists()


def test_the_track_is_cut_to_the_video_and_fades_out(tmp_path):
    result = _make(_service(tmp_path, FakeMusic("lyria", seconds=12.0), fade_out_s=1.5, fade_in_s=0.4), tmp_path, total_s=9.0)
    with wave.open(str(result.path), "rb") as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(9.0, abs=0.05)
    assert _level(result.path, 4.0, 5.0) > 0.3                       # the body is at full level
    assert _level(result.path, 8.8, 8.99) < 0.12                      # the last moments are nearly silent
    assert _level(result.path, 0.0, 0.05) < 0.12                      # and it comes in softly
    assert result.info.duration_s == pytest.approx(12.0, abs=0.1)     # the record keeps the track as it was made


def test_a_track_shorter_than_the_video_fades_where_it_ends_and_says_so(tmp_path):
    result = _make(_service(tmp_path, FakeMusic("lyria", seconds=5.0)), tmp_path, total_s=9.0)
    with wave.open(str(result.path), "rb") as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(5.0, abs=0.05)
    assert any("shorter than the video" in n for n in result.info.notes)


def test_the_configured_chain_is_beatoven_then_lyria_then_the_library_then_none(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    monkeypatch.setattr("src.core.config.settings.beatoven_api_key", "b", raising=False)
    service = get_music_service()
    assert [p.name for p in service.providers] == ["beatoven", "lyria", "library"]
    assert service.providers[0].is_available() and service.providers[1].is_available()
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "", raising=False)
    monkeypatch.setattr("src.core.config.settings.beatoven_api_key", "", raising=False)
    result = get_music_service().make("calm", "warm", "awareness", "", 10.0, tmp_path / "m.wav")
    assert result.path is None and result.info.provider == "none"
    assert result.info.notes[0] == "beatoven: BEATOVEN_API_KEY is not set"
    assert result.info.notes[1] == "lyria: GEMINI_API_KEY is not set" and result.info.notes[2].startswith("library:")


# -- in the voice step -----------------------------------------------------------------------------------

def test_the_voice_step_asks_for_music_from_the_brief_and_passes_the_track_on(tmp_path):
    fake = FakeMusic("lyria", seconds=20.0)
    script = VideoScript(lines=[ScriptLine(text="One two three.", role="hook"), ScriptLine(text="Book now.", role="cta")])
    board = Storyboard(shots=[Shot(index=0, seconds=4.0, line_ids=[0]),
                              Shot(index=1, kind="card", camera="still", seconds=2.0, line_ids=[1])])
    brief = VideoBrief(property_id="P", name="Aurelia", duration_s=10, objective="site_visit", tone="refined",
                       music_mood="calm cinematic", creative_direction="Golden hour.", quality="draft")
    speech = SpeechService([MockSpeechProvider()], tmp_path / "speech")
    result = voice.run(script, board, brief, tmp_path / "out", speech, _service(tmp_path, fake))

    request = fake.requests[0]
    assert (request.mood, request.tone, request.objective, request.creative_direction) == (
        "calm cinematic", "refined", "site_visit", "Golden hour.")
    assert request.duration_s == pytest.approx(result.total_s, abs=0.01)
    assert result.music_track == str(tmp_path / "out" / "audio" / "music.wav")
    assert result.music_info.provider == "lyria" and result.music_info.cost_usd == 0.04
