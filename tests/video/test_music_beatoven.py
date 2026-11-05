"""Beatoven music through fake transports: the compose call, the polling, the download, what each answer means,
the record it leaves and the chain it leads."""
from __future__ import annotations

import pytest

from src.models.music import MusicService
from src.models.music.base import MusicAuthError, MusicError, MusicQuotaError, MusicRequest, MusicTransportError
from src.models.music.beatoven import BeatovenProvider
from tests.video.fakes import FakeMusic, FakePost, FakeResponse, tone_wav

BASE = "https://example.invalid"
TRACK_URL = "https://cdn.example.invalid/track.wav"
PROMPT = "Instrumental only, no vocals. Calm and warm."


class Bytes(FakeResponse):
    def __init__(self, content: bytes, status: int = 200):
        super().__init__(status)
        self.content = content


class FakeGet:
    """Stands in for requests.get: answers by url, in order (the last repeats)."""

    def __init__(self, tasks, track: bytes = b""):
        self.tasks, self.track = list(tasks), track
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url == TRACK_URL:
            return Bytes(self.track)
        outcome = self.tasks.pop(0) if len(self.tasks) > 1 else self.tasks[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _task(status: str, **meta) -> FakeResponse:
    body = {"status": status}
    if meta:
        body["meta"] = meta
    return FakeResponse(200, body)


def _done(**extra) -> FakeResponse:
    return _task("composed", project_id="proj-1", track_id="trk-9", version=1, track_url=TRACK_URL, **extra)


STARTED = FakeResponse(200, {"status": "started", "task_id": "task-1"})


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


@pytest.fixture
def wav_bytes(tmp_path):
    return tone_wav(tmp_path / "src.wav", 17.0).read_bytes()


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _provider(post, get, clock=None, **kwargs) -> BeatovenProvider:
    clock = clock or Clock()
    fields = dict(api_key="test-key", base_url=BASE, licence_note="Plan not confirmed.", post=post, get=get,
                  sleep=clock.sleep, clock=clock)
    fields.update(kwargs)
    return BeatovenProvider(**fields)


def _request(duration_s=15.0) -> MusicRequest:
    return MusicRequest(prompt=PROMPT, mood="calm", duration_s=duration_s)


# -- the calls ---------------------------------------------------------------------------------------

def test_the_compose_request_and_the_poll_follow_the_spec(tmp_path, wav_bytes):
    post, get = FakePost(STARTED), FakeGet([_task("composing"), _task("running"), _done()], wav_bytes)
    clock = Clock()
    _provider(post, get, clock).generate(_request(15.0), tmp_path / "t.wav")

    url, kwargs = post.calls[0]
    assert url == f"{BASE}/api/v1/tracks/compose"
    assert kwargs["headers"] == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    assert kwargs["json"] == {"prompt": {"text": f"A 16-second instrumental track. {PROMPT}"}, "format": "wav"}
    assert [c[0] for c in get.calls] == [f"{BASE}/api/v1/tasks/task-1"] * 3 + [TRACK_URL]
    assert "headers" not in get.calls[-1][1]                          # the key is not sent to the download host
    assert clock.slept == [5.0, 5.0]


def test_the_length_asked_for_is_the_videos_to_the_nearest_second_with_a_little_over():
    provider = _provider(FakePost(), FakeGet([_done()]))
    assert [provider.length_s(s) for s in (10.0, 14.2, 15.0, 15.04, 59.5)] == [11, 15, 16, 16, 61]
    assert provider.prompt_text(_request(12.0)).startswith("A 13-second instrumental track.")


def test_the_track_is_a_wav_with_its_id_cost_prompt_and_licence_note(tmp_path, wav_bytes):
    track = _provider(FakePost(STARTED), FakeGet([_done()], wav_bytes), cost_usd=0.0).generate(
        _request(), tmp_path / "t.wav")
    assert (track.provider, track.model_id, track.track_id) == ("beatoven", "beatoven-compose", "trk-9")
    assert track.prompt == f"A 16-second instrumental track. {PROMPT}"
    assert track.duration_s == pytest.approx(17.0, abs=0.05)
    assert track.licence == "Plan not confirmed."
    assert any("project proj-1, version 1, task task-1" in n for n in track.notes)
    assert any("returns no licence terms and no price" in n for n in track.notes)


def test_the_compose_call_is_made_once_even_when_it_drops(tmp_path):
    post = FakePost(ConnectionError("reset"))
    with pytest.raises(MusicTransportError):
        _provider(post, FakeGet([_done()])).generate(_request(), tmp_path / "t.wav")
    assert len(post.calls) == 1


def test_a_busy_poll_is_retried_but_a_failed_task_is_not(tmp_path, wav_bytes):
    get = FakeGet([FakeResponse(503, text="busy"), _done()], wav_bytes)
    assert _provider(FakePost(STARTED), get).generate(_request(), tmp_path / "t.wav").track_id == "trk-9"

    with pytest.raises(MusicError, match="status 'failed'"):
        _provider(FakePost(STARTED), FakeGet([_task("failed")])).generate(_request(), tmp_path / "u.wav")


def test_a_task_that_never_finishes_gives_up_at_the_deadline(tmp_path):
    clock = Clock()
    with pytest.raises(MusicTransportError, match="not done after 20 s"):
        _provider(FakePost(STARTED), FakeGet([_task("running")]), clock, poll_s=5.0, poll_timeout_s=20.0).generate(
            _request(), tmp_path / "t.wav")
    assert clock.now > 20.0


def test_a_finished_task_without_a_track_url_is_an_error(tmp_path):
    with pytest.raises(MusicError, match="without a track"):
        _provider(FakePost(STARTED), FakeGet([_task("composed", track_id="x")])).generate(_request(), tmp_path / "t.wav")


def test_a_start_without_a_task_id_is_an_error(tmp_path):
    with pytest.raises(MusicError, match="did not name the task"):
        _provider(FakePost(FakeResponse(200, {"status": "started"})), FakeGet([_done()])).generate(
            _request(), tmp_path / "t.wav")


def test_audio_that_is_not_audio_is_a_music_error(tmp_path):
    with pytest.raises(MusicError, match="could not be read"):
        _provider(FakePost(STARTED), FakeGet([_done()], b"not audio at all")).generate(_request(), tmp_path / "t.wav")


# -- what each answer means -----------------------------------------------------------------------------

@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_an_auth_error(tmp_path, status):
    with pytest.raises(MusicAuthError, match="BEATOVEN_API_KEY"):
        _provider(FakePost(FakeResponse(status, text="no")), FakeGet([_done()])).generate(_request(), tmp_path / "t.wav")


@pytest.mark.parametrize("response", [FakeResponse(402, text="pay"), FakeResponse(429, text="slow"),
                                      FakeResponse(400, text="You have no credits left on your plan")])
def test_a_spent_allowance_is_a_quota_error(tmp_path, response):
    with pytest.raises(MusicQuotaError):
        _provider(FakePost(response), FakeGet([_done()])).generate(_request(), tmp_path / "t.wav")


def test_another_refusal_is_a_plain_music_error(tmp_path):
    with pytest.raises(MusicError, match="refused the request"):
        _provider(FakePost(FakeResponse(400, text="bad prompt")), FakeGet([_done()])).generate(_request(), tmp_path / "t.wav")


def test_without_a_key_the_provider_is_unavailable_and_says_which_variable(tmp_path):
    provider = _provider(FakePost(), FakeGet([_done()]), api_key="")
    assert not provider.is_available() and provider.unavailable_reason() == "BEATOVEN_API_KEY is not set"
    with pytest.raises(MusicAuthError):
        provider.generate(_request(), tmp_path / "t.wav")


# -- through the service -----------------------------------------------------------------------------------

def _service(tmp_path, *providers) -> MusicService:
    return MusicService(list(providers), tmp_path / "cache")


def test_the_track_is_cut_to_the_videos_length_and_the_record_carries_the_id_prompt_cost_and_licence(tmp_path, wav_bytes):
    service = _service(tmp_path, _provider(FakePost(STARTED), FakeGet([_done()], wav_bytes)))
    result = service.make("calm", "warm", "awareness", "", 15.0, tmp_path / "m.wav")

    info = result.info
    assert (info.provider, info.track_id, info.cost_usd, info.from_cache) == ("beatoven", "trk-9", 0.0, False)
    assert info.licence == "Plan not confirmed." and info.prompt.startswith("A 16-second instrumental track.")
    assert "instrumental" in info.prompt.lower() and "calm" in info.prompt
    import wave
    with wave.open(str(result.path), "rb") as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(15.0, abs=0.05)


def test_a_repeat_is_served_from_the_cache_with_the_same_record_and_no_second_generation(tmp_path, wav_bytes):
    post = FakePost(STARTED)
    service = _service(tmp_path, _provider(post, FakeGet([_done()], wav_bytes)))
    service.make("calm", "warm", "awareness", "", 15.0, tmp_path / "a.wav")
    again = service.make("calm", "warm", "awareness", "", 15.0, tmp_path / "b.wav")
    assert len(post.calls) == 1
    assert (again.info.from_cache, again.info.cost_usd, again.info.track_id) == (True, 0.0, "trk-9")
    assert again.info.licence == "Plan not confirmed." and any("project proj-1" in n for n in again.info.notes)


def test_a_different_length_is_a_different_track(tmp_path, wav_bytes):
    post = FakePost(STARTED)
    service = _service(tmp_path, _provider(post, FakeGet([_done()], wav_bytes)))
    service.make("calm", "warm", "awareness", "", 15.0, tmp_path / "a.wav")
    service.make("calm", "warm", "awareness", "", 30.0, tmp_path / "b.wav")
    assert len(post.calls) == 2


def test_when_beatoven_has_no_plan_for_it_the_next_provider_makes_the_track(tmp_path):
    quota = _provider(FakePost(FakeResponse(402, text="pay")), FakeGet([_done()]))
    lyria = FakeMusic("lyria", seconds=20.0)
    service = _service(tmp_path, quota, lyria)
    first = service.make("calm", "warm", "awareness", "", 10.0, tmp_path / "a.wav")
    second = service.make("calm", "warm", "awareness", "", 10.0, tmp_path / "b.wav")

    assert first.info.provider == second.info.provider == "lyria"
    assert first.info.notes[0].startswith("beatoven: Beatoven's allowance is spent")
    assert second.info.notes[0].startswith("beatoven: skipped for this run")          # not asked again
    assert service.dropped.keys() == {"beatoven"}
