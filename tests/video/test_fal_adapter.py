"""The fal adapter against a fake fal on 127.0.0.1: the queue's submit, status and
result, the download, the errors, and what is sent for each model. Nothing
reaches the network (tests/conftest.py allows loopback only)."""
from __future__ import annotations

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from src.models.video_gen import registry, selector
from src.models.video_gen.base import (
    VideoContentRefusedError, VideoGenerationAuthError, VideoGenerationError, VideoGenerationQuotaError,
    VideoGenerationTransportError,
)
from src.models.video_gen.fal import FalClient, FalOutpainter, FalVideoProvider
from src.rendering import ffmpeg
from tests.video.fakes import make_photo, small_request

KEY = "test-key-123"


@pytest.fixture(scope="module")
def clip_bytes(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("fal") / "answer.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", "color=c=0x336699:s=72x128:r=10:d=1.2", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(path)])
    return path.read_bytes()


class FakeFal:
    """A fal queue on loopback. `script` sets what each step answers."""

    def __init__(self, clip: bytes):
        self.clip = clip
        self.requests: list[dict] = []
        self.submit = (200, None)                     # (status, body) or body None for the normal answer
        self.statuses: list[tuple[int, dict]] = [(200, {"status": "IN_QUEUE"}), (200, {"status": "IN_PROGRESS"}),
                                                 (200, {"status": "COMPLETED"})]
        self.result = (200, None)
        self.file = (200, None)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status: int, body, ctype: str = "application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _record(self, body=None):
                fake.requests.append({"method": self.command, "path": self.path,
                                      "auth": self.headers.get("Authorization"), "body": body})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                self._record(body)
                status, answer = fake.submit
                if answer is None:
                    base = f"{fake.url}/owner/app/requests/req-1"
                    answer = {"request_id": "req-1", "status_url": f"{base}/status", "response_url": base}
                self._send(status, answer)

            def do_GET(self):
                self._record()
                if self.path.endswith("/status"):
                    status, answer = fake.statuses.pop(0) if len(fake.statuses) > 1 else fake.statuses[0]
                    self._send(status, answer)
                elif self.path.startswith("/files/"):
                    status, answer = fake.file
                    if self.path.endswith(".png"):
                        self._send(status, answer if answer is not None else fake.png, "image/png")
                    else:
                        self._send(status, answer if answer is not None else fake.clip, "video/mp4")
                else:
                    status, answer = fake.result
                    self._send(status, answer if answer is not None else
                               {"video": {"url": f"{fake.url}/files/clip.mp4"}, "expanded_prompt": None})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.png = b""
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fal(clip_bytes):
    server = FakeFal(clip_bytes)
    yield server
    server.close()


def _client(fal: FakeFal, **kwargs) -> FalClient:
    return FalClient(KEY, queue_url=fal.url, poll_s=0.0, sleep=lambda s: None, **kwargs)


def _provider(fal: FakeFal, model: str = "minimax_h3_max") -> FalVideoProvider:
    spec = selector.load_catalog()[model]
    return FalVideoProvider(spec, KEY, _client(fal))


def _request(tmp_path, **overrides):
    image = make_photo(tmp_path / "kf.jpg", (72, 128), seed=3)
    return small_request(image, width=72, height=128, prompt="push in", **overrides)


# ---------------------------------------------------------------------------

def test_it_is_registered_and_each_fal_entry_has_its_endpoint():
    assert registry.ADAPTERS["fal"] is FalVideoProvider
    catalog = selector.load_catalog()
    assert catalog["minimax_h3_max"].endpoint == "minimax/h3-max/image-to-video"
    assert catalog["kling_25_turbo_pro"].endpoint == "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"
    assert FalVideoProvider(catalog["seedance_2"], KEY).is_available() is False      # endpoint not set
    assert FalVideoProvider(catalog["minimax_h3_max"], "").is_available() is False
    assert FalVideoProvider(catalog["minimax_h3_max"], KEY).is_available() is True


def test_the_picture_goes_as_a_data_uri_of_the_file_itself(fal, tmp_path):
    request = _request(tmp_path)
    body = _provider(fal).payload(request)
    head, data = body["image_url"].split(",", 1)
    assert head == "data:image/jpeg;base64"
    assert base64.b64decode(data) == Path(request.image_path).read_bytes()


def test_submit_poll_and_download_make_a_clip_with_its_billed_seconds(fal, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    clip = _provider(fal).generate(_request(tmp_path), tmp_path / "out.mp4")
    assert clip.path.read_bytes() == fal.clip
    assert clip.meta["request_id"] == "req-1" and clip.meta["billed_s"] == pytest.approx(1.2, abs=0.05)
    assert clip.cost_usd == pytest.approx(0.08 * clip.meta["billed_s"], abs=1e-4)    # 768P rate
    assert (clip.width, clip.height) == (72, 128)
    methods = [(r["method"], r["path"]) for r in fal.requests]
    assert methods[0] == ("POST", "/minimax/h3-max/image-to-video")
    assert [m for m, p in methods if p.endswith("/status")] == ["GET"] * 3
    assert all(r["auth"] == f"Key {KEY}" for r in fal.requests if not r["path"].startswith("/files/"))
    assert [r["auth"] for r in fal.requests if r["path"].startswith("/files/")] == [None]   # no key to the CDN
    assert KEY not in caplog.text


def test_h3_gets_its_input_settings_and_a_number_for_duration(fal, tmp_path):
    _provider(fal).generate(_request(tmp_path, duration_s=3.0), tmp_path / "out.mp4")
    body = fal.posts()[0]["body"]
    assert body["resolution"] == "768P" and body["prompt_expansion_mode"] == "disabled"
    assert body["duration"] == 3.0 and body["prompt"] == "push in"
    assert "end_image_url" not in body and "negative_prompt" not in body


def test_first_and_last_frame_use_each_models_own_field(fal, tmp_path):
    end = make_photo(tmp_path / "end.jpg", (72, 128), seed=9)
    _provider(fal).generate(_request(tmp_path, end_image_path=end, negative_prompt="new people"), tmp_path / "a.mp4")
    h3 = fal.posts()[-1]["body"]
    assert h3["end_image_url"].startswith("data:image/jpeg;base64,") and "tail_image_url" not in h3
    assert "negative_prompt" not in h3                      # H3 takes none: the prompt carries the rule
    _provider(fal, "kling_25_turbo_pro").generate(_request(tmp_path, end_image_path=end, negative_prompt="new people"),
                                                  tmp_path / "b.mp4")
    kling = fal.posts()[-1]["body"]
    assert kling["tail_image_url"] == h3["end_image_url"] and "end_image_url" not in kling
    assert kling["negative_prompt"] == "new people" and kling["duration"] == "5" and kling["cfg_scale"] == 0.5
    assert fal.posts()[-1]["path"] == "/fal-ai/kling-video/v2.5-turbo/pro/image-to-video"


def test_kling_is_billed_per_five_second_clip():
    kling = FalVideoProvider(selector.load_catalog()["kling_25_turbo_pro"], KEY)
    assert kling.duration_value(3.0) == "5" and kling.duration_value(7) == "10"
    assert kling.estimate_call(3.0) == 0.35 and kling.estimate_call(7) == 0.70
    h3 = FalVideoProvider(selector.load_catalog()["minimax_h3_max"], KEY)
    assert h3.estimate_call(3.0) == pytest.approx(0.08 * 3.7)
    assert h3.duration_value(0.5) == 2.0                    # the shortest H3 is asked for


@pytest.mark.parametrize("status,body,error", [
    (500, {"detail": "boom"}, VideoGenerationTransportError),
    (401, {"detail": "bad key"}, VideoGenerationAuthError),
    (403, {"detail": "User is locked. Reason: Exhausted balance."}, VideoGenerationQuotaError),
    (402, {"detail": "pay"}, VideoGenerationQuotaError),
    (429, {"detail": "slow down"}, VideoGenerationQuotaError),
    (422, {"detail": "image_url is not valid"}, VideoGenerationError),
])
def test_a_failed_submit_is_never_sent_again(fal, tmp_path, status, body, error):
    fal.submit = (status, body)
    with pytest.raises(error):
        _provider(fal).generate(_request(tmp_path), tmp_path / "out.mp4")
    assert len(fal.posts()) == 1
    assert not [r for r in fal.requests if r["method"] == "GET"]


def test_a_dropped_connection_on_submit_is_not_retried(tmp_path, network_attempts):
    client = FalClient(KEY, queue_url="http://127.0.0.1:9", poll_s=0.0, sleep=lambda s: None)   # nothing listens
    provider = FalVideoProvider(selector.load_catalog()["minimax_h3_max"], KEY, client)
    calls = []
    real = client._call
    client._call = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    with pytest.raises(VideoGenerationTransportError):
        provider.generate(_request(tmp_path), tmp_path / "out.mp4")
    assert len(calls) == 1


def test_a_busy_status_is_polled_again_but_not_resubmitted(fal, tmp_path):
    fal.statuses = [(503, {"detail": "busy"}), (200, {"status": "IN_PROGRESS"}), (200, {"status": "COMPLETED"})]
    _provider(fal).generate(_request(tmp_path), tmp_path / "out.mp4")
    assert len(fal.posts()) == 1


@pytest.mark.parametrize("status,error", [
    ({"status": "COMPLETED", "error": "Content policy violation", "error_type": "content_policy"},
     VideoContentRefusedError),
    ({"status": "COMPLETED", "error": "Internal failure"}, VideoGenerationError),
    ({"status": "CANCELLED"}, VideoGenerationError),
])
def test_a_failed_job_says_why_and_is_not_resubmitted(fal, tmp_path, status, error):
    fal.statuses = [(200, status)]
    with pytest.raises(error):
        _provider(fal).generate(_request(tmp_path), tmp_path / "out.mp4")
    assert len(fal.posts()) == 1


def test_a_refusal_in_the_result_is_a_content_refusal(fal, tmp_path):
    fal.result = (422, {"detail": [{"msg": "The image was flagged by the safety checker"}]})
    with pytest.raises(VideoContentRefusedError):
        _provider(fal).generate(_request(tmp_path), tmp_path / "out.mp4")


def test_a_result_without_a_video_is_an_error(fal, tmp_path):
    fal.result = (200, {"images": []})
    with pytest.raises(VideoGenerationError, match="without a video"):
        _provider(fal).generate(_request(tmp_path), tmp_path / "out.mp4")


def test_a_job_that_never_finishes_times_out(fal, tmp_path):
    fal.statuses = [(200, {"status": "IN_PROGRESS"})]
    ticks = iter(range(0, 10_000, 100))
    client = FalClient(KEY, queue_url=fal.url, poll_s=0.0, poll_timeout_s=250, sleep=lambda s: None,
                       clock=lambda: next(ticks))
    provider = FalVideoProvider(selector.load_catalog()["minimax_h3_max"], KEY, client)
    with pytest.raises(VideoGenerationTransportError, match="not done"):
        provider.generate(_request(tmp_path), tmp_path / "out.mp4")
    assert len(fal.posts()) == 1


def test_no_key_means_no_request(fal, tmp_path):
    provider = FalVideoProvider(selector.load_catalog()["minimax_h3_max"], "", _client(fal))
    with pytest.raises(VideoGenerationAuthError):
        provider.generate(_request(tmp_path), tmp_path / "out.mp4")
    assert fal.requests == []


def test_a_model_without_a_last_frame_field_refuses_one(tmp_path):
    spec = selector.load_catalog()["minimax_h3_max"]
    provider = FalVideoProvider(spec, KEY)
    provider.params = {**provider.params, "end_frame_field": None}
    with pytest.raises(VideoGenerationError, match="last frame"):
        provider.payload(_request(tmp_path, end_image_path=make_photo(tmp_path / "e.jpg", (72, 128))))


def test_the_outpainter_sends_the_picture_and_the_margins_and_returns_the_answer(fal, tmp_path):
    import io

    answer = Image.new("RGB", (40, 60), (10, 200, 30))
    buffer = io.BytesIO()
    answer.save(buffer, "PNG")
    fal.png = buffer.getvalue()
    fal.result = (200, {"images": [{"url": f"{fal.url}/files/out.png"}]})
    painter = FalOutpainter(_client(fal), "fal-ai/flux-2-pro/outpaint", {"mode": "high", "output_format": "png"})
    got = painter(Image.new("RGB", (40, 20), (1, 2, 3)),
                  {"expand_top": 20, "expand_bottom": 20, "expand_left": 0, "expand_right": 0})
    assert got.size == (40, 60) and got.getpixel((5, 5)) == (10, 200, 30)
    body = fal.posts()[0]["body"]
    assert fal.posts()[0]["path"] == "/fal-ai/flux-2-pro/outpaint"
    assert body["image_url"].startswith("data:image/png;base64,") and body["expand_top"] == 20
    assert body["mode"] == "high" and painter.last_request_id == "req-1"
