"""fal.ai: image-to-video models (and the outpaint that keyframe.extend uses)
through fal's queue API, read 7 Nov 2026 (fal.ai/docs/model-apis/model-endpoints/queue):

    POST {queue}/{endpoint}                 the model's input as JSON   -> request_id, status_url, response_url
    GET  status_url                         IN_QUEUE | IN_PROGRESS | COMPLETED (error, error_type on a failure)
    GET  response_url                       the model's output, e.g. {"video": {"url": ...}}

with "Authorization: Key <FAL_KEY>". Status and result use the owner/app part of
the id only, so the URLs the submit returns are used as given.

Pictures go in as data URIs. fal documents uploads to its CDN through its SDKs
only (no HTTP procedure is published), the SDK would be a new dependency, and a
CDN file is public to anyone with the URL; a data URI is accepted by the model
endpoints and leaves nothing behind.

One entry of configs/video/providers.yaml per model; this adapter reads its
`endpoint` and `params`:

    params.input            sent as written (resolution, prompt_expansion_mode, cfg_scale ...)
    params.end_frame_field  the input that takes a last frame (end_image_url, tail_image_url)
    params.duration         {kind: seconds, min, overrun_s} or {kind: choices, choices: ["5", "10"]}
    params.negative_prompt  true when the model takes one
    params.billing          per_second (the clip's length) | per_clip (cost_per_clip for clip_s)

A submit costs money, so it is sent ONCE: a dropped connection or a 5xx on the
submit is not retried, because the job may have started. Polls and downloads are
retried. Errors:

    401, 403                                   VideoGenerationAuthError
    402, 429, or a body naming balance/credit  VideoGenerationQuotaError     never retried
    a failed job naming content or safety      VideoContentRefusedError
    5xx, timeout, dropped connection           VideoGenerationTransportError
    anything else                              VideoGenerationError
"""
from __future__ import annotations

import base64
import io
import logging
import math
import mimetypes
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.core.exceptions import RenderingError
from src.models.video_gen.base import (
    ImageToVideoRequest, ProviderSpec, VideoClip, VideoContentRefusedError, VideoGenerationAuthError,
    VideoGenerationError, VideoGenerationQuotaError, VideoGenerationTransportError, VideoProvider, check_request,
)
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

QUEUE_URL = "https://queue.fal.run"
CONNECT_TIMEOUT = 10.0
_NEEDS_PAYMENT = ("balance", "credit", "quota", "exhausted", "billing", "top up", "payment", "locked")
_REFUSED = ("content", "safety", "nsfw", "policy", "moderation", "flagged")
_WAITING = ("IN_QUEUE", "IN_PROGRESS")


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def file_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return data_uri(Path(path).read_bytes(), mime)


class FalClient:
    """fal's queue over requests: submit once, poll, fetch the result, download."""

    def __init__(self, api_key: str, queue_url: str = QUEUE_URL, poll_s: float = 3.0,
                 poll_timeout_s: float = 900.0, timeout_s: float = 120.0, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self._api_key = (api_key or "").strip()
        self._queue = queue_url.rstrip("/")
        self._poll_s, self._poll_timeout_s, self._timeout_s = poll_s, poll_timeout_s, timeout_s
        self._max_retries = max(1, max_retries)
        self._sleep, self._clock = sleep, clock

    def _headers(self) -> dict:
        return {"Authorization": f"Key {self._api_key}", "Content-Type": "application/json"}

    def _call(self, method: str, url: str, **kwargs):
        import requests

        try:
            return requests.request(method, url, timeout=(CONNECT_TIMEOUT, self._timeout_s), **kwargs)
        except Exception as exc:  # noqa: BLE001 - an unrecognised exception from an HTTP call is a failed call
            raise VideoGenerationTransportError(
                f"fal could not be reached or did not answer in time ({type(exc).__name__})") from exc

    @staticmethod
    def _raise_for(status: int, detail: str) -> None:
        low = detail.lower()
        if status in (402, 429) or (status in (400, 403, 422) and any(w in low for w in _NEEDS_PAYMENT)):
            raise VideoGenerationQuotaError(f"fal's balance or allowance is spent (HTTP {status}): {detail[:160]}")
        if status in (401, 403):
            raise VideoGenerationAuthError(f"fal rejected the key (HTTP {status}). Check FAL_KEY.")
        if status >= 500:
            raise VideoGenerationTransportError(f"fal is busy or failed (HTTP {status})")
        if any(w in low for w in _REFUSED):
            raise VideoContentRefusedError(f"fal declined the input (HTTP {status}): {detail[:160]}")
        raise VideoGenerationError(f"fal refused the request (HTTP {status}): {detail[:160]}")

    def _json(self, response) -> dict:
        status = getattr(response, "status_code", 0)
        if status in (200, 201, 202):
            try:
                return response.json()
            except Exception as exc:  # noqa: BLE001
                raise VideoGenerationError("fal answered with something that is not JSON") from exc
        self._raise_for(status, (getattr(response, "text", "") or "")[:400])
        return {}

    def _retried(self, fn):
        """Polls and downloads: a transport error is tried again; nothing else is."""
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn()
            except VideoGenerationTransportError:
                if attempt == self._max_retries:
                    raise
                self._sleep(self._poll_s * attempt)
        raise AssertionError("unreachable")

    # -- the steps ------------------------------------------------------------------

    def submit(self, endpoint: str, payload: dict) -> dict:
        """One POST. Never retried: the job may have started and is paid for."""
        if not self._api_key:
            raise VideoGenerationAuthError("FAL_KEY is not set")
        data = self._json(self._call("POST", f"{self._queue}/{endpoint}", headers=self._headers(), json=payload))
        if not data.get("request_id"):
            raise VideoGenerationError("fal did not name the request it queued")
        owner_app = "/".join(endpoint.split("/")[:2])
        base = f"{self._queue}/{owner_app}/requests/{data['request_id']}"
        data.setdefault("status_url", f"{base}/status")
        data.setdefault("response_url", base)
        return data

    def wait(self, submitted: dict) -> dict:
        started = self._clock()
        while True:
            status = self._retried(lambda: self._json(self._call("GET", submitted["status_url"],
                                                                 headers=self._headers())))
            state = str(status.get("status", "")).upper()
            if state == "COMPLETED":
                if status.get("error"):
                    self._job_failed(str(status.get("error")), str(status.get("error_type") or ""))
                return status
            if state not in _WAITING:
                raise VideoGenerationError(f"fal's job ended with status '{state or 'unknown'}'")
            if self._clock() - started > self._poll_timeout_s:
                raise VideoGenerationTransportError(f"fal's job was not done after {self._poll_timeout_s:.0f} s")
            self._sleep(self._poll_s)

    def _job_failed(self, error: str, error_type: str) -> None:
        text = f"{error_type} {error}".strip()
        if any(w in text.lower() for w in _REFUSED):
            raise VideoContentRefusedError(f"fal declined the input: {text[:200]}")
        if any(w in text.lower() for w in _NEEDS_PAYMENT):
            raise VideoGenerationQuotaError(f"fal's balance or allowance is spent: {text[:200]}")
        raise VideoGenerationError(f"fal's job failed: {text[:200]}")

    def result(self, submitted: dict) -> dict:
        response = self._retried(lambda: self._call("GET", submitted["response_url"], headers=self._headers()))
        status = getattr(response, "status_code", 0)
        if 400 <= status < 500 and status not in (401, 402, 403, 429):
            detail = (getattr(response, "text", "") or "")[:400]
            self._job_failed(detail, "")
        return self._json(response)

    def download(self, url: str) -> bytes:
        """A file fal made. Its CDN needs no key, so none is sent."""
        def fetch() -> bytes:
            if url.startswith("data:"):
                return base64.b64decode(url.split(",", 1)[1])
            response = self._call("GET", url)
            status = getattr(response, "status_code", 0)
            if status != 200:
                if status >= 500:
                    raise VideoGenerationTransportError(f"fal's file could not be downloaded (HTTP {status})")
                raise VideoGenerationError(f"fal's file could not be downloaded (HTTP {status})")
            return response.content
        return self._retried(fetch)

    def run(self, endpoint: str, payload: dict) -> tuple[dict, str]:
        """(the model's output, the request id)."""
        submitted = self.submit(endpoint, payload)
        logger.info("fal %s queued as %s", endpoint, submitted["request_id"])
        self.wait(submitted)
        return self.result(submitted), str(submitted["request_id"])


# ---------------------------------------------------------------------------
# Image to video
# ---------------------------------------------------------------------------

def _usable_endpoint(endpoint: str) -> bool:
    return bool(endpoint) and endpoint.upper() != "TODO"


class FalVideoProvider(VideoProvider):
    """Every fal video model; the providers.yaml entry says which and how."""

    id = "fal"

    def __init__(self, spec: Optional[ProviderSpec] = None, api_key: str = "", client: Optional[FalClient] = None):
        super().__init__(spec, api_key)
        self.endpoint = spec.endpoint if spec is not None else ""
        self.model_id = self.endpoint or "fal"
        self.params = dict(spec.params) if spec is not None else {}
        self.supports_first_last_frame = bool(self.params.get("end_frame_field")) and \
            (spec.supports_first_last_frame if spec is not None else True)
        self._client = client

    @property
    def client(self) -> FalClient:
        if self._client is None:
            self._client = FalClient(self._api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key) and _usable_endpoint(self.endpoint)

    # -- what is sent ---------------------------------------------------------------

    def duration_value(self, seconds: float):
        """The duration input for `seconds`: the model's own form (a number, or the shortest choice that holds it)."""
        rule = self.params.get("duration") or {"kind": "seconds"}
        if rule.get("kind") == "choices":
            choices = [str(c) for c in rule["choices"]]
            fits = [c for c in choices if float(c) >= seconds - 1e-6]
            return fits[0] if fits else choices[-1]
        return max(float(rule.get("min", 1)), min(float(seconds), self.max_duration_s))

    def billed_seconds(self, seconds: float) -> float:
        """The seconds a call for `seconds` is expected to be billed for, before it is made."""
        rule = self.params.get("duration") or {}
        value = self.duration_value(seconds)
        return float(value) + (float(rule.get("overrun_s", 0)) if rule.get("kind") != "choices" else 0.0)

    def estimate_call(self, seconds: float) -> Optional[float]:
        """USD for one call asking for `seconds`."""
        if self.spec is None:
            return None
        if self.params.get("billing") == "per_clip" and self.spec.cost_per_clip is not None and self.spec.clip_s:
            return round(self.spec.cost_per_clip * math.ceil(self.billed_seconds(seconds) / self.spec.clip_s), 4)
        rate = self._rate()
        return None if rate is None else round(rate * self.billed_seconds(seconds), 4)

    def _rate(self) -> Optional[float]:
        resolution = (self.params.get("input") or {}).get("resolution")
        if self.spec is not None and resolution and resolution in self.spec.cost_by_resolution:
            return float(self.spec.cost_by_resolution[resolution])
        return self.spec.rate_per_second if self.spec is not None else None

    def payload(self, request: ImageToVideoRequest) -> dict:
        body = dict(self.params.get("input") or {})
        body["prompt"] = request.prompt
        body["image_url"] = file_data_uri(request.image_path)
        body["duration"] = self.duration_value(request.duration_s)
        if request.end_image_path is not None:
            field = self.params.get("end_frame_field")
            if not field or not self.supports_first_last_frame:
                raise VideoGenerationError(f"{self.id} does not take a last frame")
            body[field] = file_data_uri(request.end_image_path)
        if request.negative_prompt and self.params.get("negative_prompt"):
            body["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            body["seed"] = request.seed
        return body

    # -- the provider contract ------------------------------------------------------------

    def generate(self, request: ImageToVideoRequest, output_path: Path) -> VideoClip:
        check_request(request)
        if not self._api_key:
            raise VideoGenerationAuthError("FAL_KEY is not set")
        if not _usable_endpoint(self.endpoint):
            raise VideoGenerationError(f"{self.id} has no fal endpoint set in providers.yaml")
        body = self.payload(request)
        output, request_id = self.client.run(self.endpoint, body)
        url = ((output.get("video") or {}).get("url")) if isinstance(output, dict) else None
        if not url:
            raise VideoGenerationError("fal finished without a video to download")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.client.download(url))
        try:
            info = ffmpeg.probe(output_path)
        except RenderingError as exc:
            raise VideoGenerationError(f"fal's video could not be read: {exc}") from exc
        billed = info.duration_s
        if self.params.get("billing") == "per_clip" and self.spec is not None and self.spec.clip_s:
            billed = float(self.duration_value(request.duration_s))
            cost = (self.spec.cost_per_clip or 0.0) * math.ceil(billed / self.spec.clip_s)
        else:
            rate = self._rate()
            cost = (rate or 0.0) * billed
        meta = {"request_id": request_id, "endpoint": self.endpoint, "billed_s": round(billed, 3),
                "asked_s": body["duration"], "camera": request.camera,
                "first_last": request.end_image_path is not None}
        if output.get("expanded_prompt"):
            meta["expanded_prompt"] = output["expanded_prompt"]
        return VideoClip(path=output_path, provider=self.id, model_id=self.model_id, duration_s=info.duration_s,
                         width=info.width, height=info.height, fps=int(round(info.fps or 0)),
                         cost_usd=round(cost, 4), meta=meta)


# ---------------------------------------------------------------------------
# Outpaint (keyframe.extend)
# ---------------------------------------------------------------------------

class FalOutpainter:
    """keyframe.Outpaint on a fal endpoint: the picture in, the picture with its edges continued out."""

    def __init__(self, client: FalClient, endpoint: str, params: Optional[dict] = None):
        self.client, self.endpoint, self.params = client, endpoint, dict(params or {})
        self.last_request_id = ""

    def payload(self, image: Image.Image, expand: dict) -> dict:
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return {**self.params, "image_url": data_uri(buffer.getvalue(), "image/png"),
                **{k: int(v) for k, v in expand.items()}}

    def __call__(self, image: Image.Image, expand: dict) -> Image.Image:
        output, self.last_request_id = self.client.run(self.endpoint, self.payload(image, expand))
        images = output.get("images") or []
        url = images[0].get("url") if images and isinstance(images[0], dict) else None
        if not url:
            raise VideoGenerationError("fal's outpaint finished without a picture")
        try:
            with Image.open(io.BytesIO(self.client.download(url))) as handle:
                return handle.convert("RGB")
        except VideoGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - an unreadable answer is a failed call
            raise VideoGenerationError(f"fal's outpaint answer could not be read: {exc}") from exc
